#!/usr/bin/env

from abc import ABC, abstractmethod
from itertools import product as iter_product
from pathlib import Path
from time import time
from typing import *
from warnings import warn


import h5py
import numpy as np
from scipy.sparse import csc_matrix


# TODO: Is there a better way to do this?
slice_or_int_or_range = Union[int, slice, range]


class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """
    @property
    def dtype(self) -> np.dtype:
        """
        np.dtype: numpy dtype
        """
        # compute the slice at the first index and return the dtype of that
        # TODO: there might be a nicer way to do this
        return self[0].dtype

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]: (n_frames, dims_x, dims_y)
        """
        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """
        int: number of frames
        """
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        """
        float: min value of the array if it were fully computed
        """
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        """
        float: max value of the array if it were fully computed
        """
        pass

    @property
    def ndim(self) -> int:
        """
        int: Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int: number of bytes required for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,))

    @property
    def nbytes_gb(self) -> float:
        """
        float: number of gigabytes required for the array if it were fully computed
        """
        return self.nbytes / 1e9

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Must be implemented in sublcass. Lazy computation logic goes here. Computes the array at the desired indices.

        Args:
            indices (Union[int, slice]): the user's desired slice, i.e. slice object or int passed from `__getitem__()`

        Returns:
            np.ndarray: slice of the array at the desired indices

        """
        pass

    def as_numpy(self):
        """
        Converts to a standard numpy array in RAM.

        NOT RECOMMENDED, THIS COULD BE EXTREMELY LARGE. Check ``nbytes_gb`` first!

        Returns:
            np.ndarray: full dense array in RAM
        """

        warn(
            f"\nYou are trying to create a numpy.ndarray from a LazyArray, "
            f"this is not recommended and could take a while.\n\n"
            f"Estimated size of final numpy array: "
            f"{self.nbytes_gb:.2f} GB"
        )
        dense = np.zeros(shape=self.shape, dtype=self.dtype)

        for i in range(self.n_frames):
            dense[i] = self[i]

        return dense

    def save_hdf5(self, filename: Union[str, Path], dataset_name: str = "data"):
        """
        Save the full dense array as an hdf5 file.

        **Note: This could result in a very large file, use ``nbytes_gb`` to check the
        size of the full dense array**

        Args:
            filename:
            dataset_name:

        Returns:

        """
        pass

    def __getitem__(
            self,
            item: Union[int, Tuple[slice_or_int_or_range]]
    ):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, range, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of <{start}> "
                                     f"lies beyond `n_frames` <{self.n_frames}>")
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of <{stop}> "
                                     f"lies beyond `n_frames` <{self.n_frames}>")

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    def __repr__(self):
        return f"{self.__class__.__name__} @{hex(id(self))}\n" \
               f"{self.__class__.__doc__}\n" \
               f"Frames are computed only upon indexing\n" \
               f"shape [frames, x, y]: {self.shape}\n"


class LazyArrayRCM(LazyArray):
    """LazyArray for reconstructed movie, i.e. A ⊗ C"""
    def __init__(
            self,
            spatial: Union[np.ndarray, csc_matrix],
            temporal: np.ndarray,
            frame_dims: Tuple[int, int],
    ):
        """
        Construct a Lazy Array of the reconstructed movie ``A ⊗ C``
        using the spatial and temporal components from CNMF.

        Args:
            spatial (np.ndarray or csc_matrix): ``A``, spatial components as a dense (np.ndarray)
             or Compressed Sparse Column matrix (csc_matrix)

            temporal (np.ndarray): ``C``, temporal components

            frame_dims (Tuple[int, int]): frame dimensions

        """

        if spatial.shape[1] != temporal.shape[0]:
            raise ValueError(
                f"Number of temporal components provided: `{temporal.shape[0]}` "
                f"does not equal number of spatial components provided: `{spatial.shape[1]}`"
            )

        self._spatial = spatial
        self._temporal = temporal

        self._shape: Tuple[int, int, int] = (temporal.shape[1], *frame_dims)

        # precompute min and max vals for each component in spatial and temporal domain
        temporal_max = np.nanmax(self.temporal, axis=1)
        temporal_min = np.nanmin(self.temporal, axis=1)

        # to get the component-wise min max of spatial, it must be converted to a dense matrix first
        if isinstance(self.spatial, csc_matrix):
            spatial_max = self.spatial.max(axis=0).toarray()
            spatial_min = self.spatial.min(axis=0).toarray()
        else:
            spatial_max = self.spatial.max(axis=0)
            spatial_min = self.spatial.min(axis=0)

        # get the min and max of every component, and then get the min and max of the entire array
        prods = list()
        for t, s in iter_product([temporal_min, temporal_max], [spatial_min, spatial_max]):
            _p = np.multiply(t, s)
            prods.append(np.nanmin(_p))
            prods.append(np.nanmax(_p))

        self._max = np.max(prods)
        self._min = np.min(prods)

        # get the mean of each temporal component
        temporal_mean = np.nanmean(self.temporal, axis=1)

        # get the standard deviation of each temporal component
        temporal_std = np.nanstd(self.temporal, axis=1)

        # compute mean, max, min, and std projection images
        self._mean_image = (self.spatial @ temporal_mean).reshape(frame_dims, order="F")
        self._max_image = (self.spatial @ temporal_max).reshape(frame_dims, order="F")
        self._min_image = (self.spatial @ temporal_min).reshape(frame_dims, order="F")
        self._std_image = (self.spatial @ temporal_std).reshape(frame_dims, order="F")

    @property
    def spatial(self) -> np.ndarray:
        return self._spatial

    @property
    def temporal(self) -> np.ndarray:
        return self._temporal

    @property
    def n_components(self) -> int:
        return self._spatial.shape[1]

    @property
    def n_frames(self) -> int:
        return self._temporal.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def mean_image(self) -> np.ndarray:
        """mean projection image"""
        return self._mean_image

    @property
    def max_image(self) -> np.ndarray:
        """max projection image"""
        return self._max_image

    @property
    def min_image(self) -> np.ndarray:
        """min projection image"""
        return self._min_image

    @property
    def std_image(self) -> np.ndarray:
        """standard deviation projection image"""
        return self._std_image

    def _compute_at_indices(self, indices: Union[int, Tuple[int, int]]) -> np.ndarray:
        rcm = (self.spatial @ self.temporal[:, indices]).reshape(
            self.shape[1:] + (-1,), order="F"
        ).transpose([2, 0, 1])

        if rcm.shape[0] == 1:
            return rcm[0]  # 2d single frame
        else:
            return rcm

    def __repr__(self):
        r = super().__repr__()
        return f"{r}" \
               f"n_components: {self.n_components}"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot compute equality against types that are not {self.__class__.__name__}")

        if (self.spatial == other.spatial) and (self.temporal == other.temporal):
            return True
        else:
            return False


# implementation for reconstructed background is identical
# this is just a subclass to separate them
# TODO: Is this really necessary? Maybe we just make one class called "Reconstructed Array"
#  and they keep track of spatial and temporal components that the construct them with?
class LazyArrayRCB(LazyArrayRCM):
    """Lazy array for reconstructed background, i.e. b ⊗ f"""


class LazyArrayResiduals(LazyArray):
    """Lazy array for residuals, i.e. Y - (A ⊗ C) - (b ⊗ f)"""
    def __init__(
            self,
            raw_movie: np.memmap,
            rcm: LazyArrayRCM,
            rcb: LazyArrayRCB,
            raw_movie_mean_projection: Optional[np.ndarray] = None,
            raw_movie_max_projection: Optional[np.ndarray] = None,
            raw_movie_min_projection: Optional[np.ndarray] = None,
    ):
        """
        Construct a LazyArray of the residuals, ``Y - (A ⊗ C) - (b ⊗ f)``.

        If projections of the raw movie are provided, then projections of the residuals will also be computed.

        Args:
            raw_movie (np.memmap): ``Y``, numpy memmap of the raw movie

            rcm (LazyArrayRCM): ``A ⊗ C``, reconstructed movie lazy array

            rcb (LazyArrayRCB): ``b ⊗ f``, reconstructed background lazy array

            timeout (float): number of seconds allowed for min max calculation of raw video

            raw_movie_mean_projection (np.ndarray): optional, mean projection of the raw movie

            raw_movie_max_projection (np.ndarray): optional, max projection of the raw movie

            raw_movie_min_projection (np.ndarray): optional, min projection of the raw movie

        """
        self._raw_movie = raw_movie
        self._rcm = rcm
        self._rcb = rcb

        # shape of the residuals will be the same as the shape of the raw movie
        self._shape = self._raw_movie.shape

        if raw_movie_mean_projection is not None:
            self._mean_image = raw_movie_mean_projection - self._rcm.mean_image - self._rcb.mean_image

        if raw_movie_max_projection is not None:
            self._max_image = raw_movie_max_projection - self._rcm.max_image - self._rcb.max_image
            self._max = raw_movie_max_projection.max() - self._rcm.max - self._rcb.max

        if raw_movie_min_projection is not None:
            self._min_image = raw_movie_max_projection - self._rcm.min_image - self._rcb.min_image
            self._min = raw_movie_min_projection.min() - self._rcm.min - self._rcb.min

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def n_frames(self) -> int:
        return self._shape[0]

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def mean_image(self) -> np.ndarray:
        return self._mean_image

    @property
    def max_image(self) -> np.ndarray:
        return self._max_image

    @property
    def min_image(self) -> np.ndarray:
        return self._min_image

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        residuals = self._raw_movie[indices] - self._rcm[indices] - self._rcb[indices]
        return residuals
