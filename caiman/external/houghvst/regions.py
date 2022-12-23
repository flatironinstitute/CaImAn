import numpy as np
from skimage.util import view_as_windows


def patch_selection(img, n_patches, patch_size=32, overlapping=False):
    shape = (img.shape[0] - patch_size, img.shape[1] - patch_size)
    idx = np.arange(np.prod(shape))

    if not overlapping:
        starting_points = np.ones(shape, dtype=bool)
        available = np.ones(shape, dtype=bool)

        patch_idx = []
        patches = []
        masks = []
        while len(patches) < n_patches:
            if not np.any(starting_points):
                print('Could only produce {} patches ' 'out of the requested {}'.format(len(patches), n_patches))
                break

            k = np.random.choice(idx[starting_points.flat])
            k0, k1 = np.unravel_index(k, shape)

            sl0 = slice(k0, np.minimum(k0 + patch_size, shape[0]))
            sl1 = slice(k1, np.minimum(k1 + patch_size, shape[1]))
            if not np.all(available[sl0, sl1]):
                starting_points[k0, k1] = False
                continue

            available[sl0, sl1] = False

            sl0 = slice(np.maximum(k0 - patch_size + 1, 0), np.minimum(k0 + patch_size, shape[0]))
            sl1 = slice(np.maximum(k1 - patch_size + 1, 0), np.minimum(k1 + patch_size, shape[1]))
            starting_points[sl0, sl1] = False

            patch_idx.append((k0, k1))
            patches.append(img[k0:k0 + patch_size, k1:k1 + patch_size])
            m = np.zeros(img.shape, dtype=bool)
            m[k0:k0 + patch_size, k1:k1 + patch_size] = True
            masks.append(m)

        patch_idx = np.array(patch_idx)
        patches = np.array(patches)
        masks = np.array(masks)
        labels = sum([m * (i + 1) for i, m in enumerate(masks)])
        return patch_idx, patches, masks, labels
    else:
        k = np.random.choice(idx, size=n_patches)
        patch_idx = np.vstack(np.unravel_index(k, shape)).T
        patches = [img[k0:k0 + patch_size, k1:k1 + patch_size] for k0, k1 in patch_idx]
        patches = np.array(patches)
        return patch_idx, patches


def im2col(img, window_size, stride=1):
    arr = view_as_windows(img, window_size)
    arr = arr[::stride, ::stride, :, :]
    arr = arr.reshape(np.prod(arr.shape[:-2]), window_size, window_size)
    return arr
