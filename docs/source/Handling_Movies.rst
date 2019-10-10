Handling movies
================

This page gives a basic intro on the movie object for loading, manipulating,
displaying and saving movies within CaImAn.

Loading movies
--------------

Movies can be loaded either individually or as a set.

To load a single movie:

.. code:: python

   import caiman as cm
   single_movie = cm.load('example_movies/demoMovie.tif')
   print(single_movie.shape)

To load multiple movies and display them in sequence:

.. code:: python

   import caiman as cm
   file_names = ['example_movies/demoMovie.tif', 'example_movies/demoMovie.tif'] # for the sake of the example we repeat the same movie
   movies_chained = cm.load_movie_chain(file_names)
   print(movies_chained.shape)

One can specify several parameters while loading. For instance frame
rate, or if only some portion of the movies needs to be loaded, and so
on. Check the documentation.

Both functions returns a movie object. The movie object can also be
constructed giving as input one 3D array (time x x_dimension x
y_dimension). Example.

.. code:: python

   import caiman as cm
   movie_random = cm.movie(np.random.random([1000,100,100]))

Saving movies
-------------

Movies can be saved in several different formats (.mat, .tif, .hdf5,
etc). In order to save just call the save command with the appropriate
file extension.

.. code:: python

   movie_random.save('movie_random.tif')

It is also possible to save in a memory mappable format. This is an
advanced topic that is dealt with in the demos in the root folder.

Visualizing movies
------------------

One can very efficiently play movies with the play function. The play
function has options to modulate the exposure, the magnification, the
playing frame rate, and adjusting contrast by setting the quantiles `q_min`
and `q_max` that correspond to the minimum and maximum values being displayed
(sometimes movies have weird range of values, making this normalization necessary
for good visualization)

Example:

.. code:: python

   movies_chained.play(magnification = 2, fr=30, q_min=0.1, q_max=99.75)

Playback of a movie can be interrupted by pressing `q`.

Movie objects are stored as numpy arrays and standard operations can be applied:

.. code:: python

   import matplotlib.pyplot as plt
   plt.imshow(np.mean(movies_chained,0))
   plt.imshow(np.std(movies_chained,0))
   plt.plot(np.mean(movies_chained, axis = (1,2)))

In this sense it is also very convenient the correlation image

.. code:: python

   CI = movies_chained.local_correlations(eight_neighbours=True, swap_dim=False)
   pl.imshow(CI)

This supposes that your movie is stored is represented in T x X x Y format. If the
time dimension is last, then use `swap_dim=True`

Manipulating movies
-------------------

concatenation
~~~~~~~~~~~~~

Movie objects behave like a numpy array. They can be summed, multiplied,
divided, etc… This behavior is very versatile. The are only a few
functions that cannot be implemented as an array, for instance
concatenation. For that operation there is a special function,
cm.concatenate:

.. code:: python

   movies_chained = cm.concatenate([movie1, movie2] , axis=0)

This will concatenate movie1 and movie2 along the time axis. Note that the axis
ordering here is T x X x Y

movie resizing
~~~~~~~~~~~~~~

Sometimes it is useful to downsample or upsample the movies across some
dimensions. We have implemented an efficient way of doing so, based on
the opencv library. Below an example putting it all together:

.. code:: python

   movies_chained = cm.concatenate([movie1, movie2] , axis = 1).resize(1,1,.5).play(magnification=2, fr=50)

This command will concatenate `movie1` and `movie2` along axis `x`, then it will
downsample the resulting movie along the temporal axis by a factor of 2, and finally it
will play the resulting movie magnified by a factor of 2.

Noe that unlike `cm.concatenate`, for `movie.resize` the axis ordering is
X x Y x T (time appears in the last dimension).
