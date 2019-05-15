Overview
=========

.. image:: ../LOGOS/Caiman_logo_FI.png
    :width: 300px
    :align: right

CaImAn is a Python toolbox for large scale **Ca**\ lcium **Im**\ aging data **An**\ alysis and behavioral analysis.

CaImAn implements a set of essential methods required in the analysis pipeline of large scale calcium imaging data. Fast and scalable algorithms are implemented for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both batch and online modes. CaImAn also contains some routines for the analysis of behavior from video cameras. A list of features as well as relevant references can be found `here
<https://github.com/flatironinstitute/CaImAn/wiki/CaImAn-features-and-references>`_.

Companion paper
--------------

A paper explaining most of the implementation details and benchmarking can be found `here
<https://elifesciences.org/articles/38173>`_.

::

  @article{giovannucci2019caiman,
    title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
    author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
    journal={eLife},
    volume={8},
    pages={e38173},
    year={2019},
    publisher={eLife Sciences Publications Limited}
  }


Developers/Contributors
------------

CaImAn is being developed at the `Flatiron Institute <https://www.simonsfoundation.org/flatiron/>`_ with numerous contributions from the broader community. The main developers are

* Eftychios A. Pnevmatikakis, Flatiron Institute
* Andrea Giovannucci, University of North Carolina at Chapel Hill, previously at Flatiron Institute
* Johannes Friedrich, Flatiron Institute
* Pat Gunn, Flatiron Institute

A complete list of contributors can be found `here <https://github.com/flatironinstitute/CaImAn/graphs/contributors>`_.


Questions, comments, issues
-----------------------------

Please use our `gitter chat room <https://gitter.im/agiovann/Constrained_NMF>`_ for questions and comments and create an issue on our `repo page <https://github.com/flatironinstitute/CaImAn>`_ for any bugs you might encounter.

License
--------

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
