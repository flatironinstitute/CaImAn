Overview
=========
Python translation of Constrained Non-negative Matrix Factorization algorithm for source extraction from calcium imaging data. 


The code implements a method for simultaneous source extraction and spike inference from large scale calcium imaging movies. The code is suitable for the analysis of somatic imaging data. Implementation for the analysis of dendritic/axonal imaging data will be added in the future. 

The algorithm is presented in more detail in

Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037

Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. http://arxiv.org/abs/1409.2903

Contributors
------------

Andrea Giovannucci and 
Eftychios Pnevmatikakis 

Center for Computational Biology, Simons Foundation, New York, NY




Questions, comments, issues
-----------------------------
Please use the gitter chat room (use the button above) for questions and comments and create an issue for any bugs you might encounter.

Important note
----------------
The implementation of this package is based on the matlab implementation which can be found [here](https://github.com/epnev/ca_source_extraction). Some of the Matlab features are currently lacking, but will be included in future releases. 

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
