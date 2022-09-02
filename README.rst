====================
Record-breaking heat
====================

This repository contains the code associated with the analyses in McKinnon and Simpson (2022), How unexpected was the record breaking PNW heatwave?, currently under review at GRL. Preprint is available at https://www.essoar.org/doi/abs/10.1002/essoar.10511999.1.

The analysis relies on four different datasets which must be acquired in advance of running the code. I have provided scripts to do so:

* For CESM2-LE, scripts/save_cesm2.py can be run on a computer with access to glade campaign storage.
* For GHCND, scripts/get_ghcnd.py can be used to download and process GHCND data.
* For ISD, https://github.com/karenamckinnon/helpful_utilities/blob/master/helpful_utilities/download_isd.py can be used to download and process ISD. 
* For EC, data should first be bulk downloaded with scripts/get_BC_data.sh, then processed with scripts/get_EC.py

In most cases, (my) directories are hard-coded in, so will need to be updated appropriately. 

The easiest way to interact with the code is likely with the jupyter notebook in notebooks/GRL_figures, but the same code is provided in script form in GRL_figures.py.

Please be in touch (kmckinnon@ucla.edu) if you use the code, or have questions/comments.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
