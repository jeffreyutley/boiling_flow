.. docs-include-ref

Boiling Flow
============

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images. Full documentation is available at
https://boiling-flow.readthedocs.io .

Installing
----------
1. Clone or download the repository:

    .. code-block::

        git clone git@github.com:jeffreyutley/boiling_flow

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        Create a conda environment ``boiling_flow`` using the ``environment.yml`` file.

        .. code-block::

            conda env create -f environment.yml

        Anytime you want to use this package, this ``boiling_flow`` environment should be activated with the following:

        .. code-block::

            conda activate boiling_flow


Running Demo(s)
---------------

There are three demo scripts: ``parameter_estimates_from_measured_data.py``, ``generate_phase_screen_data.py``, and
``results_from_simulated_data.py``. The former two scripts show an example of the boiling flow algorithm on measured
data sets, while the latter first generates simulated data and then runs the boiling flow algorithm. The
``results_from_simulated_data.py`` script can be run without downloading any external data sets.

Before running the former two demo scripts, download the measured data sets:

    Option 1. Install using shell script

        Use the script ``get_demo_data_server.sh`` inside of the ``demo`` folder to automatically install the data and
        place it in the proper folder for the scripts ``parameter_estimates_from_measured_data.py`` and
        ``generate_phase_screen_data.py``.

        Inside of the parent directory (the boiling_flow folder containing this file), run the following:

        .. code-block::

            source demo/get_demo_data_server.sh

    Option 2. Manual install

        To manually install the data sets, visit the
        `Bouman data repository <https://www.datadepot.rcac.purdue.edu/bouman/>` and download the .zip file
        ``TBL_data.zip``.

        Unzip the file and place the folder ``TBL_data`` inside of the ``data/demo`` directory.

Run any of the demo scripts from the parent directory (the boiling_flow folder containing this file) with the following
command:

    .. code-block::


        python demo/demo_file.py

The script ``generate_phase_screen_data.py`` loads an .npy file that is saved by
``parameter_estimates_from_measured_data.py``, so the latter script must be run before the former.


Disclaimer
----------

Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2025-5580.
