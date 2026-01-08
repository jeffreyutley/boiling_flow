.. docs-include-ref

Boiling Flow
============

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images.

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

        1. *Create conda environment:*

            Create a new conda environment named ``boiling_flow`` using the following commands:

            .. code-block::

                conda create --name boiling_flow python=3.11
                conda activate boiling_flow
                pip install -r requirements.txt

            Anytime you want to use this package, this ``boiling_flow`` environment should be activated with the
            following:

            .. code-block::

                conda activate boiling_flow


        2. *Install boiling_flow package:*

            Navigate to the main directory ``boiling_flow/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .


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
``parameter_estimates_from_measured_data.py``, so the former script must be run before the latter.


Disclaimer
----------

Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.