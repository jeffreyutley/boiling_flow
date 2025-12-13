.. docs-include-ref

boiling_flow
============

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images.

Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

Installing
----------
1. *Clone or download the repository:*

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

            Anytime you want to use this package, this ``boiling_flow`` environment should be activated with the following:

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

After downloading the data files, run a demo script from the parent directory (the boiling_flow folder containing this
file) with the following command:

    .. code-block::

        python demo/demo_file.py