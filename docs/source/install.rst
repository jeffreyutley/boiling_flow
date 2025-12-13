============
Installation
============

The ``boiling_flow`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/jeffreyutley/boiling_flow.git``
    | ``cd boiling_flow``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

    | ``conda create --name boiling_flow python=3.11``
    | ``conda activate boiling_flow``

  Install the dependencies using:

    ``pip install -r requirements.txt``

  Install the package using:

    ``pip install .``

  or to edit the source code while using the package, install using

    ``pip install -e .``

  Now to use the package, this ``boiling_flow`` environment needs to be activated.


3. Install:

You can verify the installation by running ``pip show boiling_flow``, which should display a brief summary of the packages installed in the ``boiling_flow`` environment.
Now you will be able to use the ``boiling_flow`` python commands from any directory by running the python command ``import boiling_flow``.

