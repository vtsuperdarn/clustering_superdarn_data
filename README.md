# clustering_superdarn_data

Setup instructions:

First, make sure the Python3 tkinter package is installed. This is required for matplotlib plots.
> sudo apt-get install python3-tk


### To use in the virtual environment:
> source venv/bin/activate

> python plot_gmm_vs_empirical.py

To leave the virtual environment:
> deactivate


### To use with a local installation of Python 3
Install these dependencies if needed:

> pip3 install sklearn

> pip3 install scipy

> pip3 install numpy

> pip3 install matplotlib

Then run:
> python3 plot_gmm_vs_empirical.py
