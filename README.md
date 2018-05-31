# clustering_superdarn_data

### Setup instructions:

This project was written in Python 3.5.2 on Ubuntu 16.04.

### Ubuntu setup:

Make sure the Python3 and Python3 tkinter package is installed. This is required for matplotlib. 

`sudo apt-get install python3-tk`

Clone the repo and navigate to the root directory.

Optional step: Create a virtual environment using:

`virtualenv -p /usr/bin/python3 venv`

`source venv/bin/activate`

Install requirements using Pip (if python2 is your default, make sure to use pip3 and python3 commands):

`pip install -r requirements.txt`

Now you can run the files (again, use python3 here if python2 is your default:

`python plot_feature_selection.py`

### Windows setup

Not tested. Anaconda may be useful for Windows setup, as it contains many of the packages we use.
