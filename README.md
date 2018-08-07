# clustering_superdarn_data

### Setup instructions:

This project was written in Python 3.5 and Python 3.6 on Ubuntu 16.04 and Ubuntu 18.04.

### Ubuntu setup:

Make sure the Python3 and Python3 tkinter package is installed. This is required for matplotlib. 

`sudo apt-get install python3-tk`

Clone the repo and navigate to the root directory.

Optional: Create a virtual environment using:

`virtualenv -p /usr/bin/python3 venv`

`source venv/bin/activate`

Install these dependencies using Pip (if python2 is your default, make sure to use pip3 command):

`matplotlib`
`scipy`
`numpy`
`sklearn` 
`pillow`

Now you can run the files.

For a demonstration of how it works, see this iPython notebook:

`cluster.ipynb`


### Windows setup

Not tested. Anaconda may be useful for Windows setup, as it contains many of the packages we use.
