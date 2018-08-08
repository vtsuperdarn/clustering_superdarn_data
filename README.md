# clustering_superdarn_data

We are developing new models for classifying SuperDARN 
(Super Dual Auroral Radar Network) data using machine learning algorithms.
In the past, this data has been classified point-by-point using a 
quadratic formula based on doppler velocity and spectral width. 
Recently, researchers successfully applied unsupervised clustering 
techniques to this data. These approaches improved on past methods, but they used a 
very limited set of features to create clusters and relied on simple 
methods (k-means, depth-first search) that do not easily capture 
non-linear relationships or subtle probability distributions. 

This project applies DBSCAN and Gaussian Mixture Model (GMM) to the data, and provides a library
with different models and classification thresholds which can be used on SuperDARN data. 
Depending on characteristics of the data the user wants to study, different models, parameters,
and thresholds may be suitable. For example, the Ribiero threshold is best for mid-latitude radars, and the Blanchard
thresholds are best for high-latitude. See below for more details about the individual models
and thresholds.

Google Summer of Code 2018 project link:

https://summerofcode.withgoogle.com/projects/#5795870029643776

## Algorithms
#### GMM
GMM runs on 5 features by default: beam, gate, time, velocity, and spectral width.
It performs well overall, even on clusters that are not well-separated in space and time.
However, it will often create clusters that are too high variance,causing it to pull in
scattered points that do not look like they should be clustered together - see the
fanplots in cluster.ipynb. It is also slow, taking 5-10 minutes for one day of data.

#### DBSCAN
DBSCAN runs on 3 features: beam, gate, and time (space and time).
It can classify clusters that are well-separated in space in time,
but will not perform well on mixed scatter. It uses sklearn's implementation
of DBSCAN, which is highly optimized, so it runs in ~10s on 1 day of data.

#### DBSCAN + GMM
Applies DBSCAN on the space-time features, then applies GMM to 
separate clusters based on velocity and width. Unlike pure DBSCAN, it can identify
mixed scatter. It is also much faster than GMM, running in ~15-60s on a full day of data.

#### GBDBSCAN
Grid-based DBSCAN is a modification of regular DBSCAN designed for automotive radars.
It assumes close objects will appear wider, and distant objects will appear
narrower, and varies the search area accordingly. See Kellner et al. 2012.
It is not yet clear whether this assumption is advantageous for SuperDARN data.

My implementation of GBDBSCAN has not been optimized to the extent sklearn's DBSCAN 
has, so it takes 5-10 minutes, but there is room for improvement. So far,
it provides similar performace to DBSCAN, but creates less small clusters at close
ranges due to its wide search area.

#### GBDBSCAN + GMM
Applies GBDBSCAN on the space-time features, then applies GMM to 
separate clusters based on velocity and width. Takes 5-10 minutes. Not yet
clear if it's any better than DBSCAN + GMM.

## Classification thresholds

#### Blanchard paper
This is the 'traditional' point-by-point classification method, developed in Blanchard 2009
for high-latitude radars. We apply it to the median values of a cluster instead of to
one point at a time.

|vel| < 33.1 + 0.139 * |width| - 0.00133 * |width|^2

#### Blanchard code
This is the classification threshold used in the RST library, and it is credited there to
Blanchard et al., but we don't know why this is used instead.

|vel| < 30 - 1/3 |width|

#### Ribiero
This classificaion method was developed for mid-latitude radars, and applied on clusters created
using a depth-first search over space and time [Ribiero 2011]. Clusters are classified based on
their time duration (L, hours) and the ratio (R) of high:low velocity scatter points in the cluster.
See Ribiero 2011 Figure 4 for the full flowchart.

## Setup instructions

This project was written in Python 3.5 and Python 3.6 on Ubuntu 16.04 and Ubuntu 18.04.

#### Ubuntu setup:

Make sure the Python3 and Python3 tkinter package is installed. This is required for matplotlib. 

`sudo apt-get install python3-tk`

Install these dependencies using Pip (if python2 is your default, make sure to use pip3 command):

`matplotlib`
`scipy`
`numpy`
`sklearn` 
`pillow`
`jupyter`

Now you can run the files using Python 3.

For a demonstration of how to run the algorithms and produce plots, see this iPython notebook:

`cluster.ipynb`

#### Windows setup

Not tested. Anaconda may be useful for Windows setup, as it contains many of the packages we use.
