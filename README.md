# Photon Count
Given an input stack of images, PhotonCount will return a photon-counted average image and can also output the counts corrected for threshold and coincidence loss.

## Getting Started

### Installing

This package requires Python 3.6 or higher. To install PhotonCount, navigate to the PhotonCount directory where setup.py is located and use

	pip install .

This will install photon_count and its dependencies, which are as follows:

* numpy
* mpmath


### Usage

For an example of how to use PhotonCount, see example_script_pc.py (which utilizes the emccd_detect package, https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect).

The example script of proc_cgi_frame (https://github.jpl.nasa.gov/WFIRST-CGI/proc_cgi_frame) utilizes this package.


## Authors

* Bijan Nemati (<bijan.nemati@uah.edu>)
* Sam Miller (<sam.miller@uah.edu>)
* Kevin Ludwick (<kjl0025@uah.edu>)
