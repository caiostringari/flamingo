.. Flamingo documentation master file, created by
   sphinx-quickstart on Mon Dec 15 11:57:04 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Flamingo's documentation!
====================================

The Flamingo toolbox is an open-source toolbox for image segmentation, classification and rectification.
It is developed by the Department of Hydraulic Engineering of Delft University of Technology for coastal image analysis.
The toolbox is built around the *scikit-image*, *scikit-learn*, *OpenCV* and *pystruct* toolboxes.

Flamingo is developed and maintained by:

| Bas Hoonhout <b.m.hoonhout@tudelft.nl>
| Max Radermacher <m.radermacher@tudelft.nl>

The toolbox can be found at `<http://github.com/openearth/flamingo>`_.

Contents
--------

.. toctree::
   :maxdepth: 2

   rectification
   segmentation
   classification
   calibration

   
Command-line tools
------------------

Several command-line functions are supplied with the toolbox for batch processing of large datasets.
Each command-line function serves a specific part of the image analysis.
See for more information the *--help* option of each command.

rectify-images
^^^^^^^^^^^^^^

.. automodule:: rectify
                :members:

classify-images
^^^^^^^^^^^^^^^

.. automodule:: classify
                :members:

calibrate-camera
^^^^^^^^^^^^^^^^

.. automodule:: calibrate
                :members:
                   

File system
-----------

The toolbox uses a file system structure for the analysis of datasets.
The :mod:`filesys` module takes care of any reading and writing of
files in this file structure.  Each dataset is stored in a single
directory and can consist out of the following file types:

Image files
  Any image file recognized by the system
Cropped image files
  Names start with *cropped_*. A non-cropped version of the image file should exist.
Export files
  Pickle files with data concerning an image. Each export file name has the following format: *<image_name>.<key>.pkl*.
  A special type of export file is the feature file. Not all features are written to a single export file, but they
  are subdivided into multiple export files depending on the feature block they belong to. The block name is added to the
  export file, just before the file extension.
Log files
  Pickle files with data concerning the entire dataset. Log file names can have any name.
Model files
  Pickle files with a trained model. Each model file is accompanied by a meta file. Each model file name has the following format:
  *model_<model_type>_<dataset>_I<nr_of_images>_B<nr_of_blocks>_<timestamp>.pkl*. The corresponding meta file has *meta* added to
  the name, just before the file extension.

.. automodule:: filesys
                :members:

                   
Configuration
-------------

Only the very basic options of the toolbox are exposed through the
command-line functions.  For the full extent of options a
configuration file is used. This configuration file is parsed by the
:mod:`config` module.  The module also supplies wrappers for
the automated updating of a function call based on the configuration
file used.

.. automodule:: config
                :members:

Example configuration
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: example.cfg


Acknowledgements
================

The Flamingo toolbox is developed at `Delft University of Technology <http://www.tudelft.nl>`_ with support from the ERC-Advanced Grant 291206 -- Nearshore Monitoring and Modeling (`NEMO <http://nemo.citg.tudelft.nl>`_), STW Grant 12686 -- Nature-driven
Nourishments of Coastal Systems (NatureCoast), S1: Coastal safety and `Deltares <http://www.deltares.nl>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

