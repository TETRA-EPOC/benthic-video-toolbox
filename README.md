# Benthic Video Toolbox
This is a standalone GUI application for pre and post processing of benthic video annotations with biigle. It has been developped as part of the FUTURE-OBS research project which aims to find innovative observation methods for coastal socio-ecosystem and is based on videos taken with the towed benthic sled **Pagure** developped by Ifremer. However it can be used and extended for other acquisition systems.

There are three main objectives to video annotations post-processing:
  1. Comprehensive biodiversity monitoring
  2. Measuring the size of the organisms observed
  3. Building a catalog of annotated images to train an automatic detection model

# Application main features
## Data pre-processing
  - Video trimming manually or according to Pagure's navigation file 
  - Conversion of Pagure's navigation file to a metadata file compatible with biigle  
  - Laserpoints detection inside video, for area estimation and size measurement of annotated organisms
## Data post-processing
  - Conversion of a Biigle video annotation file to YOLO formatted files in order to buid the annotated images database
  - Adding GPS coordinates (latitude and longitude) to Biigle's video annotation file


# Installation

Clone this repository to your local environment, create a python virtual environment and activate it:
  `python -m venv .bvt`
  `source .bvt/bin/activate`
Or, on windows:
  `python -m venv .bvt`
  `.\.bvt\Scripts\activate`

Then all dependencies with `requirements.txt` file:
  `pip install -r requirements.txt`

Warning: some library versions may cause code issues or conflicts. To avoid that use versions listed in requirements file, however if you have to install newer versions, note that you need:
- numpy<=1.26.4

# Bundle exe application with pyinstaller

On windows, go to your repo folder and run:
  `pyinstaller -w -F --path ".bvt\Lib\site-packages" --icon "assets\icon2.ico" --add-data=".bvt\Lib\site-packages\tkinterDnD":. .\BenthicVideoToolbox.py`

On linux:
  `pyinstaller -w -F --add-data=".env/lib/python3.8/site-packages/tkinterDnD/linux":. BenthicVideoToolbox.py`