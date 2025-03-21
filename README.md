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

