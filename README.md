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


# Install BenthicVideoToolbox repo based on laser-auto-detection branch

Before cloning this branch, you need to download and install git lfs in order to download large file (here epoch_7.pth needed to run inference model):
- Download https://git-lfs.com/ and run installation exe
- Open a terminal and run `git lfs install`

Clone this repository (auto-laser-detection branch) to your local environment, create a python virtual environment and activate it:
  `python -m venv .bvt`
  `source .bvt/bin/activate`
Or, on windows:
  `python -m venv .bvt`
  `.\.bvt\Scripts\activate`

Note: I had issues using pysinstaller with python3.12 (when opening bundled app I get error message "DLL load failed while importing pyexpat"). As a workaround I used python3.8 env to bundle app with pyinstaller.

Then all dependencies with `requirements.txt` file:
  `pip install -r requirements.txt`

Warning: some library versions may cause code issues or conflicts. To avoid that use versions listed in requirements file, however if you have to install newer versions, note that you need:
- numpy<=1.26.4
- opencv-python<=4.10.0.84

# mmdet installation notes (from https://mmdetection.readthedocs.io/):

1. Install PyTorch 2.1.2 (newest versions need mmcv2.2 which is incompatible with current version of mmdet), adapt to your cuda toolkit version (you may need to download an older version of cuda toolkit compatible with pytorch 2.1.2)
with GPU:
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
Check installation:
    python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

OR CPU only:
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
Check installation:
    python -c "import torch;print(torch.__version__)"

2. Install mmengine and mmcv:
    <!-- pip install -U openmim -->
    mim install mmengine
    mim install mmcv==2.1.0
    
3. Install mmdet:
        mim install mmdet==3.3.0
    OR Install mmdet from source:
        git clone https://github.com/open-mmlab/mmdetection.git
        cd mmdetection
        pip install -v -e .
    Check installation with:
        mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
        python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda (or cpu if cpu install)

Note: If an error of type "cannot import name 'six' from 'pkg_resources.extern'" appears during mmengine, mmcv or mmdet installation, try upgrading pip (python -m pip install --upgrade pip) and re-try


# Bundle exe application with pyinstaller

On windows, go to your repo folder and run:
  `pyinstaller -w -F --path ".bvt\Lib\site-packages" --icon "assets\icon2.ico" --add-data="assets/:assets" --collect-all mmdet --collect-all mmengine --collect-all mmcv --noconfirm --add-data=".bvt\Lib\site-packages\tkinterDnD":. .\BenthicVideoToolbox.py`

On linux:
  `pyinstaller -w -F --add-data="assets/:assets" --collect-all mmdet --collect-all mmengine --collect-all mmcv --noconfirm --add-data=".env/lib/python3.8/site-packages/tkinterDnD/linux":. BenthicVideoToolbox.py`