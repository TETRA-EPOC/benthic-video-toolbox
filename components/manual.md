# files
1. ddod_r50SSL.py: This is the configuration file for mmdetection (https://mmdetection.readthedocs.io/)
2. det_inference.py: This is a python script to perform inference and if you want to save the files in COCO format /  upload the results to Biigle
3. biigleUtils.py: A file with utility functions for downloading volumes/images/... from biigle
4. epoch_7.pth: This file contains the trained weights for the mmdetection model
5. manual.md: This one ;-)

# Install dependencies with requirements.txt file

1. Create a conda or venv environment and activate it:
    conda create --name openmmlab python=3.8 -y
    conda activate openmmlab
or
    python -m venv .env-openmmlab
    source .env-openmmlab/bin/activate sur Linux
    .env-openmmlab/Scripts/Activate sur windows

2. pip install -r requirements.txt

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
