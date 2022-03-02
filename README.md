# ModelB5 for Self-Driving Cars

## Project B5
### Step 1. [Install Ubuntu and OP](https://docs.google.com/document/d/1tH6coTWyIQ3QZUrmNFav6xfYn9PV-mGk2FiN3yYW_IY/edit?usp=sharing)
### Step 2. Prepare Data 
[fcamera.hevc](https://drive.google.com/file/d/1GOOD4IhagzsaB_HsC6cvavKj1lLcD0nb/view?usp=sharing) => 
[hevc2yuvh5.py](https://github.com/JinnAIGroup/B5/blob/main/hevc2yuvh5.py) => yuv.h5

[rlog.bz2](https://drive.google.com/file/d/1GOOD4IhagzsaB_HsC6cvavKj1lLcD0nb/view?usp=sharing) => 
[bz2toh5.py](https://github.com/JinnAIGroup/YPNetA/blob/main/bz2toh5.py) => pathdata.h5, radardata.h5
### Step 3. Create Model 
[modelB5.py](https://github.com/JinnAIGroup/B5/blob/main/modelB5.py)
### Step 4. Generate Data
[serverB5.py](https://github.com/JinnAIGroup/B5/blob/main/serverB5.py), 
[datagenB5.py](https://github.com/JinnAIGroup/B5/blob/main/datagenB5.py)
### Step 5. Train Model
[train_modelB5.py](https://github.com/JinnAIGroup/B5/blob/main/train_modelB5.py) => modelB5.h5 => outputs:
[plot_txtB5.py](https://github.com/JinnAIGroup/B5/blob/main/plot_txtB5.py) 
### Step 6. Verify Model
[simulatorB5.py](https://github.com/JinnAIGroup/B5/blob/main/simulatorB5.py) => modelB5.h5 => 
[output.txt](https://github.com/JinnAIGroup/B5/blob/main/output.txt) => 
[Horace](https://drive.google.com/file/d/15RyzVCR_greK_NXDm_AcLsLmEsqg-e9d/view?usp=sharing)
### Step 7. [Install SNPE](https://docs.google.com/document/d/1x1OMnGbGKDapQEBx4xNi2VEwYRL0_XFLZZZDvE8Vefo/edit)
### Step 8. Deploy Model
[h5topbB5.py](https://github.com/JinnAIGroup/B5/blob/main/h5topbB5.py) => modelB5.pb => modelB5.dlc => modelB5.html =>
[Self-Driving](https://drive.google.com/file/d/10Rp19QgbRTYRh1dflaPtOj72Tc1aB7pv/view?usp=sharing)
