# ModelB5 for Self-Driving Cars

## Installation
Do [Step 1](https://docs.google.com/document/d/1tH6coTWyIQ3QZUrmNFav6xfYn9PV-mGk2FiN3yYW_IY/edit?usp=sharing) and [Step 2](https://docs.google.com/document/d/1x1OMnGbGKDapQEBx4xNi2VEwYRL0_XFLZZZDvE8Vefo/edit)

## Project B5
### Step 1. Prepare Data 
[fcamera.hevc](https://drive.google.com/file/d/1GOOD4IhagzsaB_HsC6cvavKj1lLcD0nb/view?usp=sharing)
[hevc2yuvh5.py](https://github.com/JinnAIGroup/B5/blob/main/hevc2yuvh5.py) => yuv.h5

[rlog.bz2](https://drive.google.com/file/d/1GOOD4IhagzsaB_HsC6cvavKj1lLcD0nb/view?usp=sharing)
[bz2toh5.py](https://github.com/JinnAIGroup/YPNetA/blob/main/bz2toh5.py) => radar.h5
### Step 2. Create Model 
[modelB5.py](https://github.com/JinnAIGroup/B5/blob/main/modelB5.py)
### Step 3. Generate Data
[serverB5.py](https://github.com/JinnAIGroup/B5/blob/main/serverB5.py), 
[datagenB5.py](https://github.com/JinnAIGroup/B5/blob/main/datagenB5.py)
### Step 4. Train Model
[train_modelB5.py](https://github.com/JinnAIGroup/B5/blob/main/train_modelB5.py) => modelB5.h5 => outputs:
[plot_txtB5.py](https://github.com/JinnAIGroup/B5/blob/main/plot_txtB5.py) 
### Step 5. Verify Model
[simulatorB5.py](https://github.com/JinnAIGroup/B5/blob/main/simulatorB5.py) => [Horace](https://docs.google.com/presentation/d/1S0xFpluCefNjDe3FC8mo8uWeP91X2v7C/edit?usp=sharing&ouid=117467329867185057226&rtpof=true&sd=true) => [output.txt](https://github.com/JinnAIGroup/B5/blob/main/output.txt)
### Step 6. Deploy Model
[h5topbB5.py](https://github.com/JinnAIGroup/B5/blob/main/h5topbB5.py) => modelB5.pb => modelB5.dlc => modelB5.html
([SNPE](https://docs.google.com/document/d/1x1OMnGbGKDapQEBx4xNi2VEwYRL0_XFLZZZDvE8Vefo/edit) =>
[Run on C2](https://github.com/JinnAIGroup/CAN/blob/main/zcCS7.py) => [Self-Driving](https://drive.google.com/file/d/10Rp19QgbRTYRh1dflaPtOj72Tc1aB7pv/view?usp=sharing))
