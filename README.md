# Wav2Vec2FBX

> Generate an FBX of a phoneme lip-sync animation from an sigle audio file, using Wav2Vec2 to analyze the phonemes.
> 
![alt text](https://github.com/yamahigashi/Wav2Vec2FBX/blob/doc/Screenshot_434.png?raw=true)


Table of contents
=================

<!--ts-->
   * [Installation](#installation)
     *  [Environment](#environment)
     *  [Clone Repository](#clone-repository)
     *  [FBX python SDK](#fbx-python-sdk)
  *  [Run](#run)
  *  [Configuration](#configuration)
<!--te-->


## Installation

### Environment
Python3.7 is recommended, as it is supported by the FBX Python SDK.


### Clone repository
```sh
git clone https://github.com/yamahigashi/Wav2Vec2FBX.git
cd Wav2Vec2FBX
pip install requirements.txt
```

### FBX python SDK 
https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0
download FBX SDK from autodesk and place libraries into your python path.

## Run
```sh
python src/wav2vec2fbx.py input_audio.wav
```
This will generate `input_audio.fbx` in the same folder as the input file.


## Configuration
The behaviour can be changed by the configuration file `assets/config.toml`.

### audio settings
TODO later

### ipa to arpabet table settings
TODO later

## Build binary using cx_Freeze
TODO later
