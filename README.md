# Wav2Vec2FBX
---
Generate an FBX of a phoneme lip-sync animation from an audio file, using Wav2Vec2 to analyze the phonemes.


## How to Install and Run

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

### Run
```sh
python src/wav2vec2fbx.py input_audio.wav
```

## Configuration file
The behaviour can be changed by the configuration file `assets/config.toml`.

### audio settings

### ipa to arpabet table settings
