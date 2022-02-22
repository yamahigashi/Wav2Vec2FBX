# Wav2Vec2FBX

> Generate an FBX of a phoneme lip-sync animation from an sigle audio file, using Wav2Vec2 to analyze the phonemes.
> 
![alt text](https://github.com/yamahigashi/Wav2Vec2FBX/blob/doc/Screenshot_434.png?raw=true)


## Table of contents

<!--ts-->
   * [Installation](#installation)
     *  [Environment](#environment)
     *  [Clone Repository](#clone-repository)
     *  [FBX python SDK](#fbx-python-sdk)
  *  [Run](#run)
  *  [Configuration](#configuration)
  *  [References](#references)
<!--te-->


## Installation

#### Environment
Virtual environment, Python 3.7 is highly recommended, as it is supported by the FBX Python SDK.


#### Clone repository
```sh
git clone https://github.com/yamahigashi/Wav2Vec2FBX.git
cd Wav2Vec2FBX
pip install requirements.txt
```

#### FBX python SDK 
https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0
download FBX SDK from autodesk and place libraries into your python path.

## Run
```sh
python src/wav2vec2fbx.py input_audio.wav
```
This will generate `input_audio.fbx` in the same folder as the input file.


## Configuration
The behaviour can be changed by the configuration file `assets/config.toml`.

#### keyframes settings

```toml
[keyframes]
# ipa と無口を補完するフレーム
interpolation = 5

# 複数口形素からなる ipa を補完するフレーム
consecutive_viseme_frame = 3
```

#### audio settings section
Describes settings for preprocessing an audio file. It splits the file based on the silence, and if it is still too long, splits the file based on the settings.
```toml
[audio_settings]

# 無音期間を判定する際の最小ミリセク  (初期値 500)
min_silence_len_ms = 500

# 無音判定 (初期値 -36)
silence_thresh_db = -36

# 最長オーディオファイル。これ以上は複数に分割して処理 (初期値 5000)
maximum_duration_ms = 5000   
```

#### ipa to arpabet table settings
The phonemes to morphemes correspondence table. The phonemes determined by Wav2Vec are mapped to oral morphemes. The list of morphonemes can be given as.
```toml
[ipa_to_arpabet]
'ɔ'      = ["a"]
'ɑ'     = ["a"]
'i'      = ["i"]
# Long Vowels
'e ː'   = ["e", "e"]
'o ː'   = ["o", "o"]

# -------- snip --------------
```
## Build binary using cx_Freeze
You can deploy this package as binary for the environment without python using `cx_Freeze`.
```bash
python setup.py build
```
This will generate binary for your platform.


## References
- https://huggingface.co/docs/transformers/model_doc/wav2vec2
- https://arxiv.org/abs/1904.05862
- https://arxiv.org/abs/2006.11477
- https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#wav2vec-20
