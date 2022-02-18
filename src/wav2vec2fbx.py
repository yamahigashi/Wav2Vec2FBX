import os
import sys
import copy
from itertools import groupby
from collections import defaultdict
import argparse
import pathlib
import toml

import torch
# from torch._C import default_generator

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    # Wav2Vec2PhonemeCTCTokenizer,
)
# from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES

import librosa

import fbx_writer

##############################################################################

if sys.version_info >= (3, 0):
    # For type annotation
    from typing import (  # NOQA: F401 pylint: disable=unused-import
        Optional,
        Dict,
        List,
        Tuple,
        Pattern,
        Callable,
        Any,
        Text,
        Generator,
        Union
    )
##############################################################################
INTERPOLATION_FRAME = 2
CONFIG_NAME = "config.toml"
##############################################################################


def parse_args():
    # type: () -> argparse.Namespace

    parser = argparse.ArgumentParser()
    parser.add_argument('input_audiofile', type=pathlib.Path)
    parser.add_argument('--config_file', type=pathlib.Path)
    args = parser.parse_args()

    return args


def load_config(args):

    if args.config_file is not None:
        return toml.load(args.config_file)

    config_path = os.path.join(os.path.dirname(__file__), CONFIG_NAME)
    if os.path.exists(config_path):
        return toml.load(config_path)

    config_path = os.path.join(os.path.dirname(__file__), "../", CONFIG_NAME)
    if os.path.exists(config_path):
        return toml.load(config_path)

    config_path = os.path.join(os.path.dirname(__file__), "assets", CONFIG_NAME)
    if os.path.exists(config_path):
        return toml.load(config_path)

    config_path = os.path.join(os.path.dirname(__file__), "../assets", CONFIG_NAME)
    if os.path.exists(config_path):
        return toml.load(config_path)

    raise Exception("config file not found {}".format(config_path))


def load_models(model_name='facebook/wav2vec2-large-960h-lv60-self'):
    # type: (Text) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]
    """load model & audio and run audio through model."""

    print("Start initialization of Wav2Vec2 models.")
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).cpu()
    # tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)

    print("Completed initialzation of Wav2Vec2 models")

    return processor, model


def is_silence(vals, ids):
    # type: (torch.Tensor, torch.Tensor) -> bool

    if ids[0] == 0 and vals[0] > 10.:
        return True

    if ids[0] == 0:
        x = vals[1] + (10. - vals[0]) / 2.5
        y = vals[2] + (10. - vals[0]) / 2.5

        if x < 5. and y < 5.:
            return True

    return False


def cals_power(vals, ids):

    x, y, z = 0., 0., 0.

    if ids[0] == 0:
        y = vals[1].item() + (10. - vals[0].item()) / 2.0
        z = vals[2].item() + (10. - vals[0].item()) / 2.0
        if y < 5.:
            y = 0.
        if z < 5.:
            z = 0.

    else:
        x = min(10.0, vals[0].item() * 1.5)
        y = vals[1].item() if vals[1] > 4. else 0.
        z = vals[2].item() if vals[2] > 4. else 0.

    return x, y, z


def process_audio(processor, model, audio_filepath):
    speech, sample_rate = librosa.load(audio_filepath, sr=16000)
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values.cpu()
    duration_sec = input_values.shape[1] / sample_rate

    print(f"process {duration_sec}sec audio file.")

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    vals, ids = torch.topk(logits, k=3, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower()
    print(transcription)

    return duration_sec, vals, ids
    # return duration_sec, predicted_ids, transcription


def alignment(processor, duration_sec, predicted_ids, transcription):
    """this is where the logic starts to get the start and end timestamp for each word."""

    words = [w for w in transcription.split(' ') if len(w) > 0]
    predicted_ids = predicted_ids[0].tolist()

    ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]

    # remove entries which are just "padding" (i.e. no characers are recognized)
    ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
    # print(ids_w_time)

    # now split the ids into groups of ids where each group represents a word
    split_ids_w_time = [list(group) for k, group in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id) if not k]

    # assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong  # nosec

    word_start_times = []
    word_end_times = []
    for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
        _times = [_time for _time, _id in cur_ids_w_time]
        word_start_times.append(min(_times))
        word_end_times.append(max(_times))
    
    return words, word_start_times, word_end_times


def generate_keyframes(conf, processor, duration_sec, vals, ids):

    animation_keys = []
    ipa_to_arpabet = load_ipa_arpabet_table(conf)

    for i, (top3_vals, top3_ids) in enumerate(zip(vals[0], ids[0])):

        animation_keys.append({})

        if is_silence(top3_vals, top3_ids):
            continue

        x, y, z = cals_power(top3_vals, top3_ids)
        for index, val in zip(top3_ids, (x, y, z)):
            if index != 0 and val > 0.:
                ipa = processor.decode(index.item()).lower()
                arpabet = ipa_to_arpabet.get(ipa)
                if not arpabet:
                    print(f"ipa not found {ipa}")

                if arpabet in animation_keys[i]:
                    animation_keys[i][arpabet] = 10.0
                else: 
                    animation_keys[i][arpabet] = val

    # print(len(vals[0]), len(animation_keys), duration_sec)
    for i, keys in enumerate(copy.deepcopy(animation_keys)):
        for k, _ in keys.items():
            set_zero_key(animation_keys, k, i)

    object_based_keys = defaultdict(lambda: [])
    for i, keys in enumerate(copy.deepcopy(animation_keys)):
        sec = i / len(vals[0]) * duration_sec
        for phon, val in keys.items():
            object_based_keys[phon].append((sec, val))

    # for phon, keys in object_based_keys.items():
    #     print(phon, keys)

    return object_based_keys


def set_zero_key(animation_keys, phon, frame_index):
    """CAUTION: animation_keys is mutable and changed by calling this function."""
    prev_key_index_start = max(1, frame_index - 1)
    prev_key_index_stop = max(1, frame_index - INTERPOLATION_FRAME - 1)
    next_key_index_start = min(frame_index + 1, len(animation_keys))
    next_key_index_stop = min(frame_index + INTERPOLATION_FRAME + 1, len(animation_keys))

    for frame in range(prev_key_index_start, prev_key_index_stop, -1):
        if phon in animation_keys[frame]:
            break
        # print("prev", frame, animation_keys[frame])
    else:
        animation_keys[frame - 1][phon] = 0.0  # type: ignore  #  i know this dangerous...

    for frame in range(next_key_index_start, next_key_index_stop):
        if phon in animation_keys[frame]:
            break
    else:
        animation_keys[frame + 1][phon] = 0.0  # type: ignore


def load_ipa_arpabet_table(conf):
    # type: (Dict) -> Dict[Text, Text]

    return conf.get("ipa_to_arpabet", {})


def main():

    args = parse_args()
    config = load_config(args)

    audio_filepath = args.input_audiofile  # type: pathlib.Path
    fbx_path = audio_filepath.with_suffix(".fbx")

    processor, model = load_models()
    duration_sec, vals, ids = process_audio(processor, model, audio_filepath)
    keys = generate_keyframes(config, processor, duration_sec, vals, ids)
    # words, word_start_times, word_end_times = alignment(processor, duration_sec, predicted_ids, transcription)
    # print(words, word_start_times, word_end_times)
    fbx_writer.write(keys, fbx_path, duration_sec)


if __name__ == "__main__":
    main()
