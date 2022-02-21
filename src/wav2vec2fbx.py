import os
import sys
import copy
import contextlib
import argparse
import pathlib
import multiprocessing

import concurrent.futures

import toml
import torch
# from torch._C import default_generator

import librosa


from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    # Wav2Vec2PhonemeCTCTokenizer,
)


import fbx_writer
import audio_util

##############################################################################

if sys.version_info >= (3, 0):
    # For type annotation
    from typing import (  # NOQA: F401 pylint: disable=unused-import
        MutableMapping,
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
    # type: (argparse.Namespace) -> MutableMapping[Text, Any]

    if args.config_file is not None:
        return toml.load(args.config_file)

    if getattr(sys, 'frozen', False):
        # frozen
        dir_ = os.path.dirname(sys.executable)
    else:
        # unfrozen
        dir_ = os.path.dirname(os.path.realpath(__file__))

    config_paths = [
        os.path.join(dir_, CONFIG_NAME),
        os.path.join(dir_, "../", CONFIG_NAME),
        os.path.join(dir_, "assets", CONFIG_NAME),
        os.path.join(dir_, "../assets", CONFIG_NAME),
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            return toml.load(config_path)

    raise Exception("config file not found {}".format(config_paths))


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


def calculate_voice_power(vals, ids):
    # type: (torch.Tensor, torch.Tensor) -> Tuple[float, float, float]

    x, y, z = 0., 0., 0.

    v0 = vals[0].item()
    v1 = vals[1].item()
    v2 = vals[2].item()

    if not isinstance(v0, float):
        raise
    if not isinstance(v1, float):
        raise
    if not isinstance(v2, float):
        raise

    # -------------------------------------------------
    if ids[0] == 0:
        # if silence prevails, lift the others
        y = v2 + (10. - v0) / 2.0  # type: float
        z = v2 + (10. - v0) / 2.0  # type: float
        if y < 5.:
            y = 0.
        if z < 5.:
            z = 0.

    else:
        x = min(10.0, vals[0].item() * 1.5)  # type: ignore
        y = v1 if v1 > 4. else 0.
        z = v2 if v2 > 4. else 0.

    # -------------------------------------------------
    # limit total when uttered simultaneously
    if x > 0.:
        x = (((1. / (x + y + z)) * x) * 0.5) + (x * 0.5)
        y = (((1. / (x + y + z)) * y) * 0.5) + (y * 0.5)
        z = (((1. / (x + y + z)) * z) * 0.5) + (z * 0.5)

    x = 1.0 if x > 10. else x / 10.
    y = 1.0 if y > 10. else y / 10.
    z = 1.0 if z > 10. else z / 10.

    return x, y, z  # type: ignore


def process_audio(processor, model, audio_filepath, proc_num):
    # type: (Wav2Vec2Processor, Wav2Vec2ForCTC, Text, int) -> Tuple[float, torch.Tensor, torch.Tensor]

    speech, sample_rate = librosa.load(audio_filepath, sr=16000)
    print(sample_rate)
    raise
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values.cpu()
    duration_sec = input_values.shape[1] / sample_rate

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    probabilities, ids = torch.topk(logits, k=3, dim=-1)

    transcription = processor.decode(predicted_ids[0]).lower()
    print(f"process {duration_sec:.3f}sec audio({proc_num}) file as {transcription}")

    return duration_sec, probabilities, ids


class CurveKey(tuple):
    """Contains keyframe(time in seconds), keyvalue pair."""
    def __init__(self, time, value):
        super().__init__()


class AnimCurve():
    """Contains CurveKeys."""

    def add_key(self, key):
        pass


def generate_keyframes(conf, processor, duration_sec, probabilities, ids, offset):
    # type: (MutableMapping[Text, Any], Wav2Vec2Processor, float, torch.Tensor, torch.Tensor, float) -> Dict

    # animation_frames represents all frame plotted including empty move.
    animation_frames = []  # type: List[Dict[Text, float]]
    ipa_to_arpabet = load_ipa_arpabet_table(conf)

    for i, (top3_vals, top3_ids) in enumerate(zip(probabilities[0], ids[0])):

        animation_frames.append({})

        if is_silence(top3_vals, top3_ids):
            continue

        x, y, z = calculate_voice_power(top3_vals, top3_ids)
        for index, val in zip(top3_ids, (x, y, z)):
            if index != 0 and val > 0.:
                ipa = processor.decode(index.item()).lower()
                arpabet = ipa_to_arpabet.get(ipa)
                if not arpabet:
                    print(f"ipa not found in the config.toml table replacing _: {ipa}")
                    arpabet = ipa_to_arpabet.get("default", "_")

                if arpabet in animation_frames[i]:
                    animation_frames[i][arpabet] += val
                else: 
                    animation_frames[i][arpabet] = val

    # TODO: support the ipa that has multiple visemes.

    # Since only voiced keys are placed in the `animation_frames` here,
    # we need to specify the start and end frame before and after the key.
    for i, keys in enumerate(copy.deepcopy(animation_frames)):
        for k, _ in keys.items():
            set_zero_key(conf, animation_frames, k, i)  # animation_frames got side effect

    object_based_keys = {}
    for i, keys in enumerate(copy.deepcopy(animation_frames)):
        sec = (i / len(probabilities[0])) * duration_sec + offset
        for phon, val in keys.items():

            if phon in object_based_keys:
                object_based_keys[phon].append((sec, val))
            else:
                object_based_keys[phon] = []
                object_based_keys[phon].append((sec, val))

    return object_based_keys


def set_zero_key(conf, animation_keys, phon, frame_index):
    # type: (MutableMapping[Text, Any], List[Dict[Text, float]], Text, int) -> ...
    """CAUTION: animation_keys is mutable and changed by calling this function."""

    interpolation = conf.get("keyframe", {}).get("interpolation", 5)

    prev_key_index_start = max(1, frame_index - 1)
    prev_key_index_stop = max(1, frame_index - interpolation - 1)
    next_key_index_start = min(frame_index + 1, len(animation_keys))
    next_key_index_stop = min(frame_index + interpolation + 1, len(animation_keys))

    # ------------------------------
    for frame in range(prev_key_index_start, prev_key_index_stop, -1):
        if phon in animation_keys[frame]:
            break
        # print("prev", frame, animation_keys[frame])
    else:
        with contextlib.suppress(IndexError, UnboundLocalError, AttributeError):
            animation_keys[frame - 1][phon] = 0.0  # type: ignore  #  i know this dangerous...

    # ------------------------------
    for frame in range(next_key_index_start, next_key_index_stop):
        if phon in animation_keys[frame]:
            break
    else:
        with contextlib.suppress(IndexError, UnboundLocalError, AttributeError):
            animation_keys[frame + 1][phon] = 0.0  # type: ignore


def load_ipa_arpabet_table(conf):
    # type: (MutableMapping[Text, Any]) -> Dict[Text, Text]

    return conf.get("ipa_to_arpabet", {})


def async_split_audio(audio_filepath, audio_settings):
    files_and_offsets = audio_util.split_audio(audio_filepath, **audio_settings)
    return files_and_offsets


def async_load_models():
    with torch.no_grad():
        processor, model = load_models()
        return processor, model


def async_process_audio(config, processor, model, file_path, offset, proc_num):
    duration_sec, probabilities, ids = process_audio(processor, model, file_path, proc_num)
    keys = generate_keyframes(config, processor, duration_sec, probabilities, ids, offset)

    return keys


def async_main():

    args = parse_args()
    config = load_config(args)

    audio_filepath = args.input_audiofile  # type: pathlib.Path
    if audio_filepath.suffix != ".wav":
        raise Exception("input audio is not wav")
    fbx_path = audio_filepath.with_suffix(".fbx")

    audio_config = config.get("audio_settings", {})

    futures = []
    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        f = executor.submit(async_split_audio, audio_filepath, audio_config)
        futures.append(f)

        f = executor.submit(async_load_models)
        futures.append(f)

    concurrent.futures.wait(futures)
    files_and_offsets = futures[0].result()
    processor, model = futures[1].result()

    total_keys = {}
    futures = []
    proc_count = min(int(multiprocessing.cpu_count() / 2), len(files_and_offsets))
    with concurrent.futures.ProcessPoolExecutor(proc_count) as executor:
        for i, (file_path, offset) in enumerate(files_and_offsets):
            f = executor.submit(async_process_audio, config, processor, model, file_path, offset, i)
            futures.append(f)

    concurrent.futures.wait(futures)
    for f in futures:
        keys = f.result()
        for key, value in keys.items():
            if key in total_keys:
                total_keys[key].extend(value)
            else:
                total_keys[key] = value

    fbx_writer.write(total_keys, fbx_path)
    print("done!!")


def main():

    args = parse_args()
    config = load_config(args)

    audio_filepath = args.input_audiofile  # type: pathlib.Path
    if audio_filepath.suffix != ".wav":
        raise Exception("input audio is not wav")
    fbx_path = audio_filepath.with_suffix(".fbx")

    audio_config = config.get("audio_settings", {})

    files_and_offsets = audio_util.split_audio(audio_filepath, **audio_config)
    processor, model = load_models()

    total_keys = {}
    for i, (file_path, offset) in enumerate(files_and_offsets):
        duration_sec, probabilities, ids = process_audio(processor, model, file_path, i)
        # print(f"{file_path}: duration: {duration_sec}, offset:{offset}")
        keys = generate_keyframes(config, processor, duration_sec, probabilities, ids, offset)

        for key, value in keys.items():
            if key in total_keys:
                total_keys[key].extend(value)
            else:
                total_keys[key] = value

    fbx_writer.write(total_keys, fbx_path)
    print("done!!")


if __name__ == '__main__':
    import timeit
    # print(timeit.timeit("main()", setup="from __main__ import main", number=2))
    # print(timeit.timeit("asyncio.run(async_main())", setup="import asyncio;from __main__ import async_main", number=2))
    async_main()
    # asyncio.run(main())
