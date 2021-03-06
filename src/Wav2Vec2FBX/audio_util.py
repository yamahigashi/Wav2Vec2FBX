import os
import sys
import pathlib
import tempfile

import pydub
import pydub.utils
import pydub.silence


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


def split_by_silence(audio_file, min_silence_len_ms=500, silence_thresh_db=-35):
    # type: (pathlib.Path, int, int) -> Tuple

    keep_silence_ms = min_silence_len_ms

    audio_suffix = audio_file.suffix.split(".")[-1]
    if audio_suffix == "wav":
        sound_file = pydub.AudioSegment.from_wav(audio_file.as_posix())
    else:
        sound_file = pydub.AudioSegment.from_file(audio_file.as_posix(), audio_suffix)

    audio_chunks = pydub.silence.split_on_silence(
        sound_file,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
        seek_step=1,
    )
    silences = pydub.silence.detect_silence(
        sound_file,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        seek_step=1,
    )

    if not audio_chunks or len(audio_chunks) == 1 or not silences:
        return [], []

    if silences[0][0] == 0:
        pass
    else:
        silences.insert(0, (0, 0))

    silences = [(x + keep_silence_ms if x > 0 else x, y - keep_silence_ms if y > 0 else y) for x, y in silences]

    return audio_chunks, silences


def split_audio(audio_file, min_silence_len_ms=500, silence_thresh_db=-35, maximum_duration_ms=5000):
    # type: (pathlib.Path, int, int, int) -> List[Tuple[Text, float]]
    """Split input audio by silence and returns splitted file paths and its
    offset seconds.

    Here milliseconds
    """

    print("Preprocess audio file begins. Split files by silence and that are too long")
    audio_chunks, silences = split_by_silence(audio_file, min_silence_len_ms=500, silence_thresh_db=-35)
    if not audio_chunks:
        return [(audio_file.as_posix(), 0.0)]

    # assert len(audio_chunks) == len(silences)  # nosec

    results = []
    chunk_count = 0
    tempdir = tempfile.mkdtemp(audio_file.name)
    for chunk, silence in zip(audio_chunks, silences):

        duration = len(chunk)
        if duration > maximum_duration_ms:
            average_duration_ms = duration / ((duration // maximum_duration_ms) + 1)
            over_chunks = pydub.utils.make_chunks(chunk, average_duration_ms)

            for oc in over_chunks:
                out_file = os.path.join(tempdir, "chunk{0}.wav".format(chunk_count))
                print("exporting", out_file)
                oc.export(out_file, format="wav")
                results.append((out_file, silence[1] / 1000.))
                chunk_count += 1  # noqa

        else:
            out_file = os.path.join(tempdir, "chunk{0}.wav".format(chunk_count))
            print("exporting", out_file)
            chunk.export(out_file, format="wav")

            chunk_count += 1
            results.append((out_file, silence[1] / 1000.))

    return results
