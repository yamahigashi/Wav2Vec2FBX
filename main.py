import os
import sys
lib_dir = os.path.join(os.path.dirname(__file__), "lib")
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(lib_dir)
sys.path.append(src_dir)

import src.Wav2Vec2FBX.__main__  # noqa
