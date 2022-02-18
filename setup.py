import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"packages": ["os"], "excludes": []}

base = None

setup(
    name="Wav2Vec2FBX",
    version="0.1",
    description="Recognize speech file and convert it into animation FBX ",
    options={"build_exe": build_exe_options},
    executables=[Executable("src/wav2vec2fbx.py", base=base)]
)
