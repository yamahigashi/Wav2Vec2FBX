import sys
from cx_Freeze import setup, Executable


build_exe_options = {
    "packages": [
        "os",
        "torch",
        "_soundfile_data",
        "pydub"
    ],
    "zip_include_packages": [
    ],
    "excludes": ["tkinter"],
    "include_files": [
        ("assets/config.toml", "config.toml"),
        ("build/lib", "lib"),
        ("lib", "lib"),
    ]
}

base = None

setup(
    name="Wav2Vec2FBX",
    packages=["Wav2Vec2FBX"],
    package_dir={"Wav2Vec2FBX": "src/Wav2Vec2FBX"},
    version="0.1",
    description="Recognize speech file and convert it into animation FBX ",
    options={"build_exe": build_exe_options},
    executables=[Executable("src/Wav2Vec2FBX/wav2vec2fbx.py", base=base, target_name="Wav2Vec2FBX")],
    entry_points={'console_scripts': ['Wav2Vec2FBX = src.__main__:main']}
)
