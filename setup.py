from setuptools import setup, find_packages

setup(
    name='ruhsub',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.10.0+rocm',
        'googletrans==4.0.0-rc1',
        'deepl==1.16.1',
        'moviepy==1.0.3',
        'whisper_timestamped==0.1.0',
        'TTS==0.22.0',
        'pysrt==1.1.2',
    ],
)
