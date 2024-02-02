from setuptools import setup, find_packages

setup(
    name='ruhsub',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'googletrans==4.0.0-rc1',
        'deepl==2.1.0',
        'moviepy==1.0.3',
        'whisper_timestamped==0.1.0',
        'TTS==0.0.1',
        'pysrt==1.1.1',
    ],
)
