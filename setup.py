#!/usr/bin/env python
from __future__ import unicode_literals

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='ruhsub',
    version='0.5.6',
    description='Auto-generates subtitles && edites video audio for any video or audio file',
    author='Rauf Şen',
    author_email='raufsen11@gmail.com',
    url='https://github.com/rafucuk/ruhsub',
    packages=['ruhsub'],
    entry_points={
        'console_scripts': [
            'ruhsub=ruhsub:main',
        ],
    },
    install_requires=[
        'googletrans==4.0.0-rc1',
        'deepl==1.16.1',
        'moviepy==1.0.3',
        'whisper_timestamped==1.14.4',
        'whisper==1.1.10',
        'TTS==0.22.0',
        'pysrt==1.1.2',
        'PyDeepLX==1.0.6',
        'langdetect==1.0.9',
    ]
)
