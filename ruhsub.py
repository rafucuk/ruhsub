import whisper
from TTS.api import TTS
import torch
from IPython.display import Audio, display
import subprocess
import os
from googletrans import Translator
import sys

def process_videos(video_path):
    ffmpeg_command = f"ffmpeg -i '{video_path}' -acodec pcm_s24le -ar 48000 -q:a 0 -map a -y 'output_audio.wav'"
    subprocess.run(ffmpeg_command, shell=True)

    model = whisper.load_model("base")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    result = model.transcribe("output_audio.wav")
    whisper_text = result["text"]
    whisper_language = result['language']
    
    target_language_code = "tr"
    translator = Translator()
    translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text

    tts.tts_to_file(translated_text,
        speaker_wav='output_audio.wav',
        file_path="output_synth.wav",
        language=target_language_code
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_videos(video_path)

if __name__ == "__main__":
    main()
