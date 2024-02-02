import os
import shutil
import sys
import json
from datetime import datetime, timedelta
from googletrans import Translator
import deepl
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
import logging
import numpy as np
import whisper_timestamped as whisper
from TTS.api import TTS
import pysrt
import subprocess
import tempfile
import concurrent.futures  # Import the module for parallel processing


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

translator = deepl.Translator("230371e8-9dad-4a76-8e46-50241c697d32:fx")

def extract_audio(video_path, audio_output_path):
    logger.info(f"Extracting audio from {video_path}")
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le', fps=16000)
    logger.info(f"Audio extracted and saved to {audio_output_path}")
    return audio_output_path

def merge_segments(original_segments):
    merged_segments = []

    current_start_time = original_segments[0]["start"]
    current_sentence = ""
    current_words = []
    sentence_counter = 0

    for i, segment in enumerate(original_segments):
        if i == 0:
            # If it's the first segment, create a new segment with the start time of segment[0]
            new_segment = {
                "id": len(merged_segments),
                "seek": segment["seek"],
                "start": 0.00,
                "end": segment["end"],
                "text": "[Intro]",
                "tokens": segment["tokens"],  # You may need to update this based on your requirements
                "temperature": segment["temperature"],
                "confidence": segment["confidence"],
                "words": current_words  # Include words up to the current word
            }

            merged_segments.append(new_segment)

            # Update start time for the next sentence
            current_start_time = segment["end"]
            # Reset the current sentence and current words
            current_sentence = ""
            current_words = []

        for word in segment["words"]:
            if word["text"] == "[*]":
                continue  # Skip words with text "[*]"

            current_sentence += word["text"] + " "
            current_words.append(word)

            if word["text"][-1] in {'.', '!', '?'}:
                sentence_counter += 1

                if sentence_counter == 2:
                    sentence_counter = 0

                    # Combine every two sentences or if there are no sentences left
                    new_segment = {
                        "id": len(merged_segments),
                        "seek": segment["seek"],
                        "start": current_start_time,
                        "end": word["end"],
                        "text": current_sentence.strip(),
                        "tokens": segment["tokens"],  # You may need to update this based on your requirements
                        "temperature": segment["temperature"],
                        "confidence": segment["confidence"],
                        "words": current_words  # Include words up to the current word
                    }

                    merged_segments.append(new_segment)

                    # Update start time for the next sentence
                    current_start_time = word["end"]
                    # Reset the current sentence and current words
                    current_sentence = ""
                    current_words = []

    return merged_segments

def asr_segmentation(video_path, audio_path, temp_folder="temp"):
    model = whisper.load_model("base", device="cuda")

    # Transcribe the entire audio
    result = whisper.transcribe(model, audio_path, language="en", beam_size=20, best_of=20, vad="auditok", temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), detect_disfluencies=True)

    # Get timestamps from the transcribed segments
    fortranslate = result["segments"]
    speech_timestamps = merge_segments(fortranslate)
    # Create the temp folder if not exists
    os.makedirs(temp_folder, exist_ok=True)

    translated_segments = []
    original_segments = []

    translated_srt_path = os.path.basename(video_path).split('/')[-1] + "_tr.srt"
    translated_srt = pysrt.SubRipFile()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    translated_segments_paths = []

    for i, segment in enumerate(speech_timestamps):
        start_time, end_time = segment['start'], segment['end']
        segment_path = os.path.join(temp_folder, f"segment_{i}.mp4")
        audio_output_path = os.path.join(temp_folder, f"segment_{i}.wav")
        translated_audio_output_path = os.path.join(temp_folder, f"tr_segment_{i}.wav")
        translated_segments_paths.append(segment_path)

        #
        # Use MoviePy for video slicing without audio
        #
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(end_time - start_time),
            '-c:v', 'h264_amf',
            '-an',  # Disable audio
            '-threads', '48',
            segment_path
        ]
        subprocess.run(ffmpeg_command, capture_output=False, text=False)

        #
        # Use MoviePy for audio slicing
        #
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', audio_path,
            '-t', str(end_time - start_time),
            '-acodec', 'pcm_s16le',
            '-c:v', 'h264_amf',
            '-ar', '16000',
            audio_output_path
        ]
        subprocess.run(ffmpeg_command, capture_output=False, text=False)


        #
        # Translate the text and create audio file from translated text
        #
        translated_text = translator.translate_text(segment['text'], target_lang='tr').text
        tts.tts_to_file(translated_text,
                        speaker_wav='output_audio.wav',
                        file_path=os.path.join(temp_folder, f"tr_segment_{i}.wav"),
                        language='tr'
                        )

        translated_audio_path = os.path.join(temp_folder, f"tr_segment_{i}.wav")


        videoclip = VideoFileClip(segment_path)
        audioclip = AudioFileClip(translated_audio_path)

        duration = audioclip.duration
        translated_audio_duration = float(duration)

        fps = videoclip.fps
        duration = videoclip.duration
        original_video_duration = float(duration)
        original_video_fps = float(fps)

        # Calculate speed_factor
        speed_factor = translated_audio_duration / original_video_duration

        # Apply speed change to video using FFmpeg
        ffmpeg_speed_change_command = [
            'ffmpeg',
            '-y',
            '-i', segment_path,
            '-r', '60',
            '-c:v', 'h264_amf',
            '-vf', f'setpts={speed_factor}*PTS',
            'temp_slowed.mp4'
        ]
        subprocess.run(ffmpeg_speed_change_command)

        # Concatenate slowed video with translated audio using FFmpeg
        ffmpeg_concat_command = [
            'ffmpeg',
            '-y',
            '-i', 'temp_slowed.mp4',
            '-i', translated_audio_output_path,
            '-c:v', 'h264_amf',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-threads', '48',
            '-shortest',  # Ensure output is the duration of the shorter input
            segment_path
        ]
        subprocess.run(ffmpeg_concat_command)

        # Convert start and end times to pysrt format (seconds with milliseconds)
        start_time_seconds = int(start_time)
        start_time_milliseconds = int((start_time - start_time_seconds) * 1000)
        end_time_seconds = int(end_time)
        end_time_milliseconds = int((end_time - end_time_seconds) * 1000)

        # Create pysrt SubRipItem
        translated_subtitle = pysrt.SubRipItem(
            index=i + 1,  # You may adjust the index as needed
            start=pysrt.SubRipTime(seconds=start_time_seconds, milliseconds=start_time_milliseconds),
            end=pysrt.SubRipTime(seconds=end_time_seconds, milliseconds=end_time_milliseconds),
            text=translated_text
        )
        translated_srt.append(translated_subtitle)

        original_segments.append(segment_path)
        translated_segments.append(segment_path)

    translated_srt.save(translated_srt_path, encoding='utf-8')
    # Correct indentation for the return statement
    return original_segments, translated_segments, translated_segments_paths

def concatenate_video_ffmpeg(video_segments, output_path):
    # Create a list of file paths for the video segments
    segments_file_paths = [f'file \'{segment}\'' for segment in video_segments]

    # Use ffmpeg to concatenate the video segments
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',  # Allow the use of unsafe file names
        '-i', 'segment_paths.txt',  # Use a pipe to pass the file paths
        '-c:v', 'h264_amf',
        '-c:a', 'aac',
        output_path
    ]

    # Use subprocess.PIPE to pass the file paths through a pipe
    subprocess.run(ffmpeg_command, input='\n'.join(segments_file_paths).encode('utf-8'))

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: ruhsub <video_name>")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    video_path = os.path.abspath(sys.argv[1])
    audio_output_path = os.path.join(script_dir, "output_audio.wav")
    output_synced_video_path = os.path.join(script_dir, f"{os.path.basename(video_path).split('.')[0]}_tr.mp4")

    # Step 1: Extract audio from the video
    audio_path = extract_audio(video_path, audio_output_path)

    # Step 2: Perform ASR, translation, and segmentation
    original_segments, translated_segments, translated_segments_paths = asr_segmentation(video_path, audio_path)

    # Step 3: Save the segmented MP4 paths to a list
    with open("segment_paths.txt", "w") as file:
        for path in translated_segments_paths:
            file.write(f"file '{path}'\n")

    # Step 4: Combine all the segmented MP4s into one MP4 using the list
    concatenate_video_ffmpeg(translated_segments_paths, output_synced_video_path)

    shutil.rmtree(os.path.join(script_dir, "temp/"))
    os.remove(os.path.join(script_dir, "output_audio.wav"))
    os.remove(os.path.join(script_dir, "segment_paths.txt"))
    os.remove(os.path.join(script_dir, "temp_slowed.mp4"))


    logger.info("Processing completed")

if __name__ == "__main__":
    main()
