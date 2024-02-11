import os
import shutil
import sys
import json
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
import logging
import numpy as np
import whisper_timestamped as whisper
from TTS.api import TTS
import pysrt
import subprocess
import concurrent.futures  # Import the module for parallel processing
import deepl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

translator = deepl.Translator("3b039f0e-2363-4ddf-84f0-27e921d51121")

def convert_to_mp4(input_path, output_path):
    logging.info(f"Converting {input_path} to MP4 format")
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-crf', '23',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-b:a', '192k',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Conversion to MP4 completed: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting to MP4: {e}")
        return None

def extract_audio(video_path, audio_output_path):
    logging.info(f"Processing video: {video_path}")

    if os.path.exists(audio_output_path):
        logging.info(f"Audio file already exists at {audio_output_path}. Skipping extraction.")
        return audio_output_path

    # Check if the video needs conversion (if it's not already in MP4 format)
    if video_path.lower().endswith('.mkv'):
        mp4_path = video_path.rsplit('.', 1)[0] + '.mp4'
        converted_video_path = convert_to_mp4(video_path, mp4_path)
        if not converted_video_path:
            logging.error("Conversion to MP4 failed. Exiting.")
            return None
        video_path = converted_video_path

    # Use FFmpeg to extract audio
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', video_path,
        '-vn',  # Disable video recording
        '-acodec', 'pcm_s16le',  # Set audio codec to pcm_s16le
        '-ar', '16000',  # Set audio sample rate to 16000 Hz
        audio_output_path
    ]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Audio extracted and saved to {audio_output_path}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")
        return None

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
                "end": segment["start"],
                "text": "[Intro]",
                "tokens": segment["tokens"],  # You may need to update this based on your requirements
                "temperature": segment["temperature"],
                "confidence": segment["confidence"],
                "words": current_words  # Include words up to the current word
            }

            merged_segments.append(new_segment)

            # Update start time for the next sentence
            current_start_time = segment["start"]
            # Reset the current sentence and current words
            current_sentence = ""
            current_words = []

        for word in segment["words"]:
            cleaned_text = word["text"].replace("[*]", "")
            if cleaned_text.isdigit() and len(cleaned_text) > 18:
                # If the word is a number with more than 18 characters, split it
                split_words = [cleaned_text[i:i+18] for i in range(0, len(cleaned_text), 18)]
                current_words.extend([{"text": w, "start": word["start"], "end": word["end"]} for w in split_words])
            else:
                current_sentence += cleaned_text + " "
                current_words.append({"text": cleaned_text, "start": word["start"], "end": word["end"]})

            if any(char in ['.', '!', '?'] for char in cleaned_text):
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

def asr_segmentation(video_path, audio_path, temp_folder):
    logging.info(f"Loading Whisper Model")
    model = whisper.load_model("base", device="cuda")

    # Transcribe the entire audio
    logging.info(f"Transcribing audio")
    result = whisper.transcribe(model, audio_path, language="en", beam_size=20, best_of=20, vad="auditok", temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), detect_disfluencies=True)

    # Get timestamps from the transcribed segments
    logging.info(f"Formatting segments for translation")
    fortranslate = result["segments"]
    speech_timestamps = merge_segments(fortranslate)
    # Create the temp folder if not exists
    os.makedirs(os.path.join(temp_folder), exist_ok=True)

    translated_segments = []
    original_segments = []


    logging.info(f"Loading TTS for audio creation")
    translated_srt_path = os.path.basename(video_path).split('/')[-1] + "_tr.srt"
    translated_srt = pysrt.SubRipFile()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    translated_segments_paths = []

    for i, segment in enumerate(speech_timestamps):
        logging.info(f"Start of segmentation")
        start_time, end_time = segment['start'], segment['end']
        segment_path = os.path.join(temp_folder, f"segment_{i}.mp4")
        audio_output_path = os.path.join(temp_folder, f"segment_{i}.wav")
        translated_audio_output_path = os.path.join(temp_folder, f"tr_segment_{i}.wav")
        translated_segments_paths.append(segment_path)

        # Check if it's the last segment
        if i == len(speech_timestamps) - 1:
            # If it's the last segment, set the end time to the end of the video file
            videoclip = VideoFileClip(video_path)
            end_time = videoclip.duration

        #
        # Use MoviePy for video slicing without audio
        #
        logging.info(f"Slicing video")
        ffmpeg_command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
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
        logging.info(f"Slicing audio")
        ffmpeg_command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
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
        logging.info(f"Translating text using DeepL")
        translated_text = translator.translate_text(segment['text'], target_lang='tr').text

        logging.info(f"Creating audio file")
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


        logging.info(f"Speeding up or slowing down video for matching to audio")
        # Apply speed change to video using FFmpeg
        ffmpeg_speed_change_command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-i', segment_path,
            '-r', '60',
            '-c:v', 'h264_amf',
            '-vf', f'setpts={speed_factor}*PTS',
            'temp_slowed.mp4'
        ]
        subprocess.run(ffmpeg_speed_change_command)


        logging.info(f"Combining audio with new video")
        # Concatenate slowed video with translated audio using FFmpeg
        ffmpeg_concat_command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
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

        translated_end_time = start_time + translated_audio_duration
        end_time_seconds = int(translated_end_time)
        end_time_milliseconds = int((translated_end_time - end_time_seconds) * 1000)

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

def concatenate_video_ffmpeg(output_path):
    # Use ffmpeg to concatenate the video segments
    logging.info(f"Concatenate all videos")
    ffmpeg_command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-f', 'concat',
        '-safe', '0',  # Allow the use of unsafe file names
        '-i', 'segment_paths.txt',  # Use a pipe to pass the file paths
        '-c:v', 'h264_amf',
        '-c:a', 'aac',
        output_path
    ]

    # Use subprocess.PIPE to pass the file paths through a pipe
    subprocess.run(ffmpeg_command)

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: ruhsub <video_name>")
        sys.exit(1)

    script_dir = os.getcwd()
    os.chdir(script_dir)
    logging.info(f"Version 5.0.6")
    video_path = os.path.abspath(sys.argv[1])
    temp_path = os.path.join(script_dir, "temp/")
    audio_output_path = os.path.join(script_dir, "output_audio.wav")
    output_synced_video_path = os.path.join(script_dir, f"{os.path.basename(video_path).split('.')[0]}_tr.mp4")

    # Step 1: Extract audio from the video
    audio_path = extract_audio(video_path, audio_output_path)

    # Step 2: Perform ASR, translation, and segmentation
    original_segments, translated_segments, translated_segments_paths = asr_segmentation(video_path, audio_path, temp_path)

    # Step 3: Save the segmented MP4 paths to a list
    with open("segment_paths.txt", "w") as file:
        for path in translated_segments_paths:
            file.write(f"file '{path}'\n")

    # Step 4: Combine all the segmented MP4s into one MP4 using the list
    concatenate_video_ffmpeg(output_synced_video_path)

    shutil.rmtree(os.path.join(script_dir, "temp/"))
    os.remove(os.path.join(script_dir, "output_audio.wav"))
    os.remove(os.path.join(script_dir, "segment_paths.txt"))
    os.remove(os.path.join(script_dir, "temp_slowed.mp4"))


    logger.info("Processing completed")

if __name__ == "__main__":
    main()
