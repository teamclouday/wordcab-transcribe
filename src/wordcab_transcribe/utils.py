# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.
"""Utils module of the Wordcab Transcribe."""

import asyncio
import io
import re
import subprocess
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiohttp
import huggingface_hub
import requests
import soundfile as sf
import torch
import torchaudio
from loguru import logger

if TYPE_CHECKING:
    from fastapi import UploadFile

from wordcab_transcribe.models import (
    Timestamps,
    TranscriptionOutput,
    Utterance,
    Word,
)


# pragma: no cover
async def async_run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess asynchronously.

    Args:
        command (List[str]): Command to run.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await process.communicate()

    return process.returncode, stdout, stderr


# pragma: no cover
def run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess synchronously.

    Args:
        command (List[str]): Command to run.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = subprocess.Popen(  # noqa: S603,S607
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    return process.returncode, stdout, stderr


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available on the system."""
    try:
        result = run_subprocess(["ffmpeg", "-version"])

        if result[0] != 0:
            raise subprocess.CalledProcessError(result[0], "ffmpeg")

        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,  # noqa: S404
    ):
        return False


async def check_num_channels(filepath: Union[str, Path]) -> int:
    """Check the number of channels in an audio file."""
    if isinstance(filepath, str):
        _filepath = Path(filepath)

    if not _filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_streams",
        filepath,
    ]
    returncode, stdout, stderr = await async_run_subprocess(cmd)

    if returncode != 0:
        raise Exception(f"Error converting file {filepath} to wav format: {stderr}")

    output = stdout.decode("utf-8")
    for line in output.split("\n"):
        if line.startswith("channels"):
            return int(line.split("=")[1])


def convert_timestamp(timestamp: float, target: Timestamps, round_digits: Optional[int] = 3) -> Union[str, float]:
    """
    Use the right function to convert the timestamp.

    Args:
        timestamp (float): Timestamp to convert.
        target (Timestamps): Target timestamp format.
        round_digits (int, optional): Number of digits to round the timestamp. Defaults to 3.

    Returns:
        Union[str, float]: Converted timestamp.

    Raises:
        ValueError: If the target is invalid. Valid targets are: ms, hms, s.
    """
    if target == Timestamps.milliseconds:
        return round(_convert_s_to_ms(timestamp), round_digits)
    elif target == Timestamps.hour_minute_second:
        return _convert_s_to_hms(timestamp)
    elif target == Timestamps.seconds:
        return round(timestamp, round_digits)
    else:
        raise ValueError(f"Invalid conversion target: {target}. Valid targets are: ms, hms, s.")


def _convert_s_to_ms(timestamp: float) -> float:
    """
    Convert a timestamp from seconds to milliseconds.

    Args:
        timestamp (float): Timestamp in seconds to convert.

    Returns:
        float: Milliseconds.
    """
    return timestamp * 1000


def _convert_s_to_hms(timestamp: float) -> str:
    """
    Convert a timestamp from seconds to hours, minutes and seconds.

    Args:
        timestamp (float): Timestamp in seconds to convert.

    Returns:
        str: Hours, minutes and seconds.
    """
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    ms = (seconds - int(seconds)) * 1000

    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(ms):03}"

    return output


# pragma: no cover
async def download_audio_file(
    source: str,
    url: str,
    filename: str,
    url_headers: Optional[Dict[str, str]] = None,
) -> Union[str, Awaitable[str]]:
    """
    Download an audio file from a URL.

    Args:
        source (str): Source of the audio file. Can be "url".
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (Optional[Dict[str, str]]): Headers to send with the request. Defaults to None.

    Raises:
        ValueError: If the source is invalid. Valid sources is url.

    Returns:
        Union[str, Awaitable[str]]: Path to the downloaded file.
    """
    if source == "url":
        filename = await _download_file_from_url(url, filename, url_headers)
    else:
        raise ValueError(f"Invalid source: {source}. Valid sources is url.")

    return filename


# pragma: no cover
def download_audio_file_sync(
    source: str,
    url: str,
    filename: str,
) -> str | Awaitable[str]:
    """
    Download an audio file from a URL.

    Args:
        source (str): Source of the audio file. Can be "url".
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.

    Raises:
        ValueError: If the source is invalid. Valid sources is url.

    Returns:
        Union[str, Awaitable[str]]: Path to the downloaded file.
    """
    if source == "url":
        filename = _download_file_from_url_sync(url, filename)
    else:
        raise ValueError(f"Invalid source: {source}. Valid sources is url.")

    return filename


# pragma: no cover
async def _download_file_from_url(
    url: str,
    filename: str,
    url_headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Download a file from a URL using aiohttp.

    Args:
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (Optional[Dict[str, str]]): Headers to send with the request. Defaults to None.

    Returns:
        str: Path to the downloaded file.

    Raises:
        Exception: If the file failed to download.
    """
    url_headers = url_headers or {}

    logger.info(f"Downloading audio file from {url} to {filename}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=url_headers) as response:
            if response.status == 200:
                async with aiofiles.open(filename, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)

                        if not chunk:
                            break

                        await f.write(chunk)
            else:
                raise Exception(f"Failed to download file. Status: {response.status}")

    return filename


def _download_file_from_url_sync(url: str, filename: str, url_headers: Optional[Dict[str, str]] = None) -> str:
    """
    Download a file from a URL using requests.

    Args:
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (Optional[Dict[str, str]]): Headers to send with the request. Defaults to None.

    Returns:
        str: Path to the downloaded file.

    Raises:
        Exception: If the file failed to download.
    """
    url_headers = url_headers or {}

    logger.info(f"Downloading audio file from {url} to {filename}...")

    response = requests.get(url, headers=url_headers, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    else:
        raise Exception(f"Failed to download file. Status: {response.status_code}")

    return filename


# pragma: no cover
def download_model(compute_type: str, language: str) -> Optional[str]:
    """
    Download the special models during the warmup phase.

    Args:
        compute_type (str): The target compute type.
        language (str): The target language.

    Returns:
        Optional[str]: The path to the downloaded model.
    """
    if compute_type == "float16":
        repo_id = f"wordcab/whisper-large-fp16-{language}"
    elif compute_type == "int8_float16":
        repo_id = f"wordcab/whisper-large-int8-fp16-{language}"
    elif compute_type == "int8":
        repo_id = f"wordcab/whisper-large-int8-{language}"
    else:
        # No other models are supported
        return None

    allow_patterns = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
    snapshot_path = huggingface_hub.snapshot_download(repo_id, local_files_only=False, allow_patterns=allow_patterns)

    return snapshot_path


def delete_file(filepath: Union[str, Tuple[str, Optional[str]]]) -> None:
    """
    Delete a file or a list of files.

    Args:
        filepath (Union[str, Tuple[str]]): Path to the file to delete.
    """
    if isinstance(filepath, str):
        filepath = (filepath, None)

    for path in filepath:
        if path:
            Path(path).unlink(missing_ok=True)


def early_return(duration: float) -> Tuple[List[dict], dict, float]:
    """
    Early return for empty audio files.

    Args:
        duration (float): Duration of the audio file.

    Returns:
        Tuple[List[dict], dict, float]:
            Empty segments, process times and audio duration.
    """
    return (
        [
            {
                "text": "<EMPTY AUDIO>",
                "start": 0,
                "end": duration,
                "speaker": None,
                "words": None,
            }
        ],
        {
            "total": 0,
            "transcription": 0,
            "diarization": None,
            "post_processing": 0,
        },
        duration,
    )


def is_empty_string(text: str):
    """
    Checks if a string is empty after removing spaces and periods.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the string is empty, False otherwise.
    """
    text = text.replace(".", "")
    text = re.sub(r"\s+", "", text)
    if text.strip():
        return False
    return True


def format_punct(text: str):
    """
    Removes Whisper's '...' output, and checks for weird spacing in punctuation. Also removes extra spaces.

    Args:
        text (str): The text to format.

    Returns:
        str: The formatted text.
    """
    text = text.strip()

    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in [".", "?", "!", ":", ";", ","]:
        text += "."

    text = text.replace("...", "")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bi\b", "I", text)

    return text.strip()


def format_segments(transcription_output: TranscriptionOutput) -> List[Utterance]:
    """
    Format the segments to a list of dicts with start, end and text keys. Optionally include word timestamps.

    Args:
        transcription_output (TranscriptionOutput): List of segments.
        word_timestamps (bool): Whether to include word timestamps.

    Returns:
        List[Utterance]: List of formatted segments.
    """
    formatted_segments = [
        Utterance(
            text=segment.text,
            start=segment.start,
            end=segment.end,
            words=[
                Word(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                )
                for word in segment.words
            ],
        )
        for segment in transcription_output.segments
    ]

    return formatted_segments


async def process_audio_file(filepath: str, num_channels: int = 1) -> Union[str, List[str]]:
    """Prepare the audio for inference.

    Process an audio file using ffmpeg. The file will be converted to WAV if
    num_channels is 1, or split into N channels if num_channels >= 2.
    The codec used is pcm_s16le and the sample rate is 16000.

    Args:
        filepath (str):
            Path to the file to process.
        num_channels (int):
            The number of channels desired. 1 for conversion only,
            >= 2 for splitting and conversion.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there's an error in processing.

    Returns:
        Union[str, List[str]]: Path to the converted/split files.
    """
    _filepath = Path(filepath)

    if not _filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    # Convert to WAV if num_channels is 1
    if num_channels == 1:
        new_filepath = f"{_filepath.stem}_{_filepath.stat().st_mtime_ns}.wav"
        cmd = [
            "ffmpeg",
            "-i",
            filepath,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            new_filepath,
        ]

        result = await async_run_subprocess(cmd)
        if result[0] != 0:
            raise Exception(f"Error converting file {filepath} to wav format: {result[2]}")

        return new_filepath

    # Split audio into N channels if num_channels >= 2
    else:
        output_files = [f"{_filepath.stem}_ch{channel}.wav" for channel in range(num_channels)]

        cmd = ["ffmpeg", "-i", filepath]
        for channel in range(num_channels):
            cmd.extend(
                [
                    "-map_channel",
                    f"0.0.{channel}",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    output_files[channel],
                ]
            )

        result = await async_run_subprocess(cmd)
        if result[0] != 0:
            raise Exception(f"Error splitting {num_channels}-channel file: {filepath}. {result[2]}")

        return output_files


def process_audio_file_sync(filepath: str) -> Union[str, List[str]]:
    """Prepare the audio for inference.

    Process an audio file using ffmpeg. The file will be converted to WAV.
    The codec used is pcm_s16le and the sample rate is 16000.

    Args:
        filepath (str):
            Path to the file to process.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there's an error in processing.

    Returns:
        Union[str, List[str]]: Path to the converted/split files.
    """
    _filepath = Path(filepath)

    if not _filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    new_filepath = f"{_filepath.stem}_{_filepath.stat().st_mtime_ns}.wav"
    cmd = [
        "ffmpeg",
        "-i",
        filepath,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        new_filepath,
    ]

    result = run_subprocess(cmd)

    if result[0] != 0:
        raise Exception(f"Error converting file {filepath} to wav format: {result[2]}")

    return new_filepath


def read_audio(
    audio: Union[str, bytes],
    offset_start: Union[float, None] = None,
    offset_end: Union[float, None] = None,
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, float]:
    """
    Read an audio file and return the audio tensor.

    Args:
        audio (Union[str, bytes]):
            Path to the audio file or the audio bytes.
        offset_start (Union[float, None], optional):
            When to start reading the audio file. Defaults to None.
        offset_end (Union[float, None], optional):
            When to stop reading the audio file. Defaults to None.
        sample_rate (int):
            The sample rate of the audio file. Defaults to 16000.

    Returns:
        Tuple[torch.Tensor, float]: The audio tensor and the audio duration.
    """
    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    elif isinstance(audio, bytes):
        with io.BytesIO(audio) as buffer:
            wav, sr = sf.read(buffer, format="RAW", channels=1, samplerate=16000, subtype="PCM_16")
        wav = torch.from_numpy(wav).unsqueeze(0)
    else:
        raise ValueError(f"Invalid audio type. Must be either str or bytes, got: {type(audio)}.")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = transform(wav)
        sr = sample_rate

    audio_duration = float(wav.shape[1]) / sample_rate

    # Convert start and end times to sample indices based on the new sample rate
    if offset_start is not None:
        start_sample = int(offset_start * sr)
    else:
        start_sample = 0

    if offset_end is not None:
        end_sample = int(offset_end * sr)
    else:
        end_sample = wav.shape[1]

    # Trim the audio based on the new start and end samples
    wav = wav[:, start_sample:end_sample]

    return wav.squeeze(0), audio_duration


def retrieve_user_platform() -> str:
    """
    Retrieve the user's platform.

    Returns:
        str: User's platform. Either 'linux', 'darwin' or 'win32'.
    """
    return sys.platform


async def save_file_locally(filename: str, file: "UploadFile") -> bool:
    """
    Save a file locally from an UploadFile object.

    Args:
        filename (str): The filename to save the file as.
        file (UploadFile): The UploadFile object.

    Returns:
        bool: Whether the file was saved successfully.
    """
    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    return True
