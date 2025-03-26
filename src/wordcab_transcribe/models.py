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
"""Models module of the Wordcab Transcribe."""

from enum import Enum
from typing import ClassVar, Literal, NamedTuple

import numpy as np
from faster_whisper.transcribe import Segment
from pydantic import BaseModel, HttpUrl, field_validator


class ProcessTimes(BaseModel):
    """The execution times of the different processes."""

    total: float | None = None
    transcription: float | None = None
    diarization: float | None = None
    post_processing: float | None = None


class Timestamps(str, Enum):
    """Timestamps enum for the API."""

    seconds = "s"
    milliseconds = "ms"
    hour_minute_second = "hms"


class Word(BaseModel):
    """Word model for the API."""

    word: str
    start: float
    end: float
    probability: float


class MultiChannelSegment(NamedTuple):
    """Multi-channel segment model for the API."""

    start: float
    end: float
    text: str
    words: list[Word]
    speaker: int


class Utterance(BaseModel):
    """Utterance model for the API."""

    text: str
    start: float | str
    end: float | str
    speaker: int | None = None
    words: list[Word] | None = None


class BaseResponse(BaseModel):
    """Base response model, not meant to be used directly."""

    utterances: list[Utterance]
    audio_duration: float
    offset_start: float | None
    offset_end: float | None
    num_speakers: int
    diarization: bool
    source_lang: str
    timestamps: str
    vocab: list[str] | None
    word_timestamps: bool
    internal_vad: bool
    repetition_penalty: float
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    condition_on_previous_text: bool
    process_times: ProcessTimes
    job_name: str | None = None
    task_token: str | None = None


class AudioResponse(BaseResponse):
    """Response model for the ASR audio file and url endpoint."""

    multi_channel: bool

    class Config:
        """Pydantic config class."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "utterances": [
                    {
                        "text": "Hello World!",
                        "start": 0.345,
                        "end": 1.234,
                        "speaker": 0,
                    },
                    {
                        "text": "Wordcab is awesome",
                        "start": 1.234,
                        "end": 2.678,
                        "speaker": 1,
                    },
                ],
                "audio_duration": 2.678,
                "batch_size": 1,
                "offset_start": None,
                "offset_end": None,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "process_times": {
                    "total": 2.678,
                    "transcription": 2.439,
                    "diarization": None,
                    "post_processing": 0.239,
                },
                "multi_channel": False,
            },
        }


class BaseRequest(BaseModel):
    """Base request model for the API."""

    offset_start: float | None = None
    offset_end: float | None = None
    num_speakers: int = -1
    diarization: bool = False
    batch_size: int = 1
    source_lang: str = "en"
    num_beams: int = 1
    timestamps: Timestamps = Timestamps.seconds
    vocab: list[str] | None = None
    word_timestamps: bool = False
    internal_vad: bool = False
    repetition_penalty: float = 1.2
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    job_name: str | None = None
    task_token: str | None = None

    @field_validator("vocab")
    @classmethod
    def validate_each_vocab_value(cls, value: list[str] | None) -> list[str]:
        """Validate the value of each vocab field."""
        if value == []:
            return None
        elif value is not None and not all(isinstance(v, str) for v in value):
            raise ValueError("`vocab` must be a list of strings.")  # noqa: TRY003 EM101

        return value

    class Config:
        """Pydantic config class."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "offset_start": None,
                "offset_end": None,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
            },
        }


class AudioRequest(BaseRequest):
    """Request model for the ASR audio file and url endpoint."""

    multi_channel: bool = False

    class Config:
        """Pydantic config class."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "batch_size": 1,
                "offset_start": None,
                "offset_end": None,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "multi_channel": False,
            },
        }


class PongResponse(BaseModel):
    """Response model for the ping endpoint."""

    message: str

    class Config:
        """Pydantic config class."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "message": "pong",
            },
        }


class UrlSchema(BaseModel):
    """Request model for the add_url endpoint."""

    task: Literal["transcription", "diarization"]
    url: HttpUrl


class DiarizationSegment(NamedTuple):
    """Diarization segment model for the API."""

    start: float
    end: float
    speaker: int


class DiarizationOutput(BaseModel):
    """Diarization output model for the API."""

    segments: list[DiarizationSegment]


class DiarizationRequest(BaseModel):
    """Request model for the diarize endpoint."""

    audio: np.ndarray | str
    audio_type: str | None
    duration: float
    num_speakers: int


class MultiChannelTranscriptionOutput(BaseModel):
    """Multi-channel transcription output model for the API."""

    segments: list[MultiChannelSegment]


class TranscriptionOutput(BaseModel):
    """Transcription output model for the API."""

    segments: list[Segment]


class TranscribeRequest(BaseModel):
    """Request model for the transcribe endpoint."""

    audio: np.ndarray | list[np.ndarray]
    compression_ratio_threshold: float
    condition_on_previous_text: bool
    internal_vad: bool
    log_prob_threshold: float
    no_speech_threshold: float
    repetition_penalty: float
    source_lang: str
    vocab: list[str] | None


class Token(BaseModel):
    """Token model for authentication."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """TokenData model for authentication."""

    username: str | None = None
