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
"""Configuration module of the Wordcab Transcribe."""

from os import getenv
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic import field_validator
from pydantic.dataclasses import dataclass

from app import __version__


@dataclass
class Settings:
    """Configuration settings for the Wordcab Transcribe API."""

    # General configuration
    project_name: str
    version: str
    description: str
    api_prefix: str
    debug: bool
    cache_folder: str
    # Models configuration
    # Whisper
    whisper_model: str
    whisper_live_model: str
    whisper_engine: str
    whisper_live_engine: str
    align_model: str
    compute_type: str
    extra_languages: list[str] | None
    extra_languages_model_paths: dict[str, str] | None
    # Diarization
    diarization_backend: str
    window_lengths: list[float]
    shift_lengths: list[float]
    multiscale_weights: list[float]
    # Post-processing
    enable_punctuation_based_alignment: bool
    # ASR type configuration
    asr_type: Literal["async", "only_transcription", "only_diarization"]
    # API authentication configuration
    username: str
    password: str
    openssl_key: str
    openssl_algorithm: str
    access_token_expire_minutes: int

    @field_validator("project_name")
    @classmethod
    def project_name_must_not_be_none(cls, value: str) -> str:
        """Check that the project_name is not None."""
        if value is None:
            raise ValueError("`project_name` must not be None, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    @field_validator("cache_folder")
    @classmethod
    def cache_folder_should_exist(cls, value: str) -> str:
        """Check that the cache_folder exists. If not, create one"""
        Path(value).mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("whisper_model")
    @classmethod
    def whisper_model_compatibility_check(cls, value: str) -> str:
        """Check that the whisper model is compatible."""
        if (
            value.lower()
            not in [
                "tiny",
                "tiny.en",
                "base",
                "base.en",
                "small",
                "small.en",
                "medium",
                "medium.en",
                "large",
                "large-v1",
                "large-v2",
                "large-v3",
                "distil-medium.en",
                "distil-large-v2",
                "distil-large-v3",
            ]
            and "/" not in value
        ):
            raise ValueError(  # noqa: TRY003
                "The whisper model must be one of `tiny`, `tiny.en`, `base`,"  # noqa: EM101
                " `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`,"
                " `large-v1`, `large-v2`, `large-v3`, `distil-medium.en`, `distil-large-v2`, or"
                " `distil-large-v3`.",
            )

        return value

    @field_validator("whisper_live_model")
    @classmethod
    def whisper_live_model_compatibility_check(cls, value: str) -> str:
        """Check that the whisper model is compatible."""
        if (
            value.lower()
            not in [
                "tiny",
                "tiny.en",
                "base",
                "base.en",
                "small",
                "small.en",
                "medium",
                "medium.en",
                "large",
                "large-v1",
                "large-v2",
                "large-v3",
                "distil-medium.en",
                "distil-large-v2",
                "distil-large-v3",
            ]
            and "/" not in value
        ):
            raise ValueError(  # noqa: TRY003
                "The whisper live model must be one of `tiny`, `tiny.en`, `base`,"  # noqa: EM101
                " `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`,"
                " `large-v1`, `large-v2`, `large-v3`, `distil-medium.en`, `distil-large-v2`, or"
                " `distil-large-v3`.",
            )

        return value

    @field_validator("whisper_engine")
    @classmethod
    def whisper_engine_compatibility_check(cls, value: str) -> str:
        """Check that the whisper engine is compatible."""
        if value.lower() not in [
            "faster-whisper",
            "faster-whisper-batched",
        ]:
            raise ValueError(  # noqa: TRY003
                "The whisper engine must be one of `faster-whisper` or `faster-whisper-batched`.",  # noqa: EM101
            )

        return value

    @field_validator("whisper_live_engine")
    @classmethod
    def whisper_live_engine_compatibility_check(cls, value: str) -> str:
        """Check that the whisper engine is compatible."""
        if value.lower() not in [
            "faster-whisper",
            "faster-whisper-batched",
        ]:
            raise ValueError(  # noqa: TRY003
                "The whisper live engine must be one of `faster-whisper` or `faster-whisper-batched`.",  # noqa: EM101
            )

        return value

    @field_validator("align_model")
    @classmethod
    def align_model_compatibility_check(cls, value: str) -> str:
        """Check that the whisper engine is compatible."""
        if value.lower() not in ["tiny", "small", "base", "medium"]:
            raise ValueError("The align model must be one of `tiny`, `small`, `base`, or `medium`.")  # noqa: TRY003 EM101

        return value

    @field_validator("diarization_backend")
    @classmethod
    def diarization_backend_compatibility_check(cls, value: str) -> str:
        """Check that the diarization engine is compatible."""
        if value.lower() not in ["default-diarizer", "longform-diarizer"]:
            raise ValueError("The diarization backend must be one of `default_diarizer` or `longform_diarizer`.")  # noqa: TRY003 EM101

        return value

    @field_validator("version")
    @classmethod
    def version_must_not_be_none(cls, value: str) -> str:
        """Check that the version is not None."""
        if value is None:
            raise ValueError("`version` must not be None, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    @field_validator("description")
    @classmethod
    def description_must_not_be_none(cls, value: str) -> str:
        """Check that the description is not None."""
        if value is None:
            raise ValueError("`description` must not be None, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    @field_validator("api_prefix")
    @classmethod
    def api_prefix_must_not_be_none(cls, value: str) -> str:
        """Check that the api_prefix is not None."""
        if value is None:
            raise ValueError("`api_prefix` must not be None, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    @field_validator("compute_type")
    @classmethod
    def compute_type_must_be_valid(cls, value: str) -> str:
        """Check that the model precision is valid."""
        compute_type_values = [
            "int8",
            "int8_float16",
            "int8_bfloat16",
            "int16",
            "float16",
            "bfloat16",
            "float32",
        ]
        if value not in compute_type_values:
            raise ValueError(f"{value} is not a valid compute type. Choose one of {compute_type_values}.")  # noqa: TRY003 EM102

        return value

    @field_validator("openssl_algorithm")
    @classmethod
    def openssl_algorithm_must_be_valid(cls, value: str) -> str:
        """Check that the OpenSSL algorithm is valid."""
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError("openssl_algorithm must be a valid algorithm, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    @field_validator("access_token_expire_minutes")
    @classmethod
    def access_token_expire_minutes_must_be_valid(cls, value: int) -> int:
        """Check that the access token expiration is valid. Only if debug is False."""
        if value <= 0:
            raise ValueError("access_token_expire_minutes must be positive, please verify the `.env` file.")  # noqa: TRY003 EM101

        return value

    def __post_init__(self) -> None:
        """Post initialization checks."""
        if self.debug is False:
            if self.username == "admin" or self.username is None:
                logger.warning(f"Username is set to `{self.username}`, which is not secure for production.")
            if self.password == "admin" or self.password is None:  # noqa: S105
                logger.warning(f"Password is set to `{self.password}`, which is not secure for production.")
            if self.openssl_key == "0123456789abcdefghijklmnopqrstuvwyz" or self.openssl_key is None:
                logger.warning(
                    f"OpenSSL key is set to `{self.openssl_key}`, which is the default"
                    " encryption key. It's absolutely not secure for production."
                    " Please change it in the `.env` file. You can generate a new key"
                    " with `openssl rand -hex 32`.",
                )

        if len(self.window_lengths) != len(self.shift_lengths) != len(self.multiscale_weights):
            raise ValueError(  # noqa: TRY003
                "Length of window_lengths, shift_lengths and multiscale_weights must"  # noqa: EM102
                f" be the same.\nFound: {len(self.window_lengths)},"
                f" {len(self.shift_lengths)}, {len(self.multiscale_weights)}",
            )


load_dotenv()

# Extra languages
_extra_languages = getenv("EXTRA_LANGUAGES", None)
if _extra_languages is not None and _extra_languages != "":
    extra_languages = [lang.strip() for lang in _extra_languages.split(",")]
else:
    extra_languages = None

extra_languages_model_paths = dict.fromkeys(extra_languages, "") if extra_languages is not None else None

# Diarization scales
_window_lengths = getenv("WINDOW_LENGTHS", None)
if _window_lengths is not None:
    window_lengths = [float(x.strip()) for x in _window_lengths.split(",")]
else:
    window_lengths = [1.5, 1.25, 1.0, 0.75, 0.5]

_shift_lengths = getenv("SHIFT_LENGTHS", None)
if _shift_lengths is not None:
    shift_lengths = [float(x.strip()) for x in _shift_lengths.split(",")]
else:
    shift_lengths = [0.75, 0.625, 0.5, 0.375, 0.25]

_multiscale_weights = getenv("MULTISCALE_WEIGHTS", None)
if _multiscale_weights is not None:
    multiscale_weights = [float(x.strip()) for x in _multiscale_weights.split(",")]
else:
    multiscale_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

settings = Settings(
    # General configuration
    project_name=getenv("PROJECT_NAME", "Wordcab Transcribe"),
    version=getenv("VERSION", __version__),
    description=getenv(
        "DESCRIPTION",
        "💬 ASR FastAPI server using faster-whisper and Auto-Tuning Spectral Clustering for diarization.",
    ),
    api_prefix=getenv("API_PREFIX", "/api/v1"),
    debug=getenv("DEBUG", "True").lower() == "true",
    cache_folder=getenv("CACHE_FOLDER", ".cache"),
    # Models configuration
    # Transcription
    whisper_model=getenv("WHISPER_MODEL", "distil-large-v2"),
    whisper_live_model=getenv("WHISPER_LIVE_MODEL", "distil-large-v2"),
    whisper_engine=getenv("WHISPER_ENGINE", "faster-whisper-batched"),
    whisper_live_engine=getenv("WHISPER_LIVE_ENGINE", "faster-whisper"),
    align_model=getenv("ALIGN_MODEL", "tiny"),
    compute_type=getenv("COMPUTE_TYPE", "float16"),
    extra_languages=extra_languages,
    extra_languages_model_paths=extra_languages_model_paths,
    # Diarization
    diarization_backend=getenv("DIARIZATION_BACKEND", "longform-diarizer"),
    window_lengths=window_lengths,
    shift_lengths=shift_lengths,
    multiscale_weights=multiscale_weights,
    # Post-processing
    enable_punctuation_based_alignment=getenv("ENABLE_PUNCTUATION_BASED_ALIGNMENT", "True").lower() == "true",
    # ASR type
    asr_type=getenv("ASR_TYPE", "async"),
    # API authentication configuration
    username=getenv("USERNAME", "admin"),
    password=getenv("PASSWORD", "admin"),
    openssl_key=getenv("OPENSSL_KEY", "0123456789abcdefghijklmnopqrstuvwyz"),
    openssl_algorithm=getenv("OPENSSL_ALGORITHM", "HS256"),
    access_token_expire_minutes=int(getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
)
