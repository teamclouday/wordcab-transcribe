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
"""Dependencies for Wordcab Transcribe."""

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.services.asr_service import (
    ASRAsyncService,
    ASRDiarizationOnly,
    ASRLiveService,
    ASRTranscriptionOnly,
)
from app.utils import check_ffmpeg, download_model

# Define the ASR service to use depending on the settings
asr_live = ASRLiveService(
    whisper_model=settings.whisper_live_model,
    compute_type=settings.compute_type,
    debug_mode=settings.debug,
)
if settings.asr_type == "async":
    asr = ASRAsyncService(
        whisper_model=settings.whisper_model,
        compute_type=settings.compute_type,
        window_lengths=settings.window_lengths,
        shift_lengths=settings.shift_lengths,
        multiscale_weights=settings.multiscale_weights,
        extra_languages=settings.extra_languages,
        extra_languages_model_paths=settings.extra_languages_model_paths,
        debug_mode=settings.debug,
    )
elif settings.asr_type == "only_transcription":
    asr = ASRTranscriptionOnly(
        whisper_model=settings.whisper_model,
        compute_type=settings.compute_type,
        extra_languages=settings.extra_languages,
        extra_languages_model_paths=settings.extra_languages_model_paths,
        debug_mode=settings.debug,
    )
elif settings.asr_type == "only_diarization":
    asr = ASRDiarizationOnly(
        window_lengths=settings.window_lengths,
        shift_lengths=settings.shift_lengths,
        multiscale_weights=settings.multiscale_weights,
        debug_mode=settings.debug,
    )
else:
    raise ValueError(f"Invalid ASR type: {settings.asr_type}")  # noqa: TRY003 EM102


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Context manager to handle the startup and shutdown of the application."""
    if settings.asr_type in ["async", "only_transcription"]:
        if check_ffmpeg() is False:
            logger.warning(
                "FFmpeg is not installed on the host machine.\n"
                "Please install it and try again: `sudo apt-get install ffmpeg`",
            )
            sys.exit(1)

        if settings.extra_languages is not None:
            logger.info("Downloading models for extra languages...")
            for model in settings.extra_languages:
                try:
                    model_path = download_model(compute_type=settings.compute_type, language=model)

                    if model_path is not None:
                        settings.extra_languages_model_paths[model] = model_path
                    else:
                        raise Exception(f"Coudn't download model for {model}")  # noqa: TRY301 TRY003 TRY002 EM102

                except Exception as e:
                    logger.error(f"Error downloading model for {model}: {e}")

    # Define the maximum number of files to pre-download for the async ASR service
    app.state.download_limit = asyncio.Semaphore(10)

    logger.info("Warmup initialization...")
    await asr.inference_warmup()

    yield  # This is where the execution of the application starts

    logger.info("Application shutdown...")
