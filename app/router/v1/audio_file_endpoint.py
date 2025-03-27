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
"""Audio file endpoint for the Wordcab Transcribe API."""

import asyncio
from pathlib import Path
from typing import Annotated

import shortuuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi import status as http_status
from loguru import logger

from app.dependencies import asr, settings
from app.models import AudioRequest, AudioResponse
from app.services.asr_service import ProcessException
from app.utils import (
    check_num_channels,
    delete_file,
    process_audio_file,
    save_file_locally,
)

router = APIRouter()


@router.post("", response_model=AudioResponse | str, status_code=http_status.HTTP_200_OK)
async def inference_with_audio(  # noqa: PLR0913
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    batch_size: Annotated[int, Form()] = 1,
    offset_start: Annotated[float | None, Form()] = None,
    offset_end: Annotated[float | None, Form()] = None,
    num_speakers: Annotated[int, Form()] = -1,
    diarization: Annotated[bool, Form()] = False,
    multi_channel: Annotated[bool, Form()] = False,
    source_lang: Annotated[str, Form()] = "en",
    num_beams: Annotated[int, Form()] = 1,
    timestamps: Annotated[str, Form()] = "s",
    vocab: Annotated[list[str] | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = False,
    internal_vad: Annotated[bool, Form()] = False,
    repetition_penalty: Annotated[float, Form()] = 1.2,
    compression_ratio_threshold: Annotated[float, Form()] = 2.4,
    log_prob_threshold: Annotated[float, Form()] = -1.0,
    no_speech_threshold: Annotated[float, Form()] = 0.6,
    condition_on_previous_text: Annotated[bool, Form()] = True,
) -> AudioResponse:
    """Inference endpoint with audio file."""
    extension = file.filename.split(".")[-1] if file.filename is not None else "wav"
    filename = str(Path(settings.cache_folder) / f"audio_{shortuuid.ShortUUID().random(length=32)}.{extension}")

    await save_file_locally(filename=filename, file=file)

    num_channels = await check_num_channels(filename)
    if (num_channels > 1 and multi_channel is False) or (num_channels == 1 and multi_channel is True):
        num_channels = 1  # Force mono channel if more than 1 channel or vice versa
        if multi_channel:
            diarization = True
        multi_channel = False

    try:
        filepath: str | list[str] = await process_audio_file(filename, num_channels=num_channels)
    except Exception as e:
        logger.exception(e)
        try:
            background_tasks.add_task(delete_file, filepath=filename)
            background_tasks.add_task(delete_file, filepath=filepath)
        except Exception:
            logger.debug("Failed to delete file")
        raise HTTPException(  # noqa: B904
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Process failed: {e}",
        )

    data = AudioRequest(
        offset_start=offset_start,
        offset_end=offset_end,
        num_speakers=num_speakers,
        diarization=diarization,
        batch_size=batch_size,
        source_lang=source_lang,
        num_beams=num_beams,
        timestamps=timestamps,
        vocab=vocab,
        word_timestamps=word_timestamps,
        internal_vad=internal_vad,
        repetition_penalty=repetition_penalty,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        multi_channel=multi_channel,
    )

    background_tasks.add_task(delete_file, filepath=filename)
    task = asyncio.create_task(
        asr.process_input(
            filepath=filepath,
            offset_start=data.offset_start,
            offset_end=data.offset_end,
            num_speakers=data.num_speakers,
            diarization=data.diarization,
            batch_size=data.batch_size,
            multi_channel=data.multi_channel,
            source_lang=data.source_lang,
            num_beams=data.num_beams,
            timestamps_format=data.timestamps,
            vocab=data.vocab,
            word_timestamps=data.word_timestamps,
            internal_vad=data.internal_vad,
            repetition_penalty=data.repetition_penalty,
            compression_ratio_threshold=data.compression_ratio_threshold,
            log_prob_threshold=data.log_prob_threshold,
            no_speech_threshold=data.no_speech_threshold,
            condition_on_previous_text=data.condition_on_previous_text,
        ),
    )
    result = await task

    background_tasks.add_task(delete_file, filepath=filepath)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.message),
        )

    utterances, process_times, audio_duration = result
    result = AudioResponse(
        utterances=utterances,
        audio_duration=audio_duration,
        offset_start=data.offset_start,
        offset_end=data.offset_end,
        num_speakers=data.num_speakers,
        diarization=data.diarization,
        batch_size=data.batch_size,
        multi_channel=data.multi_channel,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
        vocab=data.vocab,
        word_timestamps=data.word_timestamps,
        internal_vad=data.internal_vad,
        repetition_penalty=data.repetition_penalty,
        compression_ratio_threshold=data.compression_ratio_threshold,
        log_prob_threshold=data.log_prob_threshold,
        no_speech_threshold=data.no_speech_threshold,
        condition_on_previous_text=data.condition_on_previous_text,
        process_times=process_times,
    )

    if settings.debug:
        logger.debug(f"Result: {result.model_dump()}")

    return result
