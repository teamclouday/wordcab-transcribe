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
"""Audio url endpoint for the Wordcab Transcribe API."""

import asyncio
import contextlib

import shortuuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.dependencies import asr, download_limit
from wordcab_transcribe.models import (
    AudioRequest,
    AudioResponse,
)
from wordcab_transcribe.utils import (
    check_num_channels,
    delete_file,
    download_audio_file,
    process_audio_file,
)

router = APIRouter()


@router.post("", status_code=http_status.HTTP_202_ACCEPTED)
async def inference_with_audio_url(
    background_tasks: BackgroundTasks,
    url: str,
    data: AudioRequest | None = None,
) -> dict:
    """Inference endpoint with audio url."""
    filename = f"audio_url_{shortuuid.ShortUUID().random(length=32)}"
    data = AudioRequest() if data is None else AudioRequest(**data.model_dump())

    async def process_audio(data: AudioRequest) -> None:
        try:
            async with download_limit:
                _filepath = await download_audio_file("url", url, filename)

                num_channels = await check_num_channels(_filepath)
                if (num_channels > 1 and data.multi_channel is False) or (
                    num_channels == 1 and data.multi_channel is True
                ):
                    num_channels = 1  # Force mono channel if more than 1 channel or vice versa
                    new_data = data.model_dump()
                    if data.multi_channel:
                        new_data["diarization"] = True
                    new_data["multi_channel"] = False
                    data = AudioRequest(**new_data)

                try:
                    filepath: str | list[str] = await process_audio_file(_filepath, num_channels=num_channels)
                except Exception as e:
                    try:
                        background_tasks.add_task(delete_file, filepath=filename)
                        background_tasks.add_task(delete_file, filepath=filepath)
                    except Exception:
                        logger.debug("Failed to delete files")
                    raise HTTPException(  # noqa: B904
                        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Process failed: {e}",
                    )

                background_tasks.add_task(delete_file, filepath=filename)

                task = asyncio.create_task(
                    asr.process_input(
                        filepath=filepath,
                        url=url,
                        url_type="url",
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
                    job_name=data.job_name,
                    task_token=data.task_token,
                    process_times=process_times,
                )

                if settings.debug:
                    logger.debug(f"Result: {result.model_dump()}")

                background_tasks.add_task(delete_file, filepath=filepath)
        except Exception as e:
            try:
                background_tasks.add_task(delete_file, filepath=filename)
                background_tasks.add_task(delete_file, filepath=filepath)
            except Exception:
                logger.debug("Failed to delete files")
            error_message = f"Error during transcription: {e}"
            with contextlib.suppress(Exception):
                logger.error(result.message)
            logger.error(error_message)

    # Add the process_audio function to background tasks
    background_tasks.add_task(process_audio, data)

    # Return the job name and task token immediately
    return {"job_name": data.job_name, "task_token": data.task_token}
