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
"""Transcribe endpoint for the Remote Wordcab Transcribe API."""

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from loguru import logger

from app.dependencies import asr
from app.models import TranscribeRequest, TranscriptionOutput
from app.services.asr_service import ProcessException

router = APIRouter()


@router.post(
    "",
    response_model=TranscriptionOutput | list[TranscriptionOutput] | str,
    status_code=http_status.HTTP_200_OK,
)
async def only_transcription(
    data: TranscribeRequest,
) -> TranscriptionOutput | list[TranscriptionOutput]:
    """Transcribe endpoint for the `only_transcription` asr type."""
    result: TranscriptionOutput | list[TranscriptionOutput] = await asr.process_input(data)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.message),
        )

    return result
