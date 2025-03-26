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
"""ASR Service module that handle all AI interactions."""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import aiohttp
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from wordcab_transcribe.config import settings
from wordcab_transcribe.logging import time_and_tell, time_and_tell_async
from wordcab_transcribe.models import (
    DiarizationOutput,
    DiarizationRequest,
    ProcessTimes,
    Timestamps,
    TranscribeRequest,
    TranscriptionOutput,
    UrlSchema,
    Utterance,
)
from wordcab_transcribe.services.concurrency_services import GPUService, URLService
from wordcab_transcribe.services.diarization.diarize_service import DiarizeService
from wordcab_transcribe.services.longform_diarization.diarize_service import (
    LongFormDiarizeService,
)
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import early_return, format_segments, read_audio


class AsyncLocationTrustedRedirectSession(aiohttp.ClientSession):
    async def _request(
        self,
        method: str,
        url: str,
        location_trusted: bool,
        *args: Any,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        if not location_trusted:
            return await super()._request(method, url, *args, **kwargs)
        kwargs["allow_redirects"] = False
        response = await super()._request(method, url, *args, **kwargs)
        if response.status in (301, 302, 303, 307, 308) and "Location" in response.headers:
            new_url = response.headers["Location"]
            return await super()._request(method, new_url, *args, **kwargs)
        return response


class ExceptionSource(str, Enum):
    """Exception source enum."""

    add_url = "add_url"
    diarization = "diarization"
    get_url = "get_url"
    post_processing = "post_processing"
    remove_url = "remove_url"
    transcription = "transcription"


class ProcessException(BaseModel):
    """Process exception model."""

    source: ExceptionSource
    message: str


class LocalExecution(BaseModel):
    """Local execution model."""

    index: int | None


class RemoteExecution(BaseModel):
    """Remote execution model."""

    url: str


class ASRTask(BaseModel):
    """ASR Task model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: np.ndarray | list[np.ndarray]
    url: str | None
    url_type: str | None
    diarization: "DiarizationTask"
    duration: float
    batch_size: int
    multi_channel: bool
    offset_start: float | None
    post_processing: "PostProcessingTask"
    process_times: ProcessTimes
    timestamps_format: Timestamps
    transcription: "TranscriptionTask"
    word_timestamps: bool


class DiarizationTask(BaseModel):
    """Diarization Task model."""

    execution: LocalExecution | RemoteExecution | None
    num_speakers: int
    result: ProcessException | DiarizationOutput | None = None


class PostProcessingTask(BaseModel):
    """Post Processing Task model."""

    result: ProcessException | list[Utterance] | None = None


class TranscriptionOptions(BaseModel):
    """Transcription options model."""

    compression_ratio_threshold: float
    condition_on_previous_text: bool
    internal_vad: bool
    log_prob_threshold: float
    no_speech_threshold: float
    repetition_penalty: float
    source_lang: str
    num_beams: int
    vocab: list[str] | None


class TranscriptionTask(BaseModel):
    """Transcription Task model."""

    execution: LocalExecution | RemoteExecution
    options: TranscriptionOptions
    result: ProcessException | TranscriptionOutput | list[TranscriptionOutput] | None = None


@dataclass
class LocalServiceRegistry:
    """Registry for local services."""

    diarization: DiarizeService | LongFormDiarizeService | None = None
    post_processing: PostProcessingService = None
    transcription: TranscribeService | None = None
    vad: VadService = None

    def __post_init__(self) -> None:
        self.post_processing = PostProcessingService()
        self.vad = VadService()


@dataclass
class RemoteServiceConfig:
    """Remote service config."""

    url_handler: URLService | None = None
    use_remote: bool = False

    def get_urls(self) -> list[str]:
        """Get the list of URLs."""
        return self.url_handler.get_urls()

    def get_queue_size(self) -> int:
        """Get the queue size."""
        return self.url_handler.get_queue_size()

    async def add_url(self, url: str) -> None:
        """Add a URL to the list of URLs."""
        await self.url_handler.add_url(url)

    async def next_url(self) -> str:
        """Get the next URL."""
        return await self.url_handler.next_url()

    async def remove_url(self, url: str) -> None:
        """Remove a URL from the list of URLs."""
        await self.url_handler.remove_url(url)


@dataclass
class RemoteServiceRegistry:
    """Registry for remote services."""

    diarization: RemoteServiceConfig = None
    transcription: RemoteServiceConfig = None

    def __post_init__(self) -> None:
        self.diarization = RemoteServiceConfig()
        self.transcription = RemoteServiceConfig()


class ASRService(ABC):
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated. Use the subclasses instead.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Do we have a GPU? If so, use it!
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        logger.info(f"NVIDIA GPUs available: {self.num_gpus}")

        if self.num_gpus > 1 and self.device == "cuda":
            self.device_index = list(range(self.num_gpus))
        else:
            self.device_index = [0]

        self.gpu_handler = GPUService(device=self.device, device_index=self.device_index)

    @abstractmethod
    async def process_input(self) -> None:
        """Process the input request by creating a task and adding it to the appropriate queues."""
        raise NotImplementedError("This method should be implemented in subclasses.")  # noqa: EM101


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(  # noqa: PLR0913
        self,
        whisper_model: str,
        compute_type: str,
        window_lengths: list[float],
        shift_lengths: list[float],
        multiscale_weights: list[float],
        extra_languages: list[str] | None,
        extra_languages_model_paths: list[str] | None,
        transcribe_server_urls: list[str] | None,
        diarize_server_urls: list[str] | None,
        debug_mode: bool,
    ) -> None:
        """
        Initialize the ASRAsyncService class.

        Args:
            whisper_model (str):
                The path to the whisper model.
            compute_type (str):
                The compute type to use for inference.
            window_lengths (List[float]):
                The window lengths to use for diarization.
            shift_lengths (List[float]):
                The shift lengths to use for diarization.
            multiscale_weights (List[float]):
                The multiscale weights to use for diarization.
            extra_languages (Union[List[str], None]):
                The list of extra languages to support.
            extra_languages_model_paths (Union[List[str], None]):
                The list of paths to the extra language models.
            use_remote_servers (bool):
                Whether to use remote servers for transcription and diarization.
            transcribe_server_urls (Union[List[str], None]):
                The list of URLs to the remote transcription servers.
            diarize_server_urls (Union[List[str], None]):
                The list of URLs to the remote diarization servers.
            debug_mode (bool):
                Whether to run in debug mode.
        """
        super().__init__()

        self.whisper_model: str = whisper_model
        self.compute_type: str = compute_type
        self.window_lengths: list[float] = window_lengths
        self.shift_lengths: list[float] = shift_lengths
        self.multiscale_weights: list[float] = multiscale_weights
        self.extra_languages: list[str] | None = extra_languages
        self.extra_languages_model_paths: list[str] | None = extra_languages_model_paths

        self.local_services: LocalServiceRegistry = LocalServiceRegistry()
        self.remote_services: RemoteServiceRegistry = RemoteServiceRegistry()
        self.dual_channel_transcribe_options: dict = {
            "beam_size": 1,
            "patience": 1,
            "length_penalty": 1,
            "suppress_blank": False,
            "word_timestamps": True,
            "temperature": 0.0,
        }

        if transcribe_server_urls is not None:
            logger.info("You provided URLs for remote transcription server, no local model will be used.")
            self.remote_services.transcription = RemoteServiceConfig(
                use_remote=True,
                url_handler=URLService(remote_urls=transcribe_server_urls),
            )
        else:
            logger.info("You did not provide URLs for remote transcription server, local model will be used.")
            self.create_transcription_local_service()

        if diarize_server_urls is not None:
            logger.info("You provided URLs for remote diarization server, no local model will be used.")
            self.remote_services.diarization = RemoteServiceConfig(
                use_remote=True,
                url_handler=URLService(remote_urls=diarize_server_urls),
            )
        else:
            logger.info("You did not provide URLs for remote diarization server, local model will be used.")
            self.create_diarization_local_service()

        self.debug_mode = debug_mode

    def create_transcription_local_service(self) -> None:
        """Create a local transcription service."""
        self.local_services.transcription = TranscribeService(
            model_path=self.whisper_model,
            model_engine=settings.whisper_engine,
            compute_type=self.compute_type,
            device=self.device,
            device_index=self.device_index,
            extra_languages=self.extra_languages,
            extra_languages_model_paths=self.extra_languages_model_paths,
        )

    def create_diarization_local_service(self) -> None:
        """Create a local diarization service."""
        if settings.diarization_backend == "longform-diarizer":
            logger.info("Using LongFormDiarizeService for diarization.")
            self.local_services.diarization = LongFormDiarizeService(
                device=self.device,
            )
        else:
            logger.info("Using DiarizeService for diarization.")
            self.local_services.diarization = DiarizeService(
                device=self.device,
                device_index=self.device_index,
                window_lengths=self.window_lengths,
                shift_lengths=self.shift_lengths,
                multiscale_weights=self.multiscale_weights,
            )

    def create_local_service(self, task: Literal["transcription", "diarization"]) -> None:
        """Create a local service."""
        if task == "transcription":
            self.create_transcription_local_service()
        elif task == "diarization":
            self.create_diarization_local_service()
        else:
            raise NotImplementedError("No task specified.")  # noqa: EM101

    async def inference_warmup(self) -> None:
        """Warmup the GPU by loading the models."""
        sample_path = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(
                filepath=str(sample_path),
                batch_size=1,
                offset_start=None,
                offset_end=None,
                num_speakers=1,
                diarization=True,
                multi_channel=False,
                source_lang="en",
                num_beams=1,
                timestamps_format="s",
                vocab=None,
                word_timestamps=False,
                internal_vad=False,
                repetition_penalty=1.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )

    async def process_input(  # noqa: PLR0913 PLR0912
        self,
        filepath: str | list[str],
        batch_size: int | None,
        offset_start: float | None,
        offset_end: float | None,
        num_speakers: int,
        diarization: bool,
        multi_channel: bool,
        source_lang: str,
        num_beams: int,
        timestamps_format: str,
        vocab: list[str] | None,
        word_timestamps: bool,
        internal_vad: bool,
        repetition_penalty: float,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        condition_on_previous_text: bool,
        url: str | None = None,
        url_type: str | None = None,
    ) -> tuple[list[dict], ProcessTimes, float] | Exception:
        """Process the input request and return the results.

        This method will create a task and add it to the appropriate queues.
        All tasks are added to the transcription queue, but will be added to the
        diarization queues only if the user requested it.
        Each step will be processed asynchronously and the results will be returned
        and stored in separated keys in the task dictionary.

        Args:
            filepath (Union[str, List[str]]):
                Path to the audio file or list of paths to the audio files to process.
            batch_size (Union[int, None]):
                The batch size to use for the transcription. For faster-whisper-batch engines only.
            offset_start (Union[float, None]):
                The start time of the audio file to process.
            offset_end (Union[float, None]):
                The end time of the audio file to process.
            num_speakers (int):
                The number of oracle speakers.
            diarization (bool):
                Whether to do diarization or not.
            multi_channel (bool):
                Whether to do multi-channel diarization or not.
            source_lang (str):
                Source language of the audio file.
            num_beams (int):
                The number of beams to use for the beam search.
            timestamps_format (str):
                Timestamps format to use.
            vocab (Union[List[str], None]):
                List of words to use for the vocabulary.
            word_timestamps (bool):
                Whether to return word timestamps or not.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                The repetition penalty to use for the beam search.
            compression_ratio_threshold (float):
                If the gzip compression ratio is above this value, treat as failed.
            log_prob_threshold (float):
                If the average log probability over sampled tokens is below this value, treat as failed.
            no_speech_threshold (float):
                If the no_speech probability is higher than this value AND the average log probability
                over sampled tokens is below `log_prob_threshold`, consider the segment as silent.
            condition_on_previous_text (bool):
                If True, the previous output of the model is provided as a prompt for the next window;
                disabling may make the text inconsistent across windows, but the model becomes less prone
                to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        Returns:
            Union[Tuple[List[dict], ProcessTimes, float], Exception]:
                The results of the ASR pipeline or an exception if something went wrong.
                Results are returned as a tuple of the following:
                    * List[dict]: The final results of the ASR pipeline.
                    * ProcessTimes: The process times of each step of the ASR pipeline.
                    * float: The audio duration
        """
        if isinstance(filepath, list):
            audio, durations = [], []
            for path in filepath:
                _audio, _duration = read_audio(path, offset_start=offset_start, offset_end=offset_end)

                audio.append(_audio)
                durations.append(_duration)

            duration = sum(durations) / len(durations)

        else:
            audio, duration = read_audio(filepath, offset_start=offset_start, offset_end=offset_end)

        gpu_index = None
        if self.remote_services.transcription.use_remote is True:
            _url = await self.remote_services.transcription.next_url()
            transcription_execution = RemoteExecution(url=_url)
        else:
            gpu_index = await self.gpu_handler.get_device()
            transcription_execution = LocalExecution(index=gpu_index)

        if diarization and multi_channel is False:
            if self.remote_services.diarization.use_remote is True:
                _url = await self.remote_services.diarization.next_url()
                diarization_execution = RemoteExecution(url=_url)
            else:
                if gpu_index is None:
                    gpu_index = await self.gpu_handler.get_device()

                diarization_execution = LocalExecution(index=gpu_index)
        else:
            diarization_execution = None

        task = ASRTask(
            audio=audio,
            url=url,
            url_type=url_type,
            diarization=DiarizationTask(execution=diarization_execution, num_speakers=num_speakers),
            duration=duration,
            batch_size=batch_size,
            multi_channel=multi_channel,
            offset_start=offset_start,
            post_processing=PostProcessingTask(),
            process_times=ProcessTimes(),
            timestamps_format=timestamps_format,
            transcription=TranscriptionTask(
                execution=transcription_execution,
                options=TranscriptionOptions(
                    compression_ratio_threshold=compression_ratio_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    internal_vad=internal_vad,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    repetition_penalty=repetition_penalty,
                    source_lang=source_lang,
                    num_beams=num_beams,
                    vocab=vocab,
                ),
            ),
            word_timestamps=word_timestamps,
        )

        try:
            start_process_time = time.time()

            transcription_task = self.process_transcription(task, self.debug_mode)
            diarization_task = self.process_diarization(task, self.debug_mode)

            await asyncio.gather(transcription_task, diarization_task)

            if isinstance(task.diarization.result, ProcessException):
                return task.diarization.result

            if diarization and task.diarization.result is None and multi_channel is False:
                # Empty audio early return
                return early_return(duration=duration)

            if isinstance(task.transcription.result, ProcessException):
                return task.transcription.result

            await asyncio.get_event_loop().run_in_executor(
                None,
                self.process_post_processing,
                task,
            )

            if isinstance(task.post_processing.result, ProcessException):
                return task.post_processing.result

            task.process_times.total = time.time() - start_process_time

        except Exception as e:
            return e

        else:
            return task.post_processing.result, task.process_times, duration

        finally:
            del task

            if gpu_index is not None:
                self.gpu_handler.release_device(gpu_index)

    async def process_transcription(self, task: ASRTask, debug_mode: bool) -> None:
        """
        Process a task of transcription and update the task with the result.

        Args:
            task (ASRTask): The task and its parameters.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            if isinstance(task.transcription.execution, LocalExecution):
                out = await time_and_tell_async(
                    lambda: self.local_services.transcription(
                        audio=task.audio,
                        model_index=task.transcription.execution.index,
                        source_lang=task.transcription.options.source_lang,
                        batch_size=task.batch_size,
                        num_beams=task.transcription.options.num_beams,
                        suppress_blank=False,  # TODO: Add this to the options
                        vocab=task.transcription.options.vocab,
                        word_timestamps=task.word_timestamps,
                        internal_vad=task.transcription.options.internal_vad,
                        repetition_penalty=task.transcription.options.repetition_penalty,
                        compression_ratio_threshold=task.transcription.options.compression_ratio_threshold,
                        log_prob_threshold=task.transcription.options.log_prob_threshold,
                        no_speech_threshold=task.transcription.options.no_speech_threshold,
                        condition_on_previous_text=task.transcription.options.condition_on_previous_text,
                    ),
                    func_name="transcription",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            elif isinstance(task.transcription.execution, RemoteExecution):
                data = TranscribeRequest(
                    audio=task.audio,
                    **task.transcription.options.model_dump(),
                )
                out = await time_and_tell_async(
                    self.remote_transcription(
                        url=task.transcription.execution.url,
                        data=data,
                    ),
                    func_name="transcription",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            else:
                raise NotImplementedError("No execution method specified.")  # noqa: TRY301 EM101

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.transcription,
                message=f"Error in transcription: {e}\n{traceback.format_exc()}",
            )
            process_time = None

        finally:
            task.process_times.transcription = process_time
            task.transcription.result = result

    async def process_diarization(self, task: ASRTask, debug_mode: bool) -> None:
        """
        Process a task of diarization.

        Args:
            task (ASRTask): The task and its parameters.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            if isinstance(task.diarization.execution, LocalExecution):
                if settings.diarization_backend == "longform-diarizer":
                    out = await time_and_tell_async(
                        lambda: self.local_services.diarization(
                            waveform=task.audio,
                            oracle_num_speakers=task.diarization.num_speakers,
                        ),
                        func_name="diarization",
                        debug_mode=debug_mode,
                    )
                    result, process_time = out
                else:
                    out = await time_and_tell_async(
                        lambda: self.local_services.diarization(
                            waveform=task.audio,
                            audio_duration=task.duration,
                            oracle_num_speakers=task.diarization.num_speakers,
                            model_index=task.diarization.execution.index,
                            vad_service=self.local_services.vad,
                        ),
                        func_name="diarization",
                        debug_mode=debug_mode,
                    )
                    result, process_time = out

            elif isinstance(task.diarization.execution, RemoteExecution):
                if task.url:
                    audio = task.url
                    audio_type = task.url_type
                else:
                    audio = task.audio
                    audio_type = "tensor"

                data = DiarizationRequest(
                    audio=audio,
                    audio_type=audio_type,
                    duration=task.duration,
                    num_speakers=task.diarization.num_speakers,
                )
                out = await time_and_tell_async(
                    self.remote_diarization(
                        url=task.diarization.execution.url,
                        data=data,
                    ),
                    func_name="diarization",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            elif task.diarization.execution is None:
                result = None
                process_time = None

            else:
                raise NotImplementedError("No execution method specified.")  # noqa: TRY301 EM101

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.diarization,
                message=f"Error in diarization: {e}\n{traceback.format_exc()}",
            )
            process_time = None

        finally:
            task.process_times.diarization = process_time
            task.diarization.result = result

    def process_post_processing(self, task: ASRTask) -> None:
        """
        Process a task of post-processing.

        Args:
            task (ASRTask): The task and its parameters.

        Returns:
            None: The task is updated with the result.
        """
        try:
            total_post_process_time = 0

            if task.multi_channel:
                utterances, process_time = time_and_tell(
                    self.local_services.post_processing.multi_channel_speaker_mapping(task.transcription.result),
                    func_name="multi_channel_speaker_mapping",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

            else:
                formatted_segments, process_time = time_and_tell(
                    format_segments(
                        transcription_output=task.transcription.result,
                    ),
                    func_name="format_segments",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

                if task.diarization.execution is not None:
                    utterances, process_time = time_and_tell(
                        self.local_services.post_processing.single_channel_speaker_mapping(
                            transcript_segments=formatted_segments,
                            speaker_timestamps=task.diarization.result,
                            word_timestamps=task.word_timestamps,
                        ),
                        func_name="single_channel_speaker_mapping",
                        debug_mode=self.debug_mode,
                    )
                    total_post_process_time += process_time
                else:
                    utterances = formatted_segments

            if settings.enable_punctuation_based_alignment:
                utterances, process_time = time_and_tell(
                    self.local_services.post_processing.punctuation_based_alignment(
                        utterances=utterances,
                        speaker_timestamps=task.diarization.result,
                    ),
                    func_name="punctuation_based_alignment",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

            final_utterances, process_time = time_and_tell(
                self.local_services.post_processing.final_processing_before_returning(
                    utterances=utterances,
                    offset_start=task.offset_start,
                    timestamps_format=task.timestamps_format,
                    word_timestamps=task.word_timestamps,
                ),
                func_name="final_processing_before_returning",
                debug_mode=self.debug_mode,
            )
            total_post_process_time += process_time

        except Exception as e:
            final_utterances = ProcessException(
                source=ExceptionSource.post_processing,
                message=f"Error in post-processing: {e}\n{traceback.format_exc()}",
            )
            total_post_process_time = None

        finally:
            task.process_times.post_processing = total_post_process_time
            task.post_processing.result = final_utterances

    async def remote_transcription(
        self,
        url: str,
        data: TranscribeRequest,
    ) -> TranscriptionOutput:
        """Remote transcription method."""
        headers = {"Content-Type": "application/json"}

        if not settings.debug:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            auth_url = f"{url}/api/v1/auth"
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url=auth_url,
                    data={"username": settings.username, "password": settings.password},
                    headers=headers,
                ) as response,
            ):
                if not response.ok:
                    raise Exception(response.status)  # noqa: TRY002

                token = await response.json()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token['access_token']}",
                }

        transcription_timeout = aiohttp.ClientTimeout(total=1200)
        async with (
            AsyncLocationTrustedRedirectSession(timeout=transcription_timeout) as session,
            session.post(
                url=f"{url}/api/v1/transcribe",
                data=data.model_dump_json(),
                headers=headers,
                location_trusted=True,
            ) as response,
        ):
            if not response.ok:
                r = await response.json()
                raise Exception(r["detail"])  # noqa: TRY002

            return TranscriptionOutput(**await response.json())

    async def remote_diarization(
        self,
        url: str,
        data: DiarizationRequest,
    ) -> DiarizationOutput:
        """Remote diarization method."""
        headers = {"Content-Type": "application/json"}

        if not settings.debug:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            auth_url = f"{url}/api/v1/auth"
            diarization_timeout = aiohttp.ClientTimeout(total=10)
            async with (
                AsyncLocationTrustedRedirectSession(timeout=diarization_timeout) as session,
                session.post(
                    url=auth_url,
                    data={"username": settings.username, "password": settings.password},
                    headers=headers,
                    location_trusted=True,
                ) as response,
            ):
                if not response.ok:
                    raise Exception(response.status)  # noqa: TRY002

                token = await response.json()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token['access_token']}",
                }
        diarization_timeout = aiohttp.ClientTimeout(total=1200)
        async with (
            AsyncLocationTrustedRedirectSession(timeout=diarization_timeout) as session,
            session.post(
                url=f"{url}/api/v1/diarize",
                data=data.model_dump_json(),
                headers=headers,
                location_trusted=True,
            ) as response,
        ):
            if not response.ok:
                r = await response.json()
                raise Exception(r["detail"])  # noqa: TRY002

            return DiarizationOutput(**await response.json())

    async def get_url(self, task: Literal["transcription", "diarization"]) -> list[str] | ProcessException:
        """Get the list of remote URLs."""
        logger.info(self.remote_services.transcription)
        logger.info(self.remote_services.diarization)
        try:
            selected_task = getattr(self.remote_services, task)
            logger.info(selected_task)
            # Case 1: We are not using remote task
            if selected_task.use_remote is False:
                return ProcessException(
                    source=ExceptionSource.get_url,
                    message=f"You are not using remote {task}.",
                )
            # Case 2: We are using remote task
            else:
                return selected_task.get_urls()

        except Exception as e:
            return ProcessException(
                source=ExceptionSource.get_url,
                message=f"Error in getting URL: {e}\n{traceback.format_exc()}",
            )

    async def add_url(self, data: UrlSchema) -> UrlSchema | ProcessException:
        """Add a remote URL to the list of URLs."""
        try:
            selected_task = getattr(self.remote_services, data.task)
            # Case 1: We are not using remote task yet
            if selected_task.use_remote is False:
                setattr(
                    self.remote_services,
                    data.task,
                    RemoteServiceConfig(
                        use_remote=True,
                        url_handler=URLService(remote_urls=[str(data.url)]),
                    ),
                )
                setattr(self.local_services, data.task, None)
            # Case 2: We are already using remote task
            else:
                await selected_task.add_url(str(data.url))

        except Exception as e:
            return ProcessException(
                source=ExceptionSource.add_url,
                message=f"Error in adding URL: {e}\n{traceback.format_exc()}",
            )

        return data

    async def remove_url(self, data: UrlSchema) -> UrlSchema | ProcessException:
        """Remove a remote URL from the list of URLs."""
        try:
            selected_task = getattr(self.remote_services, data.task)
            # Case 1: We are not using remote task
            if selected_task.use_remote is False:
                raise ValueError(f"You are not using remote {data.task}.")  # noqa: TRY003 TRY301 EM102
            # Case 2: We are using remote task
            await selected_task.remove_url(str(data.url))
            if selected_task.get_queue_size() == 0:
                # No more remote URLs, switch to local service
                self.create_local_service(task=data.task)
                setattr(self.remote_services, data.task, RemoteServiceConfig())

        except Exception as e:
            return ProcessException(
                source=ExceptionSource.remove_url,
                message=f"Error in removing URL: {e}\n{traceback.format_exc()}",
            )

        else:
            return data


class ASRLiveService(ASRService):
    """ASR Service module for live endpoints."""

    def __init__(self, whisper_model: str, compute_type: str, debug_mode: bool) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()

        self.transcription_service = TranscribeService(
            model_path=whisper_model,
            model_engine=settings.whisper_engine,
            compute_type=compute_type,
            device=self.device,
            device_index=self.device_index,
        )
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by loading the models."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"
        with Path.open(sample_audio, "rb") as audio_file:
            async for _ in self.process_input(
                data=audio_file.read(),
                source_lang="en",
            ):
                pass

    async def process_input(self, data: bytes, source_lang: str) -> AsyncGenerator[dict]:
        """
        Process the input data and return the results as a tuple of text and duration.

        Args:
            data (bytes):
                The raw audio bytes to process.
            source_lang (str):
                The source language of the audio data.

        Yields:
            Iterable[dict]: The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            waveform, _ = read_audio(data)

            async for result in self.transcription_service.async_live_transcribe(
                audio=waveform,
                source_lang=source_lang,
                model_index=gpu_index,
            ):
                yield result

        except Exception as e:
            logger.error(f"Error in transcription gpu {gpu_index}: {e}\n{traceback.format_exc()}")

        finally:
            self.gpu_handler.release_device(gpu_index)


class ASRTranscriptionOnly(ASRService):
    """ASR Service module for transcription-only endpoint."""

    def __init__(
        self,
        whisper_model: str,
        compute_type: str,
        extra_languages: list[str] | None,
        extra_languages_model_paths: list[str] | None,
        debug_mode: bool,
    ) -> None:
        """Initialize the ASRTranscriptionOnly class."""
        super().__init__()

        self.transcription_service = TranscribeService(
            model_path=whisper_model,
            model_engine=settings.whisper_engine,
            compute_type=compute_type,
            device=self.device,
            device_index=self.device_index,
            extra_languages=extra_languages,
            extra_languages_model_paths=extra_languages_model_paths,
        )
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by doing one inference."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        audio, _ = read_audio(str(sample_audio))
        ts = audio.numpy()

        data = TranscribeRequest(
            audio=ts,
            source_lang="en",
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            internal_vad=False,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.0,
            vocab=None,
        )

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(data=data)

    async def process_input(self, data: TranscribeRequest) -> TranscriptionOutput | list[TranscriptionOutput]:
        """
        Process the input data and return the results as a list of segments.

        Args:
            data (TranscribeRequest):
                The input data to process.

        Returns:
            Union[TranscriptionOutput, List[TranscriptionOutput]]:
                The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            result = self.transcription_service(
                audio=data.audio,
                batch_size=data.batch_size,
                source_lang=data.source_lang,
                num_beams=data.num_beams,
                model_index=gpu_index,
                suppress_blank=False,
                word_timestamps=True,
                compression_ratio_threshold=data.compression_ratio_threshold,
                condition_on_previous_text=data.condition_on_previous_text,
                internal_vad=data.internal_vad,
                log_prob_threshold=data.log_prob_threshold,
                repetition_penalty=data.repetition_penalty,
                no_speech_threshold=data.no_speech_threshold,
                vocab=data.vocab,
            )

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.transcription,
                message=f"Error in transcription: {e}\n{traceback.format_exc()}",
            )

        finally:
            self.gpu_handler.release_device(gpu_index)

        return result


class ASRDiarizationOnly(ASRService):
    """ASR Service module for diarization-only endpoint."""

    def __init__(
        self,
        window_lengths: list[int],
        shift_lengths: list[int],
        multiscale_weights: list[float],
        debug_mode: bool,
    ) -> None:
        """Initialize the ASRDiarizationOnly class."""
        super().__init__()

        if settings.diarization_backend == "longform-diarizer":
            self.diarization_service = LongFormDiarizeService(
                device=self.device,
            )
        else:
            self.diarization_service = DiarizeService(
                device=self.device,
                device_index=self.device_index,
                window_lengths=window_lengths,
                shift_lengths=shift_lengths,
                multiscale_weights=multiscale_weights,
            )
        self.vad_service = VadService()
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by doing one inference."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        audio, duration = read_audio(str(sample_audio))
        ts = audio.numpy()

        data = DiarizationRequest(
            audio=ts,
            audio_type="tensor",
            duration=duration,
            num_speakers=1,
        )

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(data=data)

    async def process_input(self, data: DiarizationRequest) -> DiarizationOutput:
        """
        Process the input data and return the results as a list of segments.

        Args:
            data (DiarizationRequest):
                The input data to process.

        Returns:
            DiarizationOutput:
                The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            if data.audio_type == "tensor":
                result = self.diarization_service(
                    waveform=data.audio,
                    audio_duration=data.duration,
                    oracle_num_speakers=data.num_speakers,
                    model_index=gpu_index,
                    vad_service=self.vad_service,
                )
            elif data.audio_type and data.audio_type == "url":
                result = self.diarization_service(
                    url=data.audio,
                    url_type=data.audio_type,
                    audio_duration=data.duration,
                    oracle_num_speakers=data.num_speakers,
                    model_index=gpu_index,
                    vad_service=self.vad_service,
                )
            else:
                raise ValueError(f"Invalid audio type: {data.audio_type}. Must be one of ['tensor', 'url'].")  # noqa: TRY003 TRY301 EM102

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.diarization,
                message=f"Error in diarization: {e}\n{traceback.format_exc()}",
            )

        finally:
            self.gpu_handler.release_device(gpu_index)

        return result
