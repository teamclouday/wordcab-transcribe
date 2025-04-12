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
"""Transcribe Service for audio files."""

import dataclasses
from collections.abc import AsyncGenerator, Iterable
from typing import NamedTuple

import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from loguru import logger

from app.models import (
    MultiChannelSegment,
    MultiChannelTranscriptionOutput,
    TranscriptionOutput,
    Word,
)


class FasterWhisperModel(NamedTuple):
    """Faster Whisper Model."""

    model: WhisperModel
    lang: str


class TranscribeService:
    """Transcribe Service for audio files."""

    def __init__(  # noqa: PLR0913
        self,
        model_path: str,
        model_engine: str,
        compute_type: str,
        device: str,
        device_index: int | list[int],
        extra_languages: list[str] | None = None,
        extra_languages_model_paths: list[str] | None = None,
    ) -> None:
        """Initialize the Transcribe Service.

        This service uses the WhisperModel from faster-whisper to transcribe audio files.

        Args:
            model_path (str):
                Path to the model checkpoint. This can be a local path or a URL.
            compute_type (str):
                Compute type to use for inference. Can be "int8", "int8_float16", "int16" or "float_16".
            device (str):
                Device to use for inference. Can be "cpu" or "cuda".
            device_index (Union[int, List[int]]):
                Index of the device to use for inference.
            extra_languages (Union[List[str], None]):
                List of extra languages to transcribe. Defaults to None.
            extra_languages_model_paths (Union[List[str], None]):
                List of paths to the extra language models. Defaults to None.
        """
        self.device = device
        self.compute_type = compute_type
        self.model_path = model_path
        self.model_engine = model_engine

        if self.model_engine == "faster-whisper":
            logger.info("Using faster-whisper model engine.")
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                device_index=device_index,
                compute_type=self.compute_type,
            )
        elif self.model_engine == "faster-whisper-batched":
            logger.info("Using faster-whisper-batched model engine.")
            self.model = BatchedInferencePipeline(
                model=WhisperModel(
                    self.model_path,
                    device=self.device,
                    device_index=device_index,
                    compute_type=self.compute_type,
                ),
            )
        else:
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                device_index=device_index,
                compute_type=self.compute_type,
            )

        self.extra_lang = extra_languages
        self.extra_lang_models = extra_languages_model_paths

    def __call__(  # noqa: PLR0913
        self,
        audio: str | torch.Tensor | np.ndarray | list[str] | list[torch.Tensor] | list[np.ndarray],
        source_lang: str,
        batch_size: int,
        num_beams: int = 1,
        suppress_blank: bool = False,
        vocab: list[str] | None = None,
        word_timestamps: bool = True,
        internal_vad: bool = False,
        repetition_penalty: float = 1.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
    ) -> TranscriptionOutput | list[TranscriptionOutput]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, torch.Tensor, TensorShare, List[str], List[torch.Tensor], List[TensorShare]]):
                Audio file path or audio tensor. If a tuple is passed, the task is assumed
                to be a multi_channel task and the list of audio files or tensors is passed.
            source_lang (str):
                Language of the audio file.
            model_index (int):
                Index of the model to use.
            batch_size (int):
                Batch size to use during generation. Only used for tensorrt_llm engine.
            num_beams (int):
                Number of beams to use during generation.
            suppress_blank (bool):
                Whether to suppress blank at the beginning of the sampling.
            vocab (Union[List[str], None]):
                Vocabulary to use during generation if not None. Defaults to None.
            word_timestamps (bool):
                Whether to return word timestamps.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                Repetition penalty to use during generation beamed search.
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
            Union[TranscriptionOutput, List[TranscriptionOutput]]:
                Transcription output. If the task is a multi_channel task, a list of TranscriptionOutput is returned.
        """

        if vocab is not None and isinstance(vocab, list) and len(vocab) > 0 and vocab[0].strip():
            words = ", ".join(vocab)
            prompt = f"Vocab: {words.strip()}."
        else:
            prompt = None

        if not isinstance(audio, list):
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()

            if self.model_engine == "faster-whisper":
                segments, _ = self.model.transcribe(
                    audio,
                    language=source_lang,
                    initial_prompt=prompt,
                    beam_size=num_beams,
                    repetition_penalty=repetition_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    suppress_blank=suppress_blank,
                    word_timestamps=word_timestamps,
                    vad_filter=internal_vad,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 100,
                        "speech_pad_ms": 30,
                    },
                )
            elif self.model_engine == "faster-whisper-batched":
                logger.debug("Batch size: ", batch_size)
                segments, _ = self.model.transcribe(
                    audio,
                    language=source_lang,
                    hotwords=prompt,
                    beam_size=num_beams,
                    repetition_penalty=repetition_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    suppress_blank=suppress_blank,
                    word_timestamps=word_timestamps,
                    batch_size=batch_size,
                )
            _outputs = [segment._asdict() for segment in segments]
            outputs = TranscriptionOutput(segments=_outputs)
        else:
            outputs = self.multi_channel(
                audio,
                source_lang=source_lang,
                num_beams=num_beams,
                suppress_blank=suppress_blank,
                word_timestamps=word_timestamps,
                internal_vad=internal_vad,
                repetition_penalty=repetition_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                prompt=prompt,
            )
        return outputs

    async def async_live_transcribe(
        self,
        audio: torch.Tensor,
        source_lang: str,
        initial_prompt: str,
        model_index: int,
    ) -> AsyncGenerator[dict]:
        """Async generator for live transcriptions.

        This method wraps the live_transcribe method to make it async.

        Args:
            audio (torch.Tensor): Audio tensor.
            source_lang (str): Language of the audio file.
            model_index (int): Index of the model to use.

        Yields:
            Iterable[dict]: Iterable of transcribed segments.
        """
        for result in self.live_transcribe(audio, source_lang, initial_prompt, model_index):
            yield result

    def live_transcribe(
        self,
        audio: torch.Tensor,
        source_lang: str,
        initial_prompt: str,
        _model_index: int,
    ) -> Iterable[dict]:
        """
        Transcribe audio from a WebSocket connection.

        Args:
            audio (torch.Tensor): Audio tensor.
            source_lang (str): Language of the audio file.
            model_index (int): Index of the model to use.

        Yields:
            Iterable[dict]: Iterable of transcribed segments.
        """
        segments, _ = self.model.transcribe(
            audio.numpy(),
            language=source_lang,
            suppress_blank=True,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=initial_prompt,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 30,
            },
        )

        for segment in segments:
            yield dataclasses.asdict(segment)

    def multi_channel(  # noqa: PLR0913
        self,
        audio_list: list[str | torch.Tensor | np.ndarray],
        source_lang: str,
        num_beams: int = 1,
        suppress_blank: bool = False,
        word_timestamps: bool = True,
        internal_vad: bool = True,
        repetition_penalty: float = 1.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        prompt: str | None = None,
    ) -> MultiChannelTranscriptionOutput:
        """
        Transcribe an audio file using the faster-whisper original pipeline.

        Args:
            audio_list (List[Union[str, torch.Tensor, TensorShare]]): List of audio file paths or audio tensors.
            source_lang (str): Language of the audio file.
            num_beams (int): Number of beams to use during generation.
            speaker_id (int): Speaker ID used in the diarization.
            suppress_blank (bool):
                Whether to suppress blank at the beginning of the sampling.
            word_timestamps (bool):
                Whether to return word timestamps.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                Repetition penalty to use during generation beamed search.
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
            prompt (Optional[str]): Initial prompt to use for the generation.

        Returns:
            MultiChannelTranscriptionOutput: Multi-channel transcription segments in a list.
        """
        outputs = []

        if self.model_engine == "faster-whisper":
            for speaker_id, audio in enumerate(audio_list):
                final_segments = []
                if isinstance(audio, torch.Tensor):
                    _audio = audio.numpy()
                elif isinstance(audio, np.ndarray):
                    _audio = audio

                segments, _ = self.model.transcribe(
                    _audio,
                    language=source_lang,
                    initial_prompt=prompt,
                    beam_size=num_beams,
                    repetition_penalty=repetition_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    suppress_blank=suppress_blank,
                    word_timestamps=word_timestamps,
                    vad_filter=internal_vad,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 100,
                        "speech_pad_ms": 30,
                    },
                )

                for segment in segments:
                    _segment = MultiChannelSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=[Word(**word._asdict()) for word in segment.words],
                        speaker=speaker_id,
                    )
                    final_segments.append(_segment)

                outputs.append(MultiChannelTranscriptionOutput(segments=final_segments))

        return outputs
