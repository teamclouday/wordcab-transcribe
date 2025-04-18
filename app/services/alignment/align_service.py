"""
Forced Alignment with Whisper
C. Max Bain

Inspired by: https://github.com/m-bain/whisperX
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchaudio
from loguru import logger
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .models import AlignedTranscriptionResult, SingleAlignedSegment, SingleWordSegment

PUNKT_ABBREVIATIONS = ["dr", "vs", "mr", "mrs", "prof"]

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
}


def fill_missing_words(sentence_data: dict, default_probability: float = 0.5) -> dict:
    """
    Fills in missing word lists in sentence data using estimated timings and default probabilities.

    Parameters:
        sentence_data (dict): A dictionary containing the sentence data.
        default_probability (float, optional): The default probability to assign to each word. Defaults to 0.5.

    Returns:
        dict: The updated sentence data with filled-in words.
    """

    def estimate_word_timings(text: str, start: int, end: int) -> list:
        """
        Estimates start and end times for each word in the text based on the total duration and character count.
        """
        words = text.split()
        total_duration = end - start
        total_chars = sum(len(word) for word in words)
        avg_char_duration = total_duration / total_chars

        word_timings = []
        current_start = start
        for word in words:
            word_duration = len(word) * avg_char_duration
            word_end = current_start + word_duration
            word_timings.append({"start": current_start, "end": word_end, "word": word})
            current_start = word_end

        # Adjust the start of the first word and the end of the last word
        if word_timings:
            word_timings[0]["start"] = start
            word_timings[-1]["end"] = end

        return word_timings

    # Check if word list is missing or empty
    if "words" not in sentence_data or not sentence_data["words"]:
        sentence_text = sentence_data["text"]
        sentence_start = sentence_data["start"]
        sentence_end = sentence_data["end"]

        # Estimate word timings
        estimated_words = estimate_word_timings(sentence_text, sentence_start, sentence_end)

        # Assign default probability to each word
        for word_info in estimated_words:
            word_info["probability"] = default_probability

        sentence_data["words"] = estimated_words

    return sentence_data


def estimate_none_timestamps(timestamp_list: list) -> list:
    """
    Estimates missing timestamps in a list of timestamp segments based on the character length of segment times.

    Parameters:
    timestamp_list (list): A list of timestamp segments with text.

    Returns:
    list: The list with estimated missing timestamps.
    """
    total_duration = 0
    total_characters = 0

    for segment in timestamp_list:
        start, end = segment["timestamp"]
        if start is not None and end is not None:
            duration = end - start
            characters = len(segment["text"])
            total_duration += duration
            total_characters += characters

    avg_duration_per_char = (
        total_duration / total_characters if total_characters > 0 else 0.1
    )  # Default duration per character (assumed)

    for i, segment in enumerate(timestamp_list):
        start, end = segment["timestamp"]
        characters = len(segment["text"])
        estimated_duration = characters * avg_duration_per_char

        if start is None:
            start = (
                timestamp_list[i - 1]["timestamp"][1]
                if i > 0 and timestamp_list[i - 1]["timestamp"][1] is not None
                else 0
            )
            segment["timestamp"] = (start, start + estimated_duration)
        if end is None:
            segment["timestamp"] = (start, start + estimated_duration)
    return timestamp_list


def load_align_model(
    language_code: str,
    device: str,
    model_name: str | None = None,
    model_dir: str | None = None,
) -> tuple[Wav2Vec2ForCTC, dict]:
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(
                "There is no default alignment model set for this language"
                f" ({language_code}).                Please find a wav2vec2.0 model"
                " finetuned on this language in https://huggingface.co/models, then"
                " pass the model name in --align_model [MODEL_NAME]",
            )
            raise ValueError(f"No default align-model for language: {language_code}")  # noqa: TRY003 EM102

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "Error loading model from huggingface, check"
                " https://huggingface.co/models for finetuned wav2vec2.0 models",
            )
            raise ValueError from e
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}

    align_metadata = {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type,
    }

    return align_model, align_metadata


def align(  # noqa: PLR0913 PLR0912 PLR0915
    transcript: Iterable,
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: np.ndarray | torch.Tensor,
    device: str,
    sample_rate: int = 16000,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """

    transcript = estimate_none_timestamps(transcript)

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    max_duration = audio.shape[1] / sample_rate

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            logger.debug(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        per_word = text.split(" ") if model_lang not in LANGUAGES_WITHOUT_SPACES else text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading or (cdx > len(text) - num_trailing - 1):
                pass
            elif char_ in model_dictionary:
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any(c in model_dictionary for c in wrd):
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans

    aligned_segments: list[SingleAlignedSegment] = []

    # 2. Get prediction matrix from alignment model & align

    for _, segment in enumerate(transcript):
        t1 = segment["timestamp"][0]
        t2 = segment["timestamp"][1]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment["clean_char"]) == 0:
            logger.warning(
                f'Failed to align segment ("{segment["text"]}"): no characters in this'
                " segment found in model dictionary, resorting to original...",
            )
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= max_duration:
            logger.warning(
                f'Failed to align segment ("{segment["text"]}"): original start time'
                " longer than audio duration, skipping...",
            )
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]

        f1 = int(t1 * sample_rate)
        f2 = int(t2 * sample_rate)

        # NOTE: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:  # noqa: PLR2004
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(waveform_segment, (0, 400 - waveform_segment.shape[-1]))
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")  # noqa: EM102
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char in ["[pad]", "<pad>"]:
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                },
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES or (len(text) - 1 or text[cdx + 1] == " "):
                word_idx += 1

        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for subsdx, (sstart, send) in enumerate(segment["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[
                (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send),
                "sentence-idx",
            ] = subsdx

            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            sentence_end = curr_chars["end"].max()
            sentence_words = []
            avg_char_duration = None
            last_end = None

            for ix, word_idx in enumerate(curr_chars["word-idx"].unique()):
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # Don't use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                if "start" not in word_segment or "end" not in word_segment:
                    if avg_char_duration is None:
                        df = pd.DataFrame(sentence_words)  # noqa: PD901
                        df = df.dropna(subset=["start", "end"])  # Drop rows where 'start' or 'end' is NaN # noqa: PD901
                        if not df.empty:
                            df["duration"] = df["end"] - df["start"]
                            df["char_length"] = df["word"].apply(len)
                            avg_char_duration = (df["duration"] / df["char_length"]).mean()
                        else:
                            avg_char_duration = 0.1  # Default average character duration

                    word_len = len(word_segment["word"])
                    estimated_duration = word_len * avg_char_duration

                    if "start" not in word_segment:
                        if len(sentence_words) == 0:
                            word_segment["start"] = sentence_start
                        else:
                            prev_end = sentence_words[len(sentence_words) - 1]["end"]
                            word_segment["start"] = prev_end

                    if "end" not in word_segment:
                        estimated_end = word_segment["start"] + estimated_duration
                        if ix == len(sentence_words) - 1:
                            word_segment["end"] = sentence_end
                        else:
                            word_segment["end"] = estimated_end

                    if "score" not in word_segment:
                        word_segment["score"] = 0.5

                if last_end is not None and word_segment["start"] < last_end:
                    word_segment["start"] = last_end
                last_end = word_segment["end"]

                sentence_words.append(word_segment)

            aligned_subsegments.append(
                {
                    "text": sentence_text,
                    "start": sentence_start,
                    "end": sentence_end,
                    "words": sentence_words,
                },
            )

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars = curr_chars.fillna(-1)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict("records")
        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: list[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}


def interpolate_nans(x: pd.DataFrame, method: str = "nearest") -> pd.DataFrame:
    if x.notna().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()


"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(emission: torch.Tensor, tokens: list, blank_id: int = 0) -> torch.Tensor:
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis: torch.Tensor, emission: torch.Tensor, tokens: list, blank_id: int = 0) -> list:
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self) -> str:
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self) -> int:
        return self.end - self.start


def merge_repeats(path: list, transcript: list) -> list:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            ),
        )
        i1 = i2
    return segments


def merge_words(segments: list, separator: str = "|") -> list:
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
