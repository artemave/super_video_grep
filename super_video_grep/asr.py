import inspect
from typing import Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except Exception:  # pragma: no cover - optional dependency/version feature
    BatchedInferencePipeline = None

from .segments import Segment, Word, normalize_token


def _supports_param(callable_obj, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def build_model(
    model_size: str,
    device: str,
    compute_type: str,
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    kwargs = {}
    if cpu_threads and cpu_threads > 0 and _supports_param(WhisperModel.__init__, "cpu_threads"):
        kwargs["cpu_threads"] = cpu_threads
    return WhisperModel(model_size, device=device, compute_type=compute_type, **kwargs)


def _transcribe_with_model(
    model: WhisperModel,
    audio_path: str,
    options: Dict,
) -> Tuple[List[Segment], List[Word]]:
    segments, _info = model.transcribe(audio_path, **options)
    return _collect_segments(segments)


def _collect_segments(segments) -> Tuple[List[Segment], List[Word]]:
    segs: List[Segment] = []
    words: List[Word] = []
    for segment in segments:
        segs.append(Segment(start=segment.start, end=segment.end, text=segment.text or ""))
        if not segment.words:
            continue
        for w in segment.words:
            text = (w.word or "").strip()
            norm = normalize_token(text)
            if not norm:
                continue
            words.append(Word(start=w.start, end=w.end, text=text, norm=norm))
    return segs, words


def transcribe(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str],
    vad_filter: bool,
    vad_parameters: Optional[Dict],
    batch_size: int,
) -> Tuple[List[Segment], List[Word], List[str]]:
    warnings: List[str] = []
    options: Dict = {
        "word_timestamps": True,
    }
    if language:
        options["language"] = language
    if vad_filter:
        options["vad_filter"] = True
    if vad_parameters:
        options["vad_parameters"] = vad_parameters

    if batch_size and batch_size > 1:
        if BatchedInferencePipeline is None:
            warnings.append("batched inference not available; falling back to non-batched mode")
            segs, words = _transcribe_with_model(model, audio_path, options)
            return segs, words, warnings

        pipeline = BatchedInferencePipeline(model=model)
        try:
            segments, _info = pipeline.transcribe(audio_path, batch_size=batch_size, **options)
            segs, words = _collect_segments(segments)
            return segs, words, warnings
        except TypeError:
            stripped = {k: v for k, v in options.items() if k not in {"vad_filter", "vad_parameters"}}
            if stripped != options:
                try:
                    segments, _info = pipeline.transcribe(audio_path, batch_size=batch_size, **stripped)
                    segs, words = _collect_segments(segments)
                    warnings.append("batched inference ignored VAD options; running without VAD")
                    return segs, words, warnings
                except TypeError:
                    warnings.append("batched inference rejected options; falling back to non-batched mode")
            else:
                warnings.append("batched inference rejected options; falling back to non-batched mode")

    segs, words = _transcribe_with_model(model, audio_path, options)
    return segs, words, warnings
