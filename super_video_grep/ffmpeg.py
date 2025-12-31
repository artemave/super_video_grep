import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def run_ffmpeg(args: List[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def run_ffprobe(args: List[str]) -> dict:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe JSON parse failed: {exc}") from exc


def extract_audio(input_path: str, output_wav: str) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            output_wav,
        ]
    )


def extract_audio_segment(input_path: str, output_wav: str, start: float, end: float) -> None:
    if end <= start:
        raise ValueError("segment end must be after start")
    duration = end - start
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            output_wav,
        ]
    )


def cut_clips(
    input_path: str,
    segments: Iterable[Tuple[float, float]],
    out_dir: str,
    prefix: str,
    counter_start: Optional[int] = None,
    counter_counts: Optional[List[int]] = None,
) -> List[str]:
    clips: List[str] = []
    out_path = Path(out_dir)
    counter_index = counter_start or 0

    for idx, (start, end) in enumerate(segments, start=1):
        clip_path = out_path / f"{prefix}_{idx:03d}.mp4"
        duration = end - start
        if duration <= 0:
            continue
        vf_filter = None
        if counter_start is not None:
            increment = 1
            if counter_counts is not None and idx - 1 < len(counter_counts):
                increment = counter_counts[idx - 1]
            counter_index += increment
            text = f"{counter_index}"
            text = (
                text.replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )
            vf_filter = (
                "drawtext="
                f"text='{text}':x=24:y=24:fontsize=56:"
                "fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6"
            )
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                input_path,
                "-t",
                f"{duration:.3f}",
                *([] if not vf_filter else ["-vf", vf_filter]),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(clip_path),
            ]
        )
        clips.append(str(clip_path))

    return clips


def concat_clips(clips: List[str], output_path: str) -> None:
    if not clips:
        raise ValueError("no clips to concat")

    if len(clips) == 1:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clips[0], output_path)
        return

    list_path = Path(output_path).with_suffix(".concat.txt")
    lines = [f"file '{clip}'" for clip in clips]
    list_path.write_text("\n".join(lines) + "\n")

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            output_path,
        ]
    )
    try:
        list_path.unlink()
    except OSError:
        pass


def extract_subtitles(
    input_path: str,
    output_srt: str,
    stream_index: Optional[int] = None,
    language: Optional[str] = None,
) -> None:
    probe = run_ffprobe(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            input_path,
        ]
    )
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    if not streams:
        raise RuntimeError("no subtitle streams found")

    selected = None
    if stream_index is not None:
        for s in streams:
            if s.get("index") == stream_index:
                selected = s
                break
        if selected is None:
            raise RuntimeError(f"subtitle stream index {stream_index} not found")
    elif language:
        lang = language.lower()
        for s in streams:
            tags = s.get("tags") or {}
            if str(tags.get("language", "")).lower() == lang:
                selected = s
                break
        if selected is None:
            raise RuntimeError(f"no subtitle stream with language '{language}'")
    else:
        selected = streams[0]

    codec = str(selected.get("codec_name") or "").lower()
    if codec not in {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text"}:
        raise RuntimeError(f"subtitle codec '{codec}' is not text-based")

    index = selected.get("index")
    if index is None:
        raise RuntimeError("selected subtitle stream has no index")

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-map",
            f"0:{index}",
            "-c:s",
            "srt",
            output_srt,
        ]
    )
