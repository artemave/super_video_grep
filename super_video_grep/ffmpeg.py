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
    counter_total: Optional[int] = None,
) -> List[str]:
    clips: List[str] = []
    out_path = Path(out_dir)

    for idx, (start, end) in enumerate(segments, start=1):
        clip_path = out_path / f"{prefix}_{idx:03d}.mp4"
        duration = end - start
        if duration <= 0:
            continue
        vf_filter = None
        if counter_total is not None:
            text = f"{idx}"
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
