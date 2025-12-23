import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .segments import normalize_query, tokens_contain_phrase


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


_TIME_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_time(match: re.Match, prefix: str) -> float:
    h = int(match.group(f"{prefix}h"))
    m = int(match.group(f"{prefix}m"))
    s = int(match.group(f"{prefix}s"))
    ms = int(match.group(f"{prefix}ms"))
    return h * 3600 + m * 60 + s + ms / 1000.0


def _clean_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = text.replace("\n", " ").strip()
    return text


def load_srt(path: str, encoding: str = "utf-8-sig") -> List[SubtitleSegment]:
    data = Path(path).read_text(encoding=encoding, errors="replace")
    blocks = re.split(r"\n\s*\n", data.strip(), flags=re.MULTILINE)
    segments: List[SubtitleSegment] = []
    for block in blocks:
        lines = [line.strip("\r") for line in block.splitlines() if line.strip("\r")]
        if len(lines) < 2:
            continue
        time_line = lines[1] if _TIME_RE.search(lines[1]) else lines[0]
        match = _TIME_RE.search(time_line)
        if not match:
            continue
        start = _parse_time(match, "s")
        end = _parse_time(match, "e")
        text_lines = lines[2:] if match and time_line == lines[1] else lines[1:]
        text = _clean_text("\n".join(text_lines))
        if not text:
            continue
        segments.append(SubtitleSegment(start=start, end=end, text=text))
    return segments


def match_subtitle_segments(
    segments: Iterable[SubtitleSegment],
    phrase: List[str],
    match_mode: str,
) -> List[SubtitleSegment]:
    matches: List[SubtitleSegment] = []
    for seg in segments:
        tokens = normalize_query(seg.text)
        if tokens_contain_phrase(tokens, phrase, match_mode=match_mode):
            matches.append(seg)
    return matches
