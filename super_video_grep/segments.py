import string
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    norm: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = token.strip(string.punctuation)
    return token


def normalize_query(query: str) -> List[str]:
    parts: List[str] = []
    for raw in query.split():
        norm = normalize_token(raw)
        if norm:
            parts.append(norm)
    return parts


_DASH_RE = re.compile(r"[-–—−‑]")


def _split_compound(token: str) -> List[str]:
    return [part for part in _DASH_RE.split(token) if part]


def token_matches(token: str, query: str, mode: str) -> bool:
    if mode == "exact":
        if token == query:
            return True
        if _DASH_RE.search(token):
            return any(part == query for part in _split_compound(token))
        return False
    if mode == "prefix":
        return token.startswith(query)
    if mode == "substring":
        return query in token
    raise ValueError(f"unknown match mode: {mode}")


def find_phrase_matches(
    words: List[Word],
    phrase: List[str],
    match_mode: str = "exact",
) -> List[Tuple[float, float]]:
    if not phrase:
        return []
    matches: List[Tuple[float, float]] = []
    max_i = len(words) - len(phrase)
    for i in range(max_i + 1):
        ok = True
        for j, token in enumerate(phrase):
            if not token_matches(words[i + j].norm, token, match_mode):
                ok = False
                break
        if ok:
            start = words[i].start
            end = words[i + len(phrase) - 1].end
            matches.append((start, end))
    return matches


def find_any_phrase_matches(
    words: List[Word],
    phrases: List[List[str]],
    match_mode: str = "exact",
) -> List[Tuple[float, float]]:
    matches: List[Tuple[float, float]] = []
    for phrase in phrases:
        if not phrase:
            continue
        matches.extend(find_phrase_matches(words, phrase, match_mode=match_mode))
    return matches


def tokens_contain_phrase(tokens: List[str], phrase: List[str], match_mode: str = "exact") -> bool:
    if not phrase:
        return False
    max_i = len(tokens) - len(phrase)
    for i in range(max_i + 1):
        ok = True
        for j, token in enumerate(phrase):
            if not token_matches(tokens[i + j], token, match_mode):
                ok = False
                break
        if ok:
            return True
    return False


def pad_and_merge(
    segments: Iterable[Tuple[float, float]],
    padding: float,
    merge_gap: float,
    min_duration: float,
) -> List[Tuple[float, float]]:
    padded: List[Tuple[float, float]] = []
    for start, end in segments:
        start = max(0.0, start - padding)
        end = end + padding
        if end > start:
            padded.append((start, end))

    if not padded:
        return []

    padded.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []

    cur_start, cur_end = padded[0]
    for start, end in padded[1:]:
        if start <= cur_end + merge_gap:
            cur_end = max(cur_end, end)
            continue
        if cur_end - cur_start >= min_duration:
            merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end

    if cur_end - cur_start >= min_duration:
        merged.append((cur_start, cur_end))
    return merged
