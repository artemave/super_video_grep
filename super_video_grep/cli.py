import argparse
import time
import sys
import tempfile
from pathlib import Path
from typing import List

from .asr import build_model, transcribe
from .ffmpeg import (
    concat_clips,
    cut_clips,
    ensure_ffmpeg,
    extract_audio_segment,
    extract_subtitles,
)
from .segments import find_any_phrase_matches, normalize_query, pad_and_merge
from .subtitles import load_srt, match_subtitle_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="svgrep",
        description="Videogrep-style supercut using subtitles + ASR refinement",
    )
    parser.add_argument("inputs", nargs="+", help="input video file(s)")
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        help="search term (OR); provide multiple times",
    )
    parser.add_argument("-o", "--output", default="output.mp4", help="output video path")
    parser.add_argument("--language", default=None, help="language code (optional)")
    parser.add_argument("--padding", type=float, default=0.25, help="seconds padding")
    parser.add_argument("--merge-gap", type=float, default=0.20, help="merge gaps (sec)")
    parser.add_argument(
        "--min-duration", type=float, default=0.05, help="drop clips shorter than this"
    )
    parser.add_argument(
        "--subtitles",
        action="append",
        default=[],
        help="path to subtitle file (SRT) for matching; one per input",
    )
    parser.add_argument(
        "--subtitles-from-media",
        action="store_true",
        help="extract subtitles from the input media file",
    )
    parser.add_argument(
        "--subtitle-language",
        default=None,
        help="subtitle language tag to extract (e.g., eng)",
    )
    parser.add_argument(
        "--subtitle-encoding",
        default="auto",
        help="subtitle encoding (default: auto)",
    )
    parser.add_argument(
        "--match-mode",
        choices=("exact", "prefix", "substring"),
        default="exact",
        help="how query tokens match transcript tokens",
    )
    parser.add_argument(
        "--print-segments",
        action="store_true",
        help="print matched segments and exit",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="print timing info for segment extraction and ASR",
    )
    parser.add_argument(
        "--no-timing",
        action="store_false",
        dest="timing",
        help="disable timing output",
    )
    parser.add_argument(
        "--counter",
        default=True,
        action="store_true",
        help="overlay a running match counter on each clip",
    )
    return parser


def _print_segments(segments: List[tuple], label: str) -> None:
    for start, end in segments:
        print(f"{label}\t{start:.3f}\t{end:.3f}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.subtitles and args.subtitles_from_media:
        print("use either --subtitles or --subtitles-from-media, not both", file=sys.stderr)
        return 2
    if not args.subtitles and not args.subtitles_from_media:
        print("either --subtitles or --subtitles-from-media is required", file=sys.stderr)
        return 2
    if args.subtitles and len(args.subtitles) != len(args.inputs):
        print("--subtitles must be provided once per input file", file=sys.stderr)
        return 2

    query_strings = list(args.search or [])
    if not query_strings:
        print("at least one --search is required", file=sys.stderr)
        return 2
    query_tokens_list = [normalize_query(q) for q in query_strings]
    query_tokens_list = [q for q in query_tokens_list if q]
    if not query_tokens_list:
        print("query has no searchable tokens", file=sys.stderr)
        return 1

    try:
        ensure_ffmpeg()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    model = build_model("small", "cpu", "int8", None)

    all_clips: List[str] = []
    total_extract = 0.0
    total_asr = 0.0
    total_segments = 0
    total_cut = 0.0
    total_concat = 0.0
    run_start = time.perf_counter()

    counter_index = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, input_path in enumerate(args.inputs, start=1):
            input_path = str(Path(input_path))
            raw_matches = []
            srt_path = None
            if args.subtitles_from_media:
                srt_path = str(Path(tmpdir) / f"embedded_{idx:03d}.srt")
                try:
                    extract_subtitles(
                        input_path,
                        srt_path,
                        language=args.subtitle_language,
                    )
                except RuntimeError as exc:
                    print(f"failed to extract subtitles: {exc}", file=sys.stderr)
                    return 1
            else:
                srt_path = args.subtitles[idx - 1]
            try:
                subs = load_srt(srt_path, encoding=args.subtitle_encoding)
            except OSError as exc:
                print(f"failed to read subtitles: {exc}", file=sys.stderr)
                return 1
            matched_subs = match_subtitle_segments(
                subs, query_tokens_list, match_mode=args.match_mode
            )
            print(f"found {len(matched_subs)} matching subtitle segments", file=sys.stderr)
            refined: List[tuple] = []
            for seg in matched_subs:
                total_segments += 1
                seg_audio = str(Path(tmpdir) / f"seg_{idx:03d}_{int(seg.start)}.wav")
                t0 = time.perf_counter()
                try:
                    extract_audio_segment(input_path, seg_audio, seg.start, seg.end)
                except (RuntimeError, ValueError) as exc:
                    print(f"segment audio extract failed: {exc}", file=sys.stderr)
                    continue
                t1 = time.perf_counter()

                _local_segs, local_words, local_warnings = transcribe(
                    model,
                    seg_audio,
                    args.language,
                    vad_filter=False,
                    vad_parameters=None,
                    batch_size=1,
                )
                t2 = time.perf_counter()

                extract_time = t1 - t0
                asr_time = t2 - t1
                total_extract += extract_time
                total_asr += asr_time
                if args.timing:
                    print(
                        "segment\t"
                        f"{seg.start:.3f}\t{seg.end:.3f}\t"
                        f"extract={extract_time:.3f}s\tasr={asr_time:.3f}s",
                        file=sys.stderr,
                    )
                for warning in local_warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                local_matches = find_any_phrase_matches(
                    local_words, query_tokens_list, match_mode=args.match_mode
                )
                if local_matches:
                    for start, end in local_matches:
                        refined.append((seg.start + start, seg.start + end))
                else:
                    refined.append((seg.start, seg.end))
            raw_matches = refined
            segments = pad_and_merge(
                raw_matches,
                padding=args.padding,
                merge_gap=args.merge_gap,
                min_duration=args.min_duration,
            )
            counter_counts = None
            if args.counter and segments:
                counter_counts = []
                for seg_start, seg_end in segments:
                    count = 0
                    for m_start, m_end in raw_matches:
                        if m_end >= seg_start and m_start <= seg_end:
                            count += 1
                    counter_counts.append(count)

            if args.print_segments:
                _print_segments(segments, input_path)
                continue

            if not segments:
                continue

            prefix = f"clip_{idx:03d}"
            cut_start = time.perf_counter()
            counter_start = counter_index if args.counter else None
            clips = cut_clips(
                input_path, segments, tmpdir, prefix, counter_start, counter_counts
            )
            cut_end = time.perf_counter()
            total_cut += cut_end - cut_start
            all_clips.extend(clips)
            if args.counter:
                counter_index += sum(counter_counts or [])

        if args.print_segments:
            return 0

        if not all_clips:
            print("no matches found", file=sys.stderr)
            return 2

        concat_start = time.perf_counter()
        try:
            concat_clips(all_clips, args.output)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        concat_end = time.perf_counter()
        total_concat += concat_end - concat_start

    if args.timing:
        total_time = time.perf_counter() - run_start
        print(
            "timing_total\t"
            f"segments={total_segments}\t"
            f"extract={total_extract:.3f}s\tasr={total_asr:.3f}s\t"
            f"cut={total_cut:.3f}s\tconcat={total_concat:.3f}s\t"
            f"elapsed={total_time:.3f}s",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
