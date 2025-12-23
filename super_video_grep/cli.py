import argparse
import sys
import tempfile
from pathlib import Path
from typing import List

from .asr import build_model, transcribe
from .ffmpeg import concat_clips, cut_clips, ensure_ffmpeg, extract_audio_segment
from .segments import find_phrase_matches, normalize_query, pad_and_merge
from .subtitles import load_srt, match_subtitle_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="svgrep",
        description="Videogrep-style supercut using subtitles + ASR refinement",
    )
    parser.add_argument("query", help="phrase to search for")
    parser.add_argument("inputs", nargs="+", help="input video file(s)")
    parser.add_argument("-o", "--output", default="output.mp4", help="output video path")
    parser.add_argument("--language", default=None, help="language code (optional)")
    parser.add_argument("--padding", type=float, default=0.25, help="seconds padding")
    parser.add_argument("--merge-gap", type=float, default=0.20, help="merge gaps (sec)")
    parser.add_argument(
        "--min-duration", type=float, default=0.05, help="drop clips shorter than this"
    )
    parser.add_argument(
        "--subtitles",
        default=None,
        help="path to subtitle file (SRT) for matching",
    )
    parser.add_argument(
        "--subtitle-encoding",
        default="utf-8-sig",
        help="subtitle encoding (default: utf-8-sig)",
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
    return parser


def _print_segments(segments: List[tuple], label: str) -> None:
    for start, end in segments:
        print(f"{label}\t{start:.3f}\t{end:.3f}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.subtitles:
        print("--subtitles is required", file=sys.stderr)
        return 2
    if len(args.inputs) != 1:
        print("only one input is supported when using subtitles", file=sys.stderr)
        return 2

    query_tokens = normalize_query(args.query)
    if not query_tokens:
        print("query has no searchable tokens", file=sys.stderr)
        return 1

    try:
        ensure_ffmpeg()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    model = build_model("small", "cpu", "int8", None)

    all_clips: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, input_path in enumerate(args.inputs, start=1):
            input_path = str(Path(input_path))
            raw_matches = []
            try:
                subs = load_srt(args.subtitles, encoding=args.subtitle_encoding)
            except OSError as exc:
                print(f"failed to read subtitles: {exc}", file=sys.stderr)
                return 1
            matched_subs = match_subtitle_segments(subs, query_tokens, match_mode=args.match_mode)
            refined: List[tuple] = []
            for seg in matched_subs:
                seg_audio = str(Path(tmpdir) / f"seg_{idx:03d}_{int(seg.start)}.wav")
                try:
                    extract_audio_segment(input_path, seg_audio, seg.start, seg.end)
                except (RuntimeError, ValueError) as exc:
                    print(f"segment audio extract failed: {exc}", file=sys.stderr)
                    continue

                _local_segs, local_words, local_warnings = transcribe(
                    model,
                    seg_audio,
                    args.language,
                    vad_filter=False,
                    vad_parameters=None,
                    batch_size=1,
                )
                for warning in local_warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                local_matches = find_phrase_matches(
                    local_words, query_tokens, match_mode=args.match_mode
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

            if args.print_segments:
                _print_segments(segments, input_path)
                continue

            if not segments:
                continue

            prefix = f"clip_{idx:03d}"
            clips = cut_clips(input_path, segments, tmpdir, prefix)
            all_clips.extend(clips)

        if args.print_segments:
            return 0

        if not all_clips:
            print("no matches found", file=sys.stderr)
            return 2

        try:
            concat_clips(all_clips, args.output)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
