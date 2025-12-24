# super_video_grep

[Videogrep](https://github.com/antiboredom/videogrep)-style supercuts using subtitle matches refined with ASR word timings (local Whisper via `faster-whisper`).

Demo. All occurances of word "cunt" in the movie The Fafourite:



https://github.com/user-attachments/assets/bcbd6e73-ffd3-40a7-93b3-81e1b082433e

This is 100% llm code - I didn't even read it once.

## Install (uv)

```bash
pip install uv
```

## Usage (uv run)

```bash
uv run svgrep "hello world" input.mp4 --subtitles input.srt -o output.mp4
```

Print matched segments without rendering:

```bash
uv run svgrep "hello" input.mp4 --subtitles input.srt --print-segments
```

### Options

- `--language`: language code (optional)
- `-o`, `--output`: output video path (default: `output.mp4`)
- `--padding`: seconds added before/after each match
- `--merge-gap`: merge segments if gap is within this many seconds
- `--min-duration`: drop segments shorter than this
- `--subtitles`: path to SRT file to match on subtitle text (required)
- `--subtitle-encoding`: subtitle file encoding (default: `utf-8-sig`)
- `--match-mode`: token match mode (`exact`, `prefix`, `substring`)
- `--print-segments`: print matched segments and exit
- `--timing`: print timing info for extraction and ASR

## Notes

- Requires `ffmpeg` in PATH.
- Word-level matching is exact on normalized tokens (case/punctuation stripped).
- Subtitles are required; run once per input file.
