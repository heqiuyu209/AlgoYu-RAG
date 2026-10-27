"""Versioned speech timeline. Text/clock bounds are evidence, not ASR accuracy.

JSON is stored immutably by HOJ. Chunks contain decoded speech text, so a
caller must validate a hit against its timed source rather than against the
JSON string's escaped bytes. No URL, filename, or executable text is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

MAX_VIDEO_BYTES = 1024 * 1024 * 1024
MAX_VIDEO_MS = 2 * 60 * 60 * 1000
MAX_VIDEO_CHARACTERS = 200000
MAX_VIDEO_SEGMENTS = 5000
SLICE_MS = 60000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Invalid video transcript")
        result[key] = value
    return result


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Invalid video transcript")
    return value


def _text(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > 4000
            or any(ord(c) < 32 and c not in "\n\t" for c in value)):
        raise ValueError("Invalid video transcript")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError("Invalid video transcript") from None
    return value


def clock(milliseconds: int) -> str:
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, fraction = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{fraction:03}"


def clock_ms(value: str) -> int:
    if not re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}", value):
        raise ValueError("Invalid video citation")
    return int(value[:2])*3600000 + int(value[3:5])*60000 + int(value[6:8])*1000 + int(value[9:])


@dataclass(frozen=True)
class Speech:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SpeechFrame:
    start: int
    end: int
    text: str

    @property
    def heading(self) -> str:
        return clock(self.start) + "–" + clock(self.end)


@dataclass(frozen=True)
class VideoText:
    duration: int
    segments: tuple[Speech, ...]

    @classmethod
    def parse(cls, content: str) -> VideoText:
        try:
            if not isinstance(content, str) or not 1 <= len(content) <= MAX_VIDEO_CHARACTERS:
                raise ValueError()
            body = json.loads(content, object_pairs_hook=_unique)
            if not isinstance(body, dict) or set(body) != {"schema", "durationMs", "segments"}:
                raise ValueError()
            _integer(body["schema"], 1, 1)
            duration = _integer(body["durationMs"], 1, MAX_VIDEO_MS)
            values = body["segments"]
            if not isinstance(values, list) or not 1 <= len(values) <= MAX_VIDEO_SEGMENTS:
                raise ValueError()
            segments, previous = [], 0
            for value in values:
                if not isinstance(value, dict) or set(value) != {"startMs", "endMs", "text"}:
                    raise ValueError()
                start = _integer(value["startMs"], previous, duration - 1)
                end = _integer(value["endMs"], start + 1, duration)
                segments.append(Speech(start, end, _text(value["text"])))
                previous = end
            return cls(duration, tuple(segments))
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise ValueError("Invalid video transcript") from None

    def canonical(self) -> str:
        return json.dumps({"schema": 1, "durationMs": self.duration,
                           "segments": [{"startMs": s.start, "endMs": s.end, "text": s.text} for s in self.segments]},
                          ensure_ascii=False, separators=(",", ":"))

    def frames(self) -> list[SpeechFrame]:
        """Keep adjacent speech together, with at most a minute per citation.

        A long individual segment is split by the ordinary chunker later;
        its source clock remains the verified original segment's range.
        """
        result, group, size = [], [], 0
        def flush():
            if group:
                result.append(SpeechFrame(group[0].start, group[-1].end, "\n".join(s.text for s in group)))
        for segment in self.segments:
            if group and (size + 1 + len(segment.text) > 1000 or segment.end - group[0].start > SLICE_MS):
                flush()
                group, size = [], 0
            group.append(segment)
            size += len(segment.text) + (1 if len(group) > 1 else 0)
        flush()
        return result

    def citation(self, heading: str, excerpt: str) -> tuple[int, int]:
        if not isinstance(heading, str) or len(heading) != 25 or heading[12] != "–":
            raise ValueError("Invalid video citation")
        start, end = clock_ms(heading[:12]), clock_ms(heading[13:])
        selected = [s for s in self.segments if s.start >= start and s.end <= end]
        if (not selected or selected[0].start != start or selected[-1].end != end or not isinstance(excerpt, str)
                or not excerpt.strip() or excerpt not in "\n".join(s.text for s in selected)):
            raise ValueError("Invalid video citation")
        return start, end
