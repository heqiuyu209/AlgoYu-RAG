"""One isolated local video probe / bounded speech slice per process.

The only source is an inherited read-only descriptor, not a browser path.
Network protocols, external MOV tracks and subtitle/data decoders are disabled.
The local speech model directory is operator-owned and loaded offline.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys

from learning_os.hoj.video_text import MAX_VIDEO_MS, VideoText

ENGINE = "faster-whisper-cpu-int8-v1"
# Recognition vocabulary, never appended to an excerpt or treated as evidence.
# General terms from HOJ's fixed XCPC outline; no question answers, per-user
# data, or fixture/reference sentences are passed to the speech model.
DOMAIN_TERMS = ("前缀和, 前缀数组, 差分, 二分, 双指针, 排序, 贪心, 递归, 分治, 动态规划, "
                "图论, 最短路径, 最小生成树, 拓扑排序, 树状数组, 线段树, 并查集, "
                "深度优先搜索, 广度优先搜索, 字符串, 哈希, 模运算, 常数时间, 时间复杂度, "
                "prefix sum, binary search, dynamic programming, Dijkstra, Bellman-Ford, "
                "Fenwick tree, segment tree, DFS, BFS, time complexity")


class VideoFailure(Exception):
    def __init__(self, code):
        self.code = code


def input_options(fd: int) -> list[str]:
    header = os.pread(fd, 4096, 0)
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return ["-protocol_whitelist", "file,pipe", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0"]
    if header.startswith(b"\x1a\x45\xdf\xa3") and b"webm" in header:
        return ["-protocol_whitelist", "file,pipe", "-f", "matroska"]
    raise VideoFailure("VIDEO_INVALID")


def probe(fd: int) -> dict:
    command = ["ffprobe", "-v", "error", "-max_alloc", "33554432", "-max_streams", "32",
               "-probesize", "8388608", "-analyzeduration", "5000000", *input_options(fd),
               "-i", f"/proc/self/fd/{fd}", "-show_entries",
               "format=duration:stream=index,codec_type,codec_name,width,height,duration,channels,sample_rate:stream_disposition=default,attached_pic",
               "-of", "json"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                pass_fds=(fd,), timeout=15, check=False)
    except FileNotFoundError:
        raise VideoFailure("VIDEO_RUNTIME_UNAVAILABLE") from None
    except subprocess.TimeoutExpired:
        raise VideoFailure("VIDEO_RESOURCE_LIMIT") from None
    if result.returncode != 0 or len(result.stdout) > 65536:
        raise VideoFailure("VIDEO_INVALID")
    try:
        body = json.loads(result.stdout)
        streams = body["streams"]
        if not isinstance(streams, list) or not 1 <= len(streams) <= 32:
            raise ValueError()
        videos = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        if not videos:
            raise VideoFailure("VIDEO_INVALID")
        if not audios:
            raise VideoFailure("VIDEO_NO_AUDIO")
        video = next((s for s in videos if s.get("disposition", {}).get("default")), videos[0])
        audio = next((s for s in audios if s.get("disposition", {}).get("default")), audios[0])
        mp4 = input_options(fd)[3] == "mov"
        if (video.get("codec_name") not in ({"h264"} if mp4 else {"vp8", "vp9", "av1"})
                or audio.get("codec_name") not in ({"aac"} if mp4 else {"opus", "vorbis"})):
            raise VideoFailure("VIDEO_UNSUPPORTED_CODEC")
        width, height = int(video["width"]), int(video["height"])
        if not (1 <= width <= 4096 and 1 <= height <= 2304 and width*height <= 9437184
                and 1 <= int(audio["channels"]) <= 8 and 1 <= int(audio["sample_rate"]) <= 192000):
            raise VideoFailure("VIDEO_RESOURCE_LIMIT")
        durations = [float(s["duration"]) for s in streams if s.get("duration") not in (None, "N/A")]
        durations.append(float(body["format"]["duration"]))
        if not all(math.isfinite(n) and n >= 0 for n in durations):
            raise ValueError()
        duration = math.ceil(max(durations)*1000)
        if not 1 <= duration <= MAX_VIDEO_MS:
            raise VideoFailure("VIDEO_DURATION_LIMIT")
        return {"durationMs": duration, "mime": "video/mp4" if mp4 else "video/webm", "width": width,
                "height": height, "videoCodec": video["codec_name"], "audioCodec": audio["codec_name"],
                "audioIndex": int(audio["index"])}
    except VideoFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError):
        raise VideoFailure("VIDEO_INVALID") from None


def decode(fd: int, metadata: dict, start: int, end: int):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-max_alloc", "33554432", "-threads", "2",
               "-probesize", "8388608", "-analyzeduration", "5000000", *input_options(fd),
               "-ss", f"{start/1000:.3f}", "-i", f"/proc/self/fd/{fd}", "-t", f"{(end-start)/1000:.3f}",
               "-map", f"0:{metadata['audioIndex']}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
               "-af", "aresample=async=1:first_pts=0", "-fs", "4194304", "-f", "s16le", "pipe:1"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                pass_fds=(fd,), timeout=20, check=False)
    except subprocess.TimeoutExpired:
        raise VideoFailure("VIDEO_RESOURCE_LIMIT") from None
    if result.returncode != 0:
        raise VideoFailure("VIDEO_INVALID")
    if len(result.stdout) > math.ceil((end-start)*16)*2+4096 or len(result.stdout)%2:
        raise VideoFailure("VIDEO_RESOURCE_LIMIT")
    import numpy as np
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32)/32768


def transcribe(fd: int, request: dict, metadata: dict) -> dict:
    start, end = request["startMs"], request["endMs"]
    if end > metadata["durationMs"]:
        raise VideoFailure("VIDEO_INVALID_SLICE")
    audio = decode(fd, metadata, start, end)
    language, speech = "und", []
    if audio.size and bool((audio != 0).any()):
        from faster_whisper import WhisperModel
        model = WhisperModel(request["modelPath"], device="cpu", compute_type="int8", cpu_threads=2,
                             num_workers=1, local_files_only=True)
        segments, info = model.transcribe(audio, language=None if request["language"] == "auto" else request["language"],
                                          beam_size=5, temperature=0, condition_on_previous_text=False,
                                          vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
                                          hotwords=DOMAIN_TERMS,
                                          word_timestamps=False)
        language = info.language
        count, previous = 0, start
        for segment in segments:
            if not math.isfinite(segment.start) or not math.isfinite(segment.end):
                raise VideoFailure("VIDEO_ASR_FAILED")
            left, right = max(start,start+round(segment.start*1000)), min(end,start+round(segment.end*1000))
            text = segment.text.strip()
            if not text:
                continue
            if left < previous or right <= left:
                raise VideoFailure("VIDEO_ASR_FAILED")
            count += len(text)
            if count > 20000 or len(speech) >= 200:
                raise VideoFailure("VIDEO_RESOURCE_LIMIT")
            speech.append({"startMs": left, "endMs": right, "text": text})
            previous = right
    if speech:
        try:
            VideoText.parse(json.dumps({"schema":1,"durationMs":metadata["durationMs"],"segments":speech},ensure_ascii=False))
        except ValueError:
            raise VideoFailure("VIDEO_ASR_FAILED") from None
    return {"startMs":start,"endMs":end,"durationMs":metadata["durationMs"],"language":language,
            "segments":speech,"engine":ENGINE,"modelHash":request["modelHash"]}


def main():
    if not sys.platform.startswith("linux"):
        sys.stdout.write('{"code":"VIDEO_RUNTIME_UNAVAILABLE"}')
        return
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(90,90))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    logging.disable(logging.CRITICAL)
    try:
        request = json.loads(sys.stdin.buffer.read(4096))
        fd = request["sourceFd"]
        with open(os.devnull,"w") as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            metadata = probe(fd)
            result = metadata if request["action"] == "probe" else transcribe(fd,request,metadata)
            result.update({name:request[name] for name in ["assetId","sourceHash","byteLength"]})
    except VideoFailure as error:
        result={"code":error.code}
    except (ImportError, FileNotFoundError):
        result={"code":"VIDEO_RUNTIME_UNAVAILABLE"}
    except MemoryError:
        result={"code":"VIDEO_RESOURCE_LIMIT"}
    except Exception:
        result={"code":"VIDEO_ASR_FAILED"}
    sys.stdout.buffer.write(json.dumps(result,ensure_ascii=False,separators=(",",":")).encode("utf-8"))


if __name__ == "__main__":
    main()
