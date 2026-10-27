"""Private immutable-asset bridge to bounded, offline media subprocesses."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys

from learning_os.hoj.auth import check_deadline, now_ms
from learning_os.hoj.contracts import ServiceError, VideoAsset, VideoSlice
from learning_os.hoj.video_text import MAX_VIDEO_MS, VideoText

ENGINE = "faster-whisper-cpu-int8-v1"
_FAILURES = {"VIDEO_INVALID", "VIDEO_NO_AUDIO", "VIDEO_UNSUPPORTED_CODEC", "VIDEO_DURATION_LIMIT",
             "VIDEO_RESOURCE_LIMIT", "VIDEO_INVALID_SLICE"}
_MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


def snapshot(value) -> tuple:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _number(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError()
    return value


class VideoExtractor:
    def __init__(self, assets: Path | None = None, model: Path | None = None):
        self.assets, self.model = assets, model

    @classmethod
    def from_env(cls):
        assets, model = os.environ.get("HOJ_RAG_MEDIA_DIR"), os.environ.get("HOJ_RAG_ASR_MODEL_PATH")
        return cls(Path(assets).resolve() if assets else None, Path(model).resolve() if model else None)

    @contextmanager
    def source(self, request: VideoAsset, deadline: int):
        if not sys.platform.startswith("linux"):
            raise ServiceError(503, "VIDEO_RUNTIME_UNAVAILABLE")
        if self.assets is None or not self.assets.is_dir():
            raise ServiceError(503, "VIDEO_ASSETS_UNAVAILABLE")
        # The asset identifier is a strict 32-character UUID, never a path.
        path = self.assets / (request.assetId + ".bin")
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            state = os.fstat(fd)
            identity = snapshot(state)
            if not stat.S_ISREG(state.st_mode) or state.st_size != request.byteLength:
                raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
            # Even nanosecond stat fields can collide within one kernel tick.
            # A previous hash under matching metadata cannot authorize new IO.
            if (self.source_hash(fd, request.byteLength, deadline) != request.sourceHash
                    or snapshot(os.fstat(fd)) != identity
                    or snapshot(path.stat(follow_symlinks=False)) != identity):
                raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
            check_deadline(deadline)
            yield fd
            if snapshot(os.fstat(fd)) != identity or snapshot(path.stat(follow_symlinks=False)) != identity:
                raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
            if (self.source_hash(fd, request.byteLength, deadline) != request.sourceHash
                    or snapshot(os.fstat(fd)) != identity
                    or snapshot(path.stat(follow_symlinks=False)) != identity):
                raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
        except FileNotFoundError:
            raise ServiceError(404, "VIDEO_ASSET_MISSING") from None
        except OSError:
            raise ServiceError(409, "VIDEO_SOURCE_CHANGED") from None
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def source_hash(fd: int, length: int, deadline: int) -> str:
        # dup() shares its offset with the worker's descriptor. Always rewind,
        # and read at most the declared bytes plus one overflow byte.
        os.lseek(fd, 0, os.SEEK_SET)
        digest, remaining = hashlib.sha256(), length
        with os.fdopen(os.dup(fd), "rb") as stream:
            while remaining:
                check_deadline(deadline)
                chunk = stream.read(min(1048576, remaining))
                if not chunk:
                    raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
                digest.update(chunk)
                remaining -= len(chunk)
            check_deadline(deadline)
            if stream.read(1):
                raise ServiceError(409, "VIDEO_SOURCE_CHANGED")
        os.lseek(fd, 0, os.SEEK_SET)
        return digest.hexdigest()

    def model_hash(self, deadline: int) -> str:
        if self.model is None or not self.model.is_dir():
            raise ServiceError(503, "VIDEO_ASR_UNAVAILABLE")
        try:
            paths = [self.model / name for name in _MODEL_FILES]
            state = tuple(snapshot(p.stat()) for p in paths)
            digest = hashlib.sha256()
            for path in paths:
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError()
                digest.update((path.name + "\n" + str(path.stat().st_size) + "\n").encode("ascii"))
                with path.open("rb") as source:
                    while chunk := source.read(1048576):
                        check_deadline(deadline)
                        digest.update(chunk)
            if tuple(snapshot(p.stat()) for p in paths) != state:
                raise ServiceError(409, "VIDEO_MODEL_CHANGED")
            # Small config/tokenizer changes can also retain the same metadata.
            # Recompute the complete revision before and after transcription.
            return digest.hexdigest()
        except (OSError, ValueError):
            raise ServiceError(503, "VIDEO_ASR_UNAVAILABLE") from None

    def extract(self, request: VideoAsset, deadline: int) -> dict:
        check_deadline(deadline)
        transcribe = isinstance(request, VideoSlice)
        action = "transcribe" if transcribe else "probe"
        model_hash = self.model_hash(deadline) if transcribe else None
        if transcribe and request.expectedModelHash is not None and request.expectedModelHash != model_hash:
            raise ServiceError(409, "VIDEO_MODEL_CHANGED")
        with self.source(request, deadline) as fd:
            body = request.model_dump() | {"action": action, "sourceFd": fd}
            if transcribe:
                body.update(modelPath=str(self.model), modelHash=model_hash)
            raw = self.run(body, fd, deadline, transcribe)
            result = self.validate(raw, request, model_hash)
        if transcribe and self.model_hash(deadline) != model_hash:
            raise ServiceError(409, "VIDEO_MODEL_CHANGED")
        check_deadline(deadline)
        return result

    def run(self, body, fd, deadline, transcribe):
        env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH"}}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        check_deadline(deadline)
        timeout = max(.001, min(95 if transcribe else 25, (deadline-now_ms())/1000))
        try:
            child = subprocess.Popen([sys.executable,"-m","learning_os.hoj.video_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, pass_fds=(fd,), env=env, start_new_session=True)
        except OSError:
            raise ServiceError(503, "VIDEO_RUNTIME_UNAVAILABLE") from None
        def kill():
            try:
                os.killpg(child.pid, getattr(signal,"SIGKILL",9))
            except ProcessLookupError:
                pass
        try:
            try:
                output,_ = child.communicate(json.dumps(body,separators=(",",":")).encode(),timeout=timeout)
            except subprocess.TimeoutExpired:
                kill()
                child.communicate()
                raise ServiceError(400,"VIDEO_RESOURCE_LIMIT") from None
        finally:
            if child.poll() is None:
                kill()
                child.wait(timeout=5)
            kill()
            for pipe in (child.stdin,child.stdout):
                if pipe is not None:pipe.close()
        if child.returncode != 0:
            raise ServiceError(400,"VIDEO_RESOURCE_LIMIT")
        return output

    def validate(self, raw, request, model_hash):
        try:
            if len(raw) > 131072:
                raise ValueError()
            def unique(pairs):
                result = {}
                for k,v in pairs:
                    if k in result:raise ValueError()
                    result[k]=v
                return result
            body = json.loads(raw,object_pairs_hook=unique)
            if isinstance(body,dict) and set(body)=={"code"}:
                if body["code"] in _FAILURES:raise ServiceError(400,body["code"])
                if body["code"] in {"VIDEO_RUNTIME_UNAVAILABLE","VIDEO_ASR_FAILED"}:raise ServiceError(503,body["code"])
                raise ValueError()
            for field in ("assetId","sourceHash","byteLength"):
                if type(body[field]) is not type(getattr(request,field)) or body[field]!=getattr(request,field):raise ValueError()
            duration = _number(body["durationMs"],1,MAX_VIDEO_MS)
            if isinstance(request,VideoSlice):
                if set(body)!={"assetId","sourceHash","byteLength","startMs","endMs","durationMs","language","segments","engine","modelHash"}:raise ValueError()
                if (type(body["startMs"]) is not int or type(body["endMs"]) is not int
                        or body["startMs"]!=request.startMs or body["endMs"]!=request.endMs or request.endMs>duration
                        or body["engine"]!=ENGINE or body["modelHash"]!=model_hash
                        or not isinstance(body["language"],str) or not re.fullmatch(r"[a-z]{2,3}",body["language"])
                        or not isinstance(body["segments"],list) or len(body["segments"])>200):raise ValueError()
                if body["segments"]:
                    speech=VideoText.parse(json.dumps({"schema":1,"durationMs":duration,"segments":body["segments"]},ensure_ascii=False))
                    if (speech.segments[0].start<request.startMs or speech.segments[-1].end>request.endMs
                            or sum(len(s.text) for s in speech.segments)>20000):raise ValueError()
            else:
                if set(body)!={"assetId","sourceHash","byteLength","durationMs","mime","width","height","videoCodec","audioCodec","audioIndex"}:raise ValueError()
                width,height=_number(body["width"],1,4096),_number(body["height"],1,2304)
                _number(body["audioIndex"],0,31)
                if width*height>9437184:raise ValueError()
                if not ((body["mime"]=="video/mp4" and body["videoCodec"]=="h264" and body["audioCodec"]=="aac")
                        or (body["mime"]=="video/webm" and body["videoCodec"] in {"vp8","vp9","av1"} and body["audioCodec"] in {"opus","vorbis"})):raise ValueError()
            return body
        except (ValueError,TypeError,KeyError,RecursionError):
            raise ServiceError(502,"INVALID_VIDEO_RESULT") from None
