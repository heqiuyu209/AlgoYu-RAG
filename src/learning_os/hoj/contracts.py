"""Bounded internal wire contract. Identity and publication belong to HOJ."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator
from learning_os.hoj.video_text import MAX_VIDEO_BYTES, MAX_VIDEO_MS, SLICE_MS
from learning_os.hoj.pdf_text import MAX_PDF_CHARACTERS


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VersionRef(Contract):
    documentId: StrictStr = Field(pattern=r"^[a-f0-9]{32}$")
    version: StrictInt = Field(ge=1, le=2147483647)
    contentHash: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")

    def key(self) -> str:
        return f"{self.documentId}:{self.version}"


class Ingest(VersionRef):
    # Text is UTF-8, with line endings already canonicalized by HOJ.
    title: StrictStr = Field(min_length=1, max_length=200)
    format: StrictStr = Field(pattern=r"^(markdown|text|pdf|video)$")
    content: StrictStr = Field(min_length=1, max_length=MAX_PDF_CHARACTERS)

    @model_validator(mode="after")
    def content_capacity(self):
        if self.format != "pdf" and len(self.content) > 200000:
            raise ValueError("Text and video retain the existing character limit")
        return self

    @field_validator("title", "content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("blank text or NUL is not supported")
        return value


class Retrieve(Contract):
    query: StrictStr = Field(min_length=1, max_length=2000)
    allowedVersions: list[VersionRef] = Field(max_length=500)
    topK: StrictInt = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("blank query or NUL is not supported")
        return value


class VideoAsset(Contract):
    assetId: StrictStr = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    sourceHash: StrictStr = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    byteLength: StrictInt = Field(ge=12, le=MAX_VIDEO_BYTES)


class VideoSlice(VideoAsset):
    startMs: StrictInt = Field(ge=0, lt=MAX_VIDEO_MS)
    endMs: StrictInt = Field(ge=1, le=MAX_VIDEO_MS)
    language: StrictStr = Field(default="auto", pattern=r"^(auto|zh|en)$")
    expectedModelHash: StrictStr | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def bounded_slice(self):
        if not 0 < self.endMs - self.startMs <= SLICE_MS:
            raise ValueError("Invalid video slice")
        return self


class ServiceError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code
        super().__init__(code)
