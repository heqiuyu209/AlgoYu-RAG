"""Local keyword projection. Only complete caller-authorized versions compete.

FTS stores normalized terms and chunk identities, not a second excerpt source.
Quoted hex terms keep user text out of both SQL and the FTS query language.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Iterator

from learning_os.hoj.auth import check_deadline, now_ms

_CJK = r"\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f"
_PARTS = re.compile(r"[" + _CJK + r"]+|[^\W_" + _CJK + r"]+(?:_[^\W_" + _CJK + r"]+)*", re.UNICODE)
_HAN = re.compile(r"^[" + _CJK + r"]+$")
_CODE_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]{1,63}(?![A-Za-z0-9_])")
_CODE_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_INLINE_CODE = re.compile(r"(?<![\\`])(`+)(?!`)(.*?)(?<!`)\1(?!`)")


def _inline_code_identifiers(query: str) -> Iterator[str]:
    fence = ""
    for line in query.splitlines():
        marker = _CODE_FENCE.match(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker and (marker[1][0] == "~" or "`" not in marker[2]):
            fence = marker[1]
            continue
        for code in _INLINE_CODE.finditer(line):
            name = code[2].strip()
            if _CODE_IDENTIFIER.fullmatch(name):
                yield name


def code_identifiers(query: str) -> list[str]:
    """Conservative explicit names; ordinary words retain semantic ranking."""
    names = list(dict.fromkeys(match.group() for match in _CODE_IDENTIFIER.finditer(query)
                               if "_" in match.group()))
    if len(names) > 4:
        return []
    if "`" not in query:
        return names
    explicit = list(dict.fromkeys(names + list(_inline_code_identifiers(query))))
    return explicit if len(explicit) <= 4 else names


def mentioned_identifiers(text: str, names) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {name for name in names if re.search(
        r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text)}


def terms(text: str, *, query: bool = False) -> list[str]:
    result = []
    for match in _PARTS.finditer(unicodedata.normalize("NFKC", text).casefold()):
        part = match.group()
        if _HAN.fullmatch(part):
            pieces = list(part) if not query or len(part) == 1 else []
            pieces.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            pieces = [part] if len(part) <= 64 else []
        result.extend("t" + piece.encode("utf-8").hex() for piece in pieces)
    # 64 bounded OR clauses, no syntax supplied by the user. Document terms
    # retain frequency; query terms do not count the same phrase repeatedly.
    return list(dict.fromkeys(result))[:64] if query else result


def initialize(db: sqlite3.Connection):
    db.execute("CREATE TABLE IF NOT EXISTS lexical_chunk (rowid INTEGER PRIMARY KEY,"
               "namespace TEXT NOT NULL,doc_id TEXT NOT NULL,version INTEGER NOT NULL,"
               "content_hash TEXT NOT NULL,chunk_id TEXT NOT NULL,tokens TEXT NOT NULL,"
               "UNIQUE(namespace,chunk_id))")
    db.execute("CREATE INDEX IF NOT EXISTS lexical_version ON lexical_chunk(namespace,doc_id,version,content_hash)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(tokens,"
               "content='lexical_chunk',content_rowid='rowid',tokenize='ascii')")
    db.execute("CREATE TRIGGER IF NOT EXISTS lexical_insert AFTER INSERT ON lexical_chunk BEGIN "
               "INSERT INTO lexical_fts(rowid,tokens) VALUES(new.rowid,new.tokens); END")
    db.execute("CREATE TRIGGER IF NOT EXISTS lexical_delete AFTER DELETE ON lexical_chunk BEGIN "
               "INSERT INTO lexical_fts(lexical_fts,rowid,tokens) VALUES('delete',old.rowid,old.tokens); END")
    db.execute("CREATE TRIGGER IF NOT EXISTS lexical_update AFTER UPDATE OF tokens ON lexical_chunk BEGIN "
               "INSERT INTO lexical_fts(lexical_fts,rowid,tokens) VALUES('delete',old.rowid,old.tokens);"
               "INSERT INTO lexical_fts(rowid,tokens) VALUES(new.rowid,new.tokens); END")


def projection(namespace: str, request, chunks) -> list[tuple[str, str]]:
    from learning_os.hoj.index import digest
    return [(digest(f"{namespace}:{request.key()}:{i}:{digest(c.content)}"),
             " ".join(terms(f"{request.title}\n{c.metadata.chapter or ''}\n{c.content}")))
            for i, c in enumerate(chunks)]


def matches(db, namespace, request, rows) -> bool:
    stored = db.execute("SELECT c.chunk_id,c.tokens,d.id,c.content_hash FROM lexical_chunk c "
                        "LEFT JOIN lexical_fts_docsize d ON d.id=c.rowid "
                        "WHERE c.namespace=? AND c.doc_id=? AND c.version=? LIMIT 401",
                        (namespace, request.documentId, request.version)).fetchall()
    return (len(stored) == len(rows) and all(row[2] is not None and row[3] == request.contentHash for row in stored)
            and sorted((row[0], row[1]) for row in stored) == sorted(rows))


def delete(db, namespace, ref):
    db.execute("DELETE FROM lexical_chunk WHERE namespace=? AND doc_id=? AND version=?",
               (namespace, ref.documentId, ref.version))


def replace(db, namespace, request, rows):
    delete(db, namespace, request)
    db.executemany("INSERT INTO lexical_chunk(namespace,doc_id,version,content_hash,chunk_id,tokens) VALUES(?,?,?,?,?,?)",
                   [(namespace, request.documentId, request.version, request.contentHash, chunk, text) for chunk, text in rows])


def search(db, namespace, refs, query, limit, deadline) -> list[str]:
    words = terms(query, query=True)
    if not words or not refs:
        return []
    db.set_progress_handler(lambda: int(now_ms() >= deadline), 1000)
    try:
        db.execute("CREATE TEMP TABLE authorized_version(doc_id TEXT,version INTEGER,content_hash TEXT,"
                   "PRIMARY KEY(doc_id,version,content_hash))")
        db.executemany("INSERT INTO authorized_version VALUES(?,?,?)",
                       [(ref.documentId, ref.version, ref.contentHash) for ref in refs])
        # Join scope and READY before ORDER BY/LIMIT: inaccessible candidates
        # cannot use up the result slots. Namespace is checked independently.
        rows = db.execute("SELECT c.chunk_id FROM lexical_fts "
                          "JOIN lexical_chunk c ON c.rowid=lexical_fts.rowid "
                          "JOIN authorized_version a ON a.doc_id=c.doc_id AND a.version=c.version AND a.content_hash=c.content_hash "
                          "JOIN version_manifest m ON m.namespace=c.namespace AND m.doc_id=c.doc_id AND m.version=c.version AND m.content_hash=c.content_hash "
                          "WHERE lexical_fts MATCH ? AND c.namespace=? AND m.state='READY' "
                          "ORDER BY bm25(lexical_fts),c.chunk_id LIMIT ?",
                          (" OR ".join('"' + word + '"' for word in words), namespace, limit)).fetchall()
        return [row[0] for row in rows]
    except sqlite3.OperationalError:
        check_deadline(deadline)
        raise
    finally:
        db.set_progress_handler(None, 0)


def fuse(dense: list[str], lexical: list[str], limit: int) -> list[str]:
    # RRF scores determine rank only. Wire score stays cosine similarity.
    scores = {}
    for ranking in (dense, lexical):
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    dense_position = {key: rank for rank, key in enumerate(dense)}
    return sorted(scores, key=lambda key: (-scores[key], dense_position.get(key, len(dense)), key))[:limit]
