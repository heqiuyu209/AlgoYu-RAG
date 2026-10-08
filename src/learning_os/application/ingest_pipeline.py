import logging
import math
from pathlib import Path

from learning_os.domain import Chunk, EmbeddingError, ChunkingError
from learning_os.domain.interfaces import Parser, Chunker, Embedder, VectorStore, Cleaner

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    文档入库流水线（Application Service）。
    负责编排解析、清洗、分块、向量化和存储的完整流程。
    """

    def __init__(
            self,
            parsers: list[Parser],
            chunker: Chunker,
            embedder: Embedder,
            vector_store: VectorStore,
            cleaner: Cleaner | None = None,
    ):
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.store = vector_store
        self.cleaner = cleaner

    def _select_parser(self, path: Path) -> Parser | None:
        for parser in self.parsers:
            if parser.supports(path):
                return parser

        return None

    def run(
            self,
            file_path: Path | str,
            subject: str | None = None,
            topic: str | None = None,
            language: str | None = None,
    ) -> int:

        """
        执行入库流程。返回成功入库的 Chunk 数量。
        """
        path = Path(file_path)
        logger.info(f"🚀 开始处理文件: {path.name}")

        # 1. 解析 (Parse)
        parser = self._select_parser(path)

        if parser is None:
            raise ValueError(f"No parser supports file: {path}")

        document = parser.parse(path)

        if subject:
            document.metadata["subject"] = subject

        if topic:
            document.metadata["topic"] = topic

        if language:
            document.metadata["language"] = language

        # 2. 清洗 (Clean) - 如果配置了 Cleaner，就洗一下
        if self.cleaner:
            document.content = self.cleaner.clean(document.content)

        # 检查正文是否为空
        if not document.content.strip():
            logger.info(f"文件 {path.name} 没有有效正文，跳过。")
            return 0

        # 3. 分块 (Chunk)
        chunks = self.chunker.chunk(document)

        # 分块器返回列表
        if not isinstance(chunks, list):
            raise ChunkingError("Chunker must return a list of chunks.")

        # 前面确认非空，却没有产生chunk，抛出异常
        if not chunks:
            raise ChunkingError(
                "Non-empty document produced no chunks."
            )

        # 检查每个 Chunk 的正文
        for index, chunk in enumerate(chunks):
            # 1. 检查对象是不是 Chunk
            if not isinstance(chunk, Chunk):
                raise ChunkingError(
                    f"Chunk {index} must be a Chunk object."
                )

            # 2. 检查正文是不是字符串
            if not isinstance(chunk.content, str):
                raise ChunkingError(
                    f"Chunk {index} content must be a string."
                )

            if not chunk.content.strip():
                raise ChunkingError(
                    f"Chunk {index} has empty content."
                )

        # 4. 向量化 (Embed)
        # 提取出所有文本，一次性交给 embedder 处理（批处理速度更快）
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        # 检查返回类型
        if not isinstance(embeddings, list):
            raise EmbeddingError("Embedder must return a list of vectors.")

        # 检查 Chunk 和 向量 数量
        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                f"Embedding count mismatch: expected {len(chunks)}, "
                f"got {len(embeddings)}."
            )

        # 检查非法向量
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list):
                raise EmbeddingError(f"Vector {index} must be a list.")

            if len(vector) != self.embedder.dimension:
                raise EmbeddingError(f"Vector {index} has an invalid dimension.")

            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingError(f"Vector {index} contains a non-numeric value.")

                if not math.isfinite(value):
                    raise EmbeddingError(f"Vector {index} contains NaN or infinity.")

        # 把算好的向量塞回 chunk 对象里
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 5. 存储 (Store)
        self.store.store(chunks)

        logger.info(f"✅ 文件 {path.name} 处理完毕，共入库 {len(chunks)} 个 Chunk。")
        return len(chunks)
