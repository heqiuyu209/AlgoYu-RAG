import logging
from pathlib import Path
from learning_os.domain.interfaces import Parser, Chunker, Embedder, VectorStore, Cleaner

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    文档入库流水线（Application Service）。
    负责编排解析、清洗、分块、向量化和存储的完整流程。
    """

    def __init__(
            self,
            parser: Parser,
            chunker: Chunker,
            embedder: Embedder,
            vector_store: VectorStore,
            cleaner: Cleaner | None = None
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = vector_store
        self.cleaner = cleaner

    def run(self, file_path: Path | str) -> int:
        """
        执行入库流程。返回成功入库的 Chunk 数量。
        """
        path = Path(file_path)
        logger.info(f"🚀 开始处理文件: {path.name}")

        # 1. 解析 (Parse)
        document = self.parser.parse(path)

        # 2. 清洗 (Clean) - 如果配置了 Cleaner，就洗一下
        if self.cleaner:
            document.content = self.cleaner.clean(document.content)

        # 3. 分块 (Chunk)
        chunks = self.chunker.chunk(document)
        if not chunks:
            logger.warning("未生成任何文本块。")
            return 0

        # 4. 向量化 (Embed)
        # 提取出所有文本，一次性交给 embedder 处理（批处理速度更快）
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        # 把算好的向量塞回 chunk 对象里
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 5. 存储 (Store)
        self.store.store(chunks)

        logger.info(f"✅ 文件 {path.name} 处理完毕，共入库 {len(chunks)} 个 Chunk。")
        return len(chunks)