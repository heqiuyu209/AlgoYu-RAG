import streamlit as st
from pathlib import Path
import yaml
from dotenv import load_dotenv
from joblib.func_inspect import filter_args

from learning_os.domain import MetadataFilter

# 1. 页面基本配置
st.set_page_config(
    page_title="Learning OS - ACM 智能知识库",
    page_icon="🧠",
    layout="wide",
)

load_dotenv()

# 2. 🛡️ 绝对路径精准加载根目录的 config.yaml
def load_config() -> dict:
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if Path("config.yaml").exists():
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# 3. 插件注册表装配
from learning_os.infrastructure.container import (
    Container,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    LLM_REGISTRY,
    PARSER_REGISTRY,
    VECTOR_STORE_REGISTRY,
    CLEANER_REGISTRY,
)

from learning_os.infrastructure.chunkers.registry import CHUNKER_REGISTRY as CR
from learning_os.infrastructure.embedders.registry import EMBEDDER_REGISTRY as ER
from learning_os.infrastructure.llms.registry import LLM_REGISTRY as LR
from learning_os.infrastructure.parsers.registry import PARSER_REGISTRY as PR
from learning_os.infrastructure.vector_stores.registry import VECTOR_STORE_REGISTRY as VSR
from learning_os.infrastructure.cleaners.registry import CLEANER_REGISTRY as CLR

CLEANER_REGISTRY.update(CLR)
CHUNKER_REGISTRY.update(CR)
EMBEDDER_REGISTRY.update(ER)
LLM_REGISTRY.update(LR)
PARSER_REGISTRY.update(PR)
VECTOR_STORE_REGISTRY.update(VSR)


# 4. 初始化容器（模型只加载一次）
@st.cache_resource
def init_container():
    cfg = load_config()
    return Container(cfg)


container = init_container()
retriever = container.retriever()
llm = container.llm()

from learning_os.domain.interfaces import Message

# 5. 侧边栏：知识库控制台与检索参数
with st.sidebar:
    st.header("🛠️ 知识库控制台")
    st.info("当前底座：Qdrant + BGE-m3 + DeepSeek")

    if st.button("🔄 刷新知识库状态", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 检索过滤设置")
    filter_subject = st.text_input("Subject 过滤 (例如 ICPC)", value="")
    filter_topic = st.text_input("Topic 过滤 (例如 博弈论)", value="")

    top_k = st.slider("检索 Chunk 数量 (Top K)", min_value=1, max_value=10, value=4)

# 6. 主界面：RAG 智能对话区
st.title("🧠 Learning OS —— 算法与大模型知识库")
st.caption("基于本地 RAG 与大模型，专为 ACM / 算法竞赛与大创项目打造的智能学伴。")

# 初始化聊天历史状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. 用户输入与 RAG 核心处理逻辑
if prompt := st.chat_input("输入你想问的问题（例如：什么是博弈论中的必胜态与必败态？）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    subject = filter_subject.strip() or None
    topic = filter_topic.strip() or None

    if subject or topic:
        filter_arg = MetadataFilter(
            subject = subject,
            topic = topic,
        )
    else :
        filter_arg = None

    with st.spinner("正在知识库中检索相关讲义..."):
        try:
            chunks = retriever.retrieve(query=prompt, top_k=top_k, filter_=filter_arg)
        except Exception as exc:
            chunks = []
            st.error(f"检索出错: {exc}")


    def get_chunk_source(c) -> str:
        if hasattr(c, "chunk"):
            c = c.chunk

        for attr in ["metadata", "meta", "source", "file_path"]:
            val = getattr(c, attr, None)
            if isinstance(val, dict):
                for k in ["source", "file_path", "path"]:
                    if k in val:
                        return val[k]
            elif isinstance(val, str) and val:
                return val

        return "未知来源"


    def get_chunk_content(c) -> str:
        if hasattr(c, "chunk"):
            c = c.chunk

        for attr in ["content", "text", "page_content", "body", "document"]:
            val = getattr(c, attr, None)
            if isinstance(val, str) and val:
                return val

        raise ValueError(f"无法从 {type(c).__name__} 获取文本内容")


    # 组装上下文
    context_text = "\n\n---\n\n".join([f"[来源: {get_chunk_source(c)}]\n{get_chunk_content(c)}" for c in chunks])

    # 构造 System Prompt
    system_prompt = (
        "你是一个专业的 ACM 竞赛教练和软件工程学长。请根据以下提供的【知识库参考上下文】，"
        "准确, 清晰地回答用户的问题。如果上下文里没有答案，请结合你的算法专业知识补充，但要说明。"
    )
    user_prompt = f"【知识库参考上下文】：\n{context_text}\n\n【用户问题】：\n{prompt}"

    messages_to_send = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt)
    ]

    # 第二步：大模型生成与流式展示
    with st.chat_message("assistant"):
        # 折叠展示召回的 Chunk 溯源
        with st.expander(f"📚 查看本次 RAG 召回的参考文档 ({len(chunks)} 个 Chunk)", expanded=False):
            for i, c in enumerate(chunks, 1):
                score = getattr(c, "score", 0.0)
                src = get_chunk_source(c)
                preview = get_chunk_content(c)
                st.markdown(f"**#{i} 相似度得分**: `{score:.4f}` | **来源**: `{src}`")
                st.text(preview[:300] + "...")

        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = llm.stream_chat(messages=messages_to_send)
            for chunk_str in stream:
                full_response += chunk_str
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as exc:
            full_response = f"❌ 大模型调用失败: {exc}"
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})