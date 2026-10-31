import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# 1. 基础设置：接入本地 Ollama 和 BGE 向量模型
# ⚠️ 注意：如果你在 Ollama 里把模型叫 cs-assistant-v1，请把下面的名字改掉！
Settings.llm = Ollama(model="qwen-medical", request_timeout=120.0) 
Settings.embed_model = HuggingFaceEmbedding(model_name="./bge-small-zh-v1.5")

print("正在加载 PDF 并构建高级索引...")
# 2. 核心：Sentence Window 句子窗口解析器
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
# ⚠️ 确保 PDF 文件在同一个目录下
docs = SimpleDirectoryReader(input_files=["薛之谦信息.pdf"]).load_data()
nodes = node_parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)

# 3. 创建基础查询引擎，并注入句子窗口后处理器
base_query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

# 4. 引入 HyDE (假设文档嵌入)
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(base_query_engine, hyde)

# 5. 引入 Sub-Query (子查询引擎)
query_engine_tools = [
    QueryEngineTool(
        query_engine=hyde_query_engine,
        metadata=ToolMetadata(
            name="XueZhiQian_KB",
            description="薛之谦的生日、粉丝名、演唱会行程等详细信息",
        ),
    ),
]
sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False 
)

# ================= 测试运行 =================
if __name__ == "__main__":
    # 注释掉原来的复杂问题，换成针对性问题
    test_question = "薛之谦今年万兽之王演唱会首站定在哪？"
    
    print("\n" + "="*50)
    print(f"🚀 发起高级 RAG 提问: {test_question}")
    print("="*50)
    
    # 不要用 sub_query_engine 了，直接用带 HyDE 和句子窗口的引擎
    response = hyde_query_engine.query(test_question)
    print(response)