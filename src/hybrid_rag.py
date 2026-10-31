import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llmlingua import PromptCompressor

# 1. 基础设置
# ⚠️ 如果你的本地模型叫 cs-assistant-v1，请把下面的名字换掉
Settings.llm = Ollama(model="qwen-medical", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="./bge-small-zh-v1.5")

print("正在加载文档并构建索引...")
# 读取 PDF 和我们刚才新建的 TXT 文件
docs = SimpleDirectoryReader(input_files=["薛之谦信息.pdf", "extra_info.txt"]).load_data()
index = VectorStoreIndex.from_documents(docs)

# 2. 构建双路检索器
print("正在初始化双路检索器 (Vector + BM25)...")
# 路1：向量检索（基于语义，容易忽略具体型号）
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
# 路2：BM25 检索（基于关键词精确匹配，专治各种型号和人名）
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2)

# 3. 核心：使用 RRF（倒数排名融合）将两路结果合并
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=1,  # 不自动生成额外问题，只做召回融合
    mode="reciprocal_rerank", # 使用 RRF 算法
    use_async=False
)

# 初始化 LLMLingua 压缩器
print("正在加载 LLMLingua 压缩模型 (首次运行可能需要下载，请耐心等待)...")
# 使用微软亚洲研究院开源的多语言小模型
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", # 改成这个官方模型
    use_llmlingua2=True,
    device_map="cpu"
)
if __name__ == "__main__":
    test_query = "型号 XR-2000 是什么？它有什么用？"
    
    print("\n" + "="*50)
    print(f"🚀 混合检索测试问题: {test_query}")
    print("="*50)
    
    # 1. 执行混合检索
    nodes = fusion_retriever.retrieve(test_query)
    
    # 将检索到的多段文本拼起来
    context_text = "\n\n".join([node.text for node in nodes])
    print(f"\n📄 原始检索结果拼接 (字符数: {len(context_text)}):")
    print("-" * 30)
    print(context_text)
    
    # 2. 执行 LLMLingua 压缩
    print("\n🗜️ 正在执行 LLMLingua 提示词压缩...")
    compressed_result = compressor.compress_prompt(
        context=context_text,
        instruction=test_query,
        target_token=150,  # 限制压缩到 150 tokens 以内
    )
    
    compressed_text = compressed_result['compressed_prompt']
    print(f"\n✨ 压缩后结果 (字符数: {len(compressed_text)}):")
    print("-" * 30)
    print(compressed_text)
    
    # 3. 将压缩后的文本喂给大模型
    print("\n🤖 正在使用本地微调模型生成最终回答...")
    prompt = f"请根据以下信息回答问题。\n信息：{compressed_text}\n问题：{test_query}\n回答："
    response = Settings.llm.complete(prompt)
    
    print("\n🎯 最终模型回答:")
    print("-" * 30)
    print(response.text)