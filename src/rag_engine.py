import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 基础设置 (接入智谱)
Settings.llm = OpenAILike(
    model="glm-4-flash",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key="6ff842fdbe6c41de96c26f47d392b51e.aEwG4WTn6PPUQIE7", # 你的Key
    is_chat_model=True,
    timeout=120.0
)
Settings.embed_model = HuggingFaceEmbedding(model_name="../bge-small-zh-v1.5") # 向上找一级模型

print("正在构建多库隔离索引...")
# 2. 构建三个独立索引 (路径指向 data)
index_cs = VectorStoreIndex.from_documents(SimpleDirectoryReader(input_files=["../data/doc_cs.txt"]).load_data())
index_hr = VectorStoreIndex.from_documents(SimpleDirectoryReader(input_files=["../data/doc_hr.txt"]).load_data())
index_common = VectorStoreIndex.from_documents(SimpleDirectoryReader(input_files=["../data/doc_common.txt"]).load_data())

# 3. 中文问答模板
qa_prompt_str = (
    "你是一个严谨的企业客服。请严格根据以下提供的上下文信息回答用户的问题。\n"
    "⚠️ 绝对不要添加任何文档中没有的信息，不要自己脑补步骤！\n"
    "如果上下文中没有提到，请直接回答“抱歉，我无法回答该问题”。\n"
    "请务必使用简体中文回答。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题：{query_str}\n"
    "回答："
)
qa_prompt = PromptTemplate(qa_prompt_str)

# 4. 智能中枢主控逻辑
def smart_hub(question: str):
    if "对比" in question or "区别" in question:
        print("\n>>> [兜底模式] 触发【分支模式 SubQuery】")
        return index_cs.as_query_engine(text_qa_template=qa_prompt).query(question)
    elif "年假" in question or "请假" in question:
        print("\n>>> [兜底模式] 路由到【HR政策库】")
        return index_hr.as_query_engine(text_qa_template=qa_prompt).query(question)
    elif "保修" in question or "产品" in question:
        print("\n>>> [兜底模式] 路由到【客服产品库】")
        return index_cs.as_query_engine(text_qa_template=qa_prompt).query(question)
    else:
        print("\n>>> [兜底模式] 路由到【通用信息库】")
        return index_common.as_query_engine(text_qa_template=qa_prompt).query(question)