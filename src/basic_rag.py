import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 为了国内下载模型提速，设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 加载 PDF 并切分 (chunk_size=500)
print("正在读取并切分 PDF...")
loader = PyPDFLoader("薛之谦信息.pdf")
docs = loader.load()

# 重点检查这里的括号是否完整
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"PDF 被切分成了 {len(splits)} 个片段。")

# 2. 向量化并存入 ChromaDB
print("正在下载 Embedding 模型并存入数据库 (首次运行需下载模型)...")
embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

# 3. 构建 Prompt 和 LLM 链路
llm = ChatOllama(
    model="qwen-medical", 
    temperature=0.1,
    num_predict=512,        
    repeat_penalty=1.2,

)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的客服。请严格根据资料回答问题，绝对不要编造。如果资料中没有，请直接说不知道。"),
    ("human", "示例资料：产品保修期为1年。\n示例问题：保修期多久？\n示例回答：您好，关于您的问题，根据公司规定：产品保修期为1年。如有疑问请致电客服，祝您生活愉快！"),
    ("human", "资料：\n{context}\n\n问题：{question}")
])

# 4. 串联整个链路
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 测试运行
if __name__ == "__main__":
    query = "万兽之王演唱会首站在哪里？"
    print(f"\n用户提问：{query}")
    print("正在思考中...\n")
    
    # 流式输出结果
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    print("\n")