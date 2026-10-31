from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# 从你跑通的 basic_rag 里导入链路
from basic_rag import rag_chain

app = FastAPI(title="企业级智能客服 API")

# 定义接收数据的格式
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 使用异步流式生成器，返回打字机效果
    async def generate_response():
        async for chunk in rag_chain.astream(request.question):
            yield chunk
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)