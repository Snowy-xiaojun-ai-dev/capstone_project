import gradio as gr
import sys
import os

# 将项目根目录加入系统路径，以便导入 src 里的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag_engine import smart_hub

def chat_interface(message, history):
    try:
        response = smart_hub(message)
        return str(response)
    except Exception as e:
        return f"⚠️ 系统执行失败，真实错误信息如下：\n\n{str(e)}"

if __name__ == "__main__":
    print("\n正在启动 Web UI 界面...")
    demo = gr.ChatInterface(
        fn=chat_interface,
        title="企业级智能中枢 (Modular RAG)",
        description="支持智能路由与复杂问题分支拆解"
    )
    demo.launch()