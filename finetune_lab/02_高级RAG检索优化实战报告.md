# Advanced RAG 检索优化实战 (LlamaIndex)

为了突破 Naive RAG 的召回瓶颈，本项目在底层链路中摒弃了基础切分，全面引入了 LlamaIndex 框架的高级检索技术：

## 1. 核心技术实现
*   **Sentence Window (句子窗口索引)**：采用 `SentenceWindowNodeParser` (window_size=3)。利用单句进行高精度向量匹配，命中后通过 `MetadataReplacementPostProcessor` 动态替换为包含前后文的滑动窗口，完美平衡了“检索精度”与“上下文完整性”。
*   **HyDE (假设文档嵌入)**：针对用户口语化提问导致向量距离过远的问题，在检索前让 LLM 生成一个“假设性答案”，用该答案的向量去检索真实文档，大幅提升了长尾查询的召回率。
*   **Sub-Query (子查询引擎尝试)**：尝试利用 `SubQuestionQueryEngine` 自动拆解复杂任务，测试小模型在 Agent 路由上的能力边界。

## 2. 实战效果与踩坑复盘 (核心产出)

在真实测试中，我们验证了高级 RAG 的强大检索能力，同时也撞出了 0.5B 小模型的能力天花板。

### 🎯 成功案例：HyDE + Sentence Window 精准召回
*(此处插入你最后跑出“广州”的那张截图)*
*   **测试问题**：“薛之谦今年万兽之王演唱会首站定在哪？”
*   **结果分析**：引入了 HyDE 和句子窗口后，系统成功越过了长文本干扰，精准从 PDF 中检索到了“**首站定于广州**”的核心事实。

### 🚨 踩坑反思：小模型在高级 RAG 中的“三大崩塌”
*(此处插入 SubQuery 报错 `No valid JSON found` 的那张截图)*

1.  **JSON 格式化崩塌**：在测试 `SubQuestionQueryEngine` 时，0.5B 模型无法遵循框架底层的 JSON 格式要求，直接输出了大段的幻觉文本（甚至抄了源码里的 `uber_10k` 示例），导致 `ValueError`。
2.  **指令注意力丢失**：在 HyDE 和句子窗口同时介入时，模型被过长的 Context 干扰，完全丢失了原本 LoRA 微调学到的“企业客服标准话术”。
3.  **过度解释（Over-explaining）**：虽然找出了正确答案，但 0.5B 模型无法直接输出结论，而是陷入了冗长的伪推理过程，导致输出废话连篇。

## 3. 生产环境架构演进建议
本次实战深刻证明了：**高级 RAG 极度吃底层 LLM 的 Reasoning（逻辑推理）能力**。
在真实的企业级落地中：
*   **路由/拆解/生成端**：必须升级至 7B 以上模型（如 Qwen2-7B-Instruct），才能同时兼顾长上下文理解、JSON 格式输出与特定业务话术。
*   **工程兜底**：若算力受限只能使用 0.5B 小模型，必须在 FastAPI 后端加入正则表达式或后处理拦截，强制过滤废话并套用固定回复模板。

---

## 📂 最终项目代码结构
```text
capstone_project/
├── advanced_rag.py         # LlamaIndex 高级 RAG 实现（HyDE & 句子窗口）
├── api_server.py           # FastAPI 流式接口服务
├── basic_rag.py            # 基础 LangChain 版本
├── 薛之谦信息.pdf          # RAG 知识库源文件
├── bge-small-zh-v1.5/      # Embedding 向量模型文件夹
├── bge-reranker-base/      # Rerank 重排序模型文件夹
├── 01_微调与部署报告.md    # 微调篇作业报告
└── 02_Hybrid_RAG_System.md # 高阶 RAG 篇作业报告（本文档）