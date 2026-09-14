岗位职责
1. 参与广告投放 Agent 研发与迭代，开发业务能力
2. 设计 Prompt、上下文组织及工具调用逻辑
3. 开发数据查询、API、投放分析等业务工具
4. 分析线上 Bad Case，持续优化 Agent 效果
5. 参与 RAG、Structured Output、模型调用等能力开发
任职要求
1. 计算机相关专业在读，本科及以上
2. 熟悉 Python / Java / TypeScript / Go 之一
3. 了解 Prompt、Tool Calling、RAG、Structured Output 等 LLM 基础能力
4. 有 LangChain、LangGraph 等框架开发经验优先
加分项
1. 独立开发过 AI Agent 项目
2. 有 Agent 评测、Bad Case 分析或效果优化经验
3. 深度使用过 Claude Code、Cursor 等 AI Coding Agent


FastAPI	用 Python 写 API 接口的框架。你写一个函数，别人通过网址就能调用你的 AI 功能。自带交互式文档，浏览器打开 /docs 就能测接口。	刚见网页长什么样
LangChain	把大模型和外部工具（数据库、API、文件）串起来的“胶水框架”。大模型是大脑，LangChain 是手脚，让大脑能真的去查资料、调接口。	看过很多也跟着做过一个简单的
RAG	先翻书再回答。模型遇到问题，先去你的资料库里检索相关段落，把检索结果和问题一起喂给模型，让它基于真实资料回答，减少胡编。	检索
LangGraph	LangChain 的升级版，用“流程图”的方式做多步骤 Agent。节点是步骤，箭头是流转方向，适合做需要反复判断、循环重试的复杂任务。	流程图
Function Calling	模型学会“点单”。你提前定义好函数（比如查天气、查库存），模型判断该用哪个，返回结构化参数告诉你去执行。	听过
Embedding	把文字变成一串数字（向量），意思相近的文字，数字串也相近。RAG 的检索就是靠这个“算相似度”实现的。	听过
Re-ranking	检索的“二轮筛选”。第一轮粗搜出 100 条，再用一个更精确的模型重新打分排序，选出最相关的 5 条给大模型。	没听过
LoRA / QLoRA	低成本微调大模型的方法。不改动原模型全部参数，只训练一小块“补丁”，QLoRA 比 LoRA 更省显存。	没听过
Docker	把代码和运行环境打包成一个“集装箱”。换电脑也能一样跑，不用再折腾“我电脑上明明可以”。	

