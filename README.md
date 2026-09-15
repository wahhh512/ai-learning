# AI Learning

基于 FastAPI 和 DeepSeek 大模型的对话接口练习项目。

## 技术栈
- Python 3.11
- FastAPI
- OpenAI SDK (兼容 DeepSeek API)
- python-dotenv

## 功能
- 提供 `/chat` POST 接口
- 请求体：`{"message": "你的问题"}`
- 返回模型回答的 JSON

## 如何运行
1. 安装依赖：`pip install fastapi uvicorn openai python-dotenv`
2. 在根目录创建 `.env`，写入 `DEEPSEEK_KEY=你的Key`
3. 启动：`uvicorn main:app --reload`
4. 浏览器打开 `http://127.0.0.1:8000/docs`
