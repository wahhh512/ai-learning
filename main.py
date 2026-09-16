from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

# 用一个简单的全局列表存对话历史（新手演示用，生产环境用数据库）
chat_history = []

# 初始化系统提示词，约束AI的角色
chat_history.append({"role": "system", "content": "你是一个AI应用开发岗的面试官，负责考察候选人的Python和AI基础知识。请用专业但友好的语气提问，每次只问一个问题，等候选人回答后再问下一个。"})

# 定义请求体，现在只传用户的新问题
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. 把用户的新问题加入历史记录
    chat_history.append({"role": "user", "content": req.message})
    
    # 2. 调用模型时，把整个历史记录传进去
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=chat_history
    )
    
    # 3. 把模型的回答也加入历史记录
    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    
    return {"reply": reply, "history_length": len(chat_history)}