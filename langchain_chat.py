from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

app = FastAPI()

# 初始化模型，指向 DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_KEY")
)

# 创建带记忆的对话链
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    reply = conversation.predict(input=req.message)
    return {"reply": reply}