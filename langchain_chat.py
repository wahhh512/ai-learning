from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_KEY")
)

# 自定义 Prompt 模板，加入面试官角色
template = """你是一个AI应用开发岗的面试官，负责考察候选人的Python和AI基础知识。
请用专业但友好的语气提问，每次只问一个问题，等候选人回答后再问下一个。
如果候选人回答正确，简短肯定后进入下一题；如果回答错误，指出问题并追问。

当前对话历史：
{history}

候选人：{input}
面试官："""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# 只保留最近5轮对话，防止Token爆炸
memory = ConversationBufferWindowMemory(k=5)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    reply = conversation.predict(input=req.message)
    return {"reply": reply}