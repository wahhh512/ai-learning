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



#定义请求体结构，FastAPI 靠这个自动校验前端传的数据
class ChatRequest(BaseModel):
    message:str


#定义post接口，路径是/chat
@app.post("/chat")
def chat(req:ChatRequest):
    response = client.chat.completions.create(
        model="deeepseek-chat",
        messages=[{"role":"useer","content":req.message}]
    )
    return {"reply": response.choices[0].message.content}