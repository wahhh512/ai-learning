from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

app = FastAPI()

# 全局初始化：只在服务器启动时执行一次，避免每次请求都加载（极其重要！）
print("正在加载模型和向量库，请稍候...")
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="./models/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

class ChatRequest(BaseModel):
    message: str

@app.post("/rag_chat")
def rag_chat(req: ChatRequest):
    # 1. 检索
    docs = vectorstore.similarity_search(req.message, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. 拼装 Prompt
    prompt = f"""请严格根据以下参考资料回答问题。如果参考资料中没有答案，直接回答“我不知道”。
    
参考资料：
{context}

问题：{req.message}
回答："""
    
    # 3. 生成
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "reply": response.choices[0].message.content,
        "retrieved_context": context
    }