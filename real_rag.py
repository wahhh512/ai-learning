import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. 加载文档
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

# 2. 切块（Chunking）——这是 RAG 的灵魂
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,   # 每块最大100字
    chunk_overlap=20, # 块与块之间重叠20字，防止语义断层
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"文档被切成了 {len(chunks)} 块")

# 3. 初始化 Embedding 模型（指向你本地下载好的模型）
embeddings = HuggingFaceEmbeddings(
    model_name="./models/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

# 4. 存入向量数据库 Chroma（会自动在本地生成一个文件夹）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化目录
)

# 5. 初始化 DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_KEY")
)

# 6. 检索 + 生成
def ask(question):
    # 从向量库检索最相关的2块
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 拼接 Prompt
    prompt = f"""请严格根据以下参考资料回答问题。如果参考资料中没有答案，直接回答“我不知道”。
    
参考资料：
{context}

问题：{question}
回答："""
    
    response = llm.invoke(prompt)
    return response.content, context

if __name__ == "__main__":
    q1 = "RAG是什么？"
    answer, ctx = ask(q1)
    print(f"问题：{q1}")
    print(f"检索到的资料：\n{ctx}")
    print(f"回答：{answer}\n")
    
    q2 = "今天天气如何？"
    answer2, _ = ask(q2)
    print(f"问题：{q2}")
    print(f"回答：{answer2}")