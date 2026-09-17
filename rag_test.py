import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

# 加载本地模型
model = SentenceTransformer("./models/bge-small-zh-v1.5", local_files_only=True)

# 知识库
knowledge_base = [
    "FastAPI 是一个用 Python 写后端接口的框架",
    "LangChain 是把大模型和外部工具串起来的框架",
    "RAG 是先检索资料再让模型回答的技术",
    "Embedding 是把文字变成向量的技术",
    "Docker 是把代码和运行环境打包的工具"
]

# 预先把知识库转成向量
kb_vectors = model.encode(knowledge_base)

# 初始化 DeepSeek
client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

def rag_chat(query):
    # 1. 把用户提问转成向量
    query_vector = model.encode([query])
    
    # 2. 计算与知识库的相似度并排序
    similarities = cosine_similarity(query_vector, kb_vectors)[0]
    best_idx = np.argmax(similarities) # 取相似度最高的索引
    best_match = knowledge_base[best_idx]
    
    # 3. 把检索到的资料拼接到 Prompt 里
    prompt = f"""请严格根据以下参考资料回答问题。如果参考资料中没有答案，直接回答“我不知道”。

参考资料：{best_match}

问题：{query}
回答："""
    
    # 4. 调用大模型生成回答
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 测试
if __name__ == "__main__":
    test_query = "我想让模型先查资料再回答，用什么技术？"
    print(f"提问：{test_query}")
    print(f"回答：{rag_chat(test_query)}")
    
    test_query2 = "今天天气怎么样？" # 知识库里没有的问题
    print(f"\n提问：{test_query2}")
    print(f"回答：{rag_chat(test_query2)}")