import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载中文效果较好的轻量模型
model = SentenceTransformer("./models/bge-small-zh-v1.5", local_files_only=True)
# 准备一个小知识库
knowledge_base = [
    "FastAPI 是一个用 Python 写后端接口的框架",
    "LangChain 是把大模型和外部工具串起来的框架",
    "RAG 是先检索资料再让模型回答的技术",
    "Embedding 是把文字变成向量的技术",
    "Docker 是把代码和运行环境打包的工具"
]

# 把所有知识库句子转成向量
kb_vectors = model.encode(knowledge_base)
print(f"知识库向量形状: {kb_vectors.shape}")
# 输出应为 (5, 512)，5个句子，每个512维向量

# 模拟用户提问
query = "我想让模型先查资料再回答，用什么技术？"
query_vector = model.encode([query])

# 计算提问和每条知识库的相似度
similarities = cosine_similarity(query_vector, kb_vectors)[0]

# 排序，找出最相似的
for i in np.argsort(similarities)[::-1]:
    print(f"相似度: {similarities[i]:.4f}  ->  {knowledge_base[i]}")