import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# 文件读取地址
file_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大二下\数据科学与数据分析\小组作业\死刑和遗言数据.csv"

# 加载数据
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='gbk')

# 去除缺失值
data = data.dropna(subset=['last statement'])

# 文本预处理函数
def preprocess_text(text):
    # 去除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = jieba.lcut(text)
    # 假设没有停用词表，这里简单去除单个字符的词
    words = [word for word in words if len(word) > 1]
    return " ".join(words)

# 对遗言进行预处理
data['last statement_preprocessed'] = data['last statement'].apply(preprocess_text)

# 使用 TF - IDF 进行文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['last statement_preprocessed'])

# 使用 K - Means 进行聚类，这里假设分为 5 类，可根据实际情况调整
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster_label'] = kmeans.fit_predict(X)

# 分析每个聚类的特征，结合其他信息构建用户画像
for cluster in range(num_clusters):
    cluster_data = data[data['cluster_label'] == cluster]
    print(f"聚类 {cluster} 的用户画像：")
    print(f"平均年龄：{cluster_data['age'].mean():.2f}")
    print(f"主要种族：{cluster_data['race'].value_counts().idxmax()}")
    print(f"主要性别：{cluster_data['gender'].value_counts().idxmax()}")
    print(f"主要教育程度：{cluster_data['education level'].value_counts().idxmax()}")
    print(f"主要职业：{cluster_data['prior occupation'].value_counts().idxmax()}")
    print(f"有监狱记录比例：{cluster_data['prior prison record(0/1)'].mean():.2%}")
    print("典型遗言示例：")
    print(cluster_data['last statement'].iloc[0])
    print("-" * 50)