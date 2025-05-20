import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from textblob import TextBlob

# 文件读取地址
file_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大二下\数据科学与数据分析\小组作业\死刑和遗言数据.csv"

# 加载数据
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='gbk')

# 去除缺失值
data = data.dropna(subset=['last statement'])

# 加载停用词
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]

# 更完善的文本预处理函数
def preprocess_text(text):
    # 去除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = jieba.lcut(text)
    # 词性标注，保留形容词、副词等可能体现情感的词性
    import jieba.posseg as pseg
    words = [word for word, flag in pseg.lcut(text) if flag in ['a', 'ad', 'd'] or word not in stopwords]
    return " ".join(words)

# 对遗言进行预处理
data['last statement_preprocessed'] = data['last statement'].apply(preprocess_text)

# 使用 TF-IDF 进行文本向量化
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data['last statement_preprocessed'])

# 增加情感分析特征
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

data['sentiment_score'] = data['last statement_preprocessed'].apply(get_sentiment)
X_sentiment = data['sentiment_score'].values.reshape(-1, 1)

# 合并特征
from scipy.sparse import hstack
X = hstack([X_tfidf, X_sentiment])

# 寻找最优聚类数
silhouette_scores = []
for n_cluster in range(2, 10):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

best_n_cluster = silhouette_scores.index(max(silhouette_scores)) + 2

# 使用最优聚类数进行 K-Means 聚类
kmeans = KMeans(n_clusters=best_n_cluster, random_state=42)
data['cluster_label'] = kmeans.fit_predict(X)

# 分析每个聚类的特征，结合其他信息构建用户画像
for cluster in range(best_n_cluster):
    cluster_data = data[data['cluster_label'] == cluster]
    print(f"聚类 {cluster} 的用户画像：")
    print(f"平均年龄：{cluster_data['age'].mean():.2f}")
    print(f"主要种族：{cluster_data['race'].value_counts().idxmax()}")
    print(f"主要性别：{cluster_data['gender'].value_counts().idxmax()}")
    print(f"主要教育程度：{cluster_data['education level'].value_counts().idxmax()}")
    print(f"主要职业：{cluster_data['prior occupation'].value_counts().idxmax()}")
    print(f"有监狱记录比例：{cluster_data['prior prison record(0/1)'].mean():.2%}")
    print(f"平均情感得分：{cluster_data['sentiment_score'].mean():.2f}")
    print("典型遗言示例：")
    print(cluster_data['last statement'].iloc[0])
    print("-" * 50)