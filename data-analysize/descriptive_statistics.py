import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re

# 加载数据
df = pd.read_csv("C:/Users/tpj/Desktop/Capital-Punishment-Research-main (1)/twitter_capital_data.csv")

# 自定义停用词列表
stop_words = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'with', 'as', 'for', 'on', 
    'is', 'are', 'be', 'was', 'were', 'by', 'at', 'this', 'from', 'or', 'an', 'have', 
    'not', 'but', 'has', 'had', 'its', 'their', 'they', 'we', 'you', 'i', 'he', 'she', 
    'his', 'her', 'him', 'our', 'your', 'all', 'any', 'no', 'will', 'would', 'can', 
    'could', 'may', 'might', 'should', 'these', 'those', 'am', 'been', 'being', 'do', 
    'does', 'did', 'so', 'such', 'than', 'then', 'there', 'here', 'when', 'where', 
    'why', 'how', 'what', 'which', 'who', 'whom', 'into', 'about', 'between', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 
    'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
    'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 
    'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

# 文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 分词并去除停用词
    tokens = [word for word in text.split() if word not in stop_words]
    # 简单词干化（去除复数形式）
    tokens = [word.rstrip('s') for word in tokens]
    return tokens

# 合并description和title列的文本
combined_text = df['description'] + ' ' + df['title']

# 预处理所有文本
processed_text = combined_text.apply(preprocess_text)

# 将处理后的文本转换回字符串（用于TfidfVectorizer）
corpus = [' '.join(tokens) for tokens in processed_text]

# 与死刑话题相关的词汇列表
death_penalty_terms = ['death', 'penalty', 'capital', 'punishment', 'execute', 'abolish', 'convict', 'sentence']

# 统计词频
all_tokens = [token for sublist in processed_text for token in sublist]
word_freq = Counter(all_tokens)

# 筛选出与死刑话题相关的词频
death_penalty_freq = {term: freq for term, freq in word_freq.items() 
                     if any(t in term for t in death_penalty_terms)}

# 按词频降序排序
sorted_death_penalty_freq = sorted(death_penalty_freq.items(), key=lambda x: x[1], reverse=True)

# 输出死刑相关词频结果
print("与死刑话题相关的词频统计（降序）：")
for term, freq in sorted_death_penalty_freq[:20]:  # 只显示前20个高频词
    print(f"{term}: {freq}")

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer(
    max_df=0.95,    # 忽略在超过95%文档中出现的词
    min_df=2,       # 忽略在少于2篇文档中出现的词
    max_features=1000,  # 保留最高频的1000个词
    stop_words='english'  # 使用sklearn内置的英文停用词
)

tfidf_matrix = vectorizer.fit_transform(corpus)

# 获取特征名称（词汇表）
feature_names = vectorizer.get_feature_names_out()

# 训练LDA模型进行主题建模
lda = LatentDirichletAllocation(
    n_components=5,  # 主题数量
    max_iter=10,     # 最大迭代次数
    learning_method='online',
    random_state=42
)

lda_model = lda.fit(tfidf_matrix)

# 输出每个主题的前10个关键词
print("\n抽取的中心话题（每个主题前10个关键词）：")
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]  # 获取前10个关键词的索引
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")