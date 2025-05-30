import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 正常显示中文，设置字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']
# 正常显示中文，设置字体为 WenQuanYi Zen Hei
#plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 明确负号显示
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv("D:\\HuaweiMoveData\\Users\\32549\\OneDrive\\twitter_capital_data.csv")

# 使用 TextBlob 进行情感分析
def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 应用函数到 description 列，获取情感得分
df['sentiment_score'] = df['description'].apply(get_sentiment_score)

# 定义一个函数来对情感得分进行分类
def get_sentiment_label(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# 应用函数到 sentiment_score 列，获取情感标签
df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

# 查看总体情感趋向分布情况
sentiment_distribution = df['sentiment_label'].value_counts(normalize=True) * 100

# 输出结果（保留两位小数）
print(sentiment_distribution.round(2))

# 将 datetime 列转换为日期时间类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 按年月统计不同情感的数量
time_series_data = df.groupby([df['datetime'].dt.to_period('M'), 'sentiment_label']).size().unstack(fill_value=0)


# 绘制时间序列图


ax = time_series_data.plot(title='死刑舆情随时间的变化趋势')
# 设置时间序列图横纵坐标名称字号为 6
ax.set_xlabel('时间', fontsize=6)
ax.set_ylabel('数量', fontsize=6)
ax.legend(title='情感倾向')


# 绘制情感得分直方图
plt.figure(figsize=(15, 6))
plt.hist(df['sentiment_score'], bins=30, edgecolor='black')
plt.title('情感得分分布直方图')
plt.xlabel('情感得分', fontsize=6)
plt.xticks(fontsize=6)
plt.ylabel('频数', fontsize=6)

plt.show()