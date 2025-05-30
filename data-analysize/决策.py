import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# 设置支持中文的字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 数据加载和预处理部分
try:
    # 加载标注的部分遗言数据
    labeled_data = pd.read_csv("C:/Users/tpj/Desktop/决策树/last_words_sampled_rows.csv")
    
    # 加载完整的遗言数据
    full_data = pd.read_csv("C:/Users/tpj/Desktop/决策树/raw_last_statement.csv")
    
    print(f"已加载标注数据: {len(labeled_data)} 条记录")
    print(f"已加载完整数据: {len(full_data)} 条记录")
except Exception as e:
    print(f"数据加载错误: {e}")
    # 为了代码演示，创建模拟数据
    print("使用模拟数据进行演示...")
    np.random.seed(42)
    labeled_data = pd.DataFrame({
        'last.statement': [f"text_{i}" for i in range(100)],
        'label-penitence': np.random.uniform(0, 10, 100)
    })
    full_data = pd.DataFrame({
        'last_statement': [f"full_text_{i}" for i in range(500)],
        'other_columns': np.random.randint(0, 100, 500)
    })

# 数据预处理函数
def preprocess_text(text):
    if isinstance(text, float):
        return ''
    # 转换为小写
    text = text.lower()
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 分词
    words = text.split()
    # 词干提取
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# 对标注数据和完整数据的遗言列进行预处理
labeled_data['last.statement'] = labeled_data['last.statement'].apply(preprocess_text)
full_data['last_statement'] = full_data['last_statement'].apply(preprocess_text)

# 数据预处理，删除标注数据中的缺失值
labeled_data = labeled_data.dropna(subset=['last.statement', 'label-penitence'])

# 使用 TF - IDF 向量化文本
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(labeled_data['last.statement'])
y = labeled_data['label-penitence']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义决策树参数搜索范围
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_model.predict(X_test)

# 计算回归评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 将回归问题转化为分类问题
def map_to_category(value):
    """将悔悟程度值映射到分类等级"""
    if value < 0.5:
        return '不悔过'
    else:
        return '悔过'

# 将真实值和预测值转换为类别
y_test_categories = y_test.apply(map_to_category)
y_pred_categories = pd.Series(y_pred).apply(map_to_category)

# 计算混淆矩阵
cm = confusion_matrix(y_test_categories, y_pred_categories, labels=['不悔过', '悔过'])

# 计算分类报告
report = classification_report(y_test_categories, y_pred_categories)

# 计算准确度
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

# 对完整遗言数据进行预测
full_data['last_statement'] = full_data['last_statement'].fillna('')
full_X = vectorizer.transform(full_data['last_statement'])
full_predictions = [round(pred, 1) for pred in best_model.predict(full_X)]
full_data['predicted_penitence'] = full_predictions
full_data['predicted_category'] = pd.Series(full_predictions).apply(map_to_category)

# 保存结果
csv_path = "C:/Users/tpj/Desktop/决策树/raw_last_statement_with_predictions.csv"
full_data.to_csv(csv_path, index=False)

# 可视化部分

# 1. 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['不悔过', '悔过'], 
            yticklabels=['不悔过', '悔过'])
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.title('悔过程度预测的混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 绘制决策树（简化版，避免过深的树）
plt.figure(figsize=(20, 10))
# 限制树的深度以便于可视化
simplified_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
simplified_tree.fit(X_train, y_train)
plot_tree(simplified_tree, 
          feature_names=vectorizer.get_feature_names_out(),  
          filled=True, 
          rounded=True,  
          fontsize=10,
          max_depth=3)
plt.title('简化决策树 (最大深度=3)')
plt.tight_layout()
plt.savefig('simplified_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出结果
print('\n混淆矩阵：')
print(cm)
print('\n分类报告：')
print(report)
print(f'\n混淆矩阵计算的准确度: {accuracy:.4f}')
print(f'\n带有预测结果的完整遗言数据保存路径：{csv_path}')
print('混淆矩阵图片已保存为: confusion_matrix.png')
print('简化决策树图片已保存为: simplified_decision_tree.png')