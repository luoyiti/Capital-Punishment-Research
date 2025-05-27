import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import re
from nltk.stem import PorterStemmer

# 加载标注的部分遗言数据
labeled_data = pd.read_csv("C:/Users/tpj/Desktop/决策树/last_words_sampled_rows.csv")

# 加载完整的遗言数据
full_data = pd.read_csv("C:/Users/tpj/Desktop/决策树/raw_last_statement.csv")

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

# 使用 GridSearchCV 寻找最佳参数
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_model.predict(X_test)

# 计算均方误差评估模型性能
mse = mean_squared_error(y_test, y_pred)

# 对完整遗言数据中的 last_statement 列进行缺失值处理，填充为空字符串
full_data['last_statement'] = full_data['last_statement'].fillna('')

# 使用正确的列名进行向量化
full_X = vectorizer.transform(full_data['last_statement'])

# 进行预测并保留一位小数
full_predictions = [round(pred, 1) for pred in best_model.predict(full_X)]

# 将预测结果添加到完整的遗言数据中
full_data['predicted_penitence'] = full_predictions

# 将结果保存为 CSV 文件
csv_path = "C:/Users/tpj/Desktop/决策树/raw_last_statement_with_predictions.csv"
full_data.to_csv(csv_path, index=False)

# 输出均方误差、最佳参数和保存路径
print(f'均方误差：{mse}')
print(f'最佳参数：{grid_search.best_params_}')
print(f'带有预测结果的完整遗言数据保存路径：{csv_path}')