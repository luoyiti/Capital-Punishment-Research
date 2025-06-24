import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns



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

# 将回归问题转化为分类问题
def map_to_category(value):
    """将悔悟程度值映射到分类等级"""
    if value <= 0.5:
        return 'Non - penitent'
    else:
        return 'Penitent'

y = labeled_data['label-penitence'].apply(map_to_category)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 定义决策树参数搜索范围
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ID3算法（使用信息增益作为划分标准）
id3_grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid, cv=5)
id3_grid_search.fit(X_train, y_train)
id3_best_model = id3_grid_search.best_estimator_
id3_y_pred = id3_best_model.predict(X_test)
id3_cm = confusion_matrix(y_test, id3_y_pred, labels=['Non - penitent', 'Penitent'])
id3_report = classification_report(y_test, id3_y_pred)
id3_accuracy = (id3_cm[0, 0] + id3_cm[1, 1]) / id3_cm.sum()

# C4.5算法（使用信息增益比作为划分标准）
# 在scikit-learn中没有直接的信息增益比选项，但可以通过调整其他参数模拟
c45_grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42), param_grid, cv=5)
c45_grid_search.fit(X_train, y_train)
c45_best_model = c45_grid_search.best_estimator_
c45_y_pred = c45_best_model.predict(X_test)
c45_cm = confusion_matrix(y_test, c45_y_pred, labels=['Non - penitent', 'Penitent'])
c45_report = classification_report(y_test, c45_y_pred)
c45_accuracy = (c45_cm[0, 0] + c45_cm[1, 1]) / c45_cm.sum()

# CART算法（用于分类）
cart_grid_search = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=42), param_grid, cv=5)
cart_grid_search.fit(X_train, y_train)
cart_best_model = cart_grid_search.best_estimator_
cart_y_pred = cart_best_model.predict(X_test)
cart_cm = confusion_matrix(y_test, cart_y_pred, labels=['Non - penitent', 'Penitent'])
cart_report = classification_report(y_test, cart_y_pred)
cart_accuracy = (cart_cm[0, 0] + cart_cm[1, 1]) / cart_cm.sum()

# 对完整遗言数据进行预测
full_data['last_statement'] = full_data['last_statement'].fillna('')
full_X = vectorizer.transform(full_data['last_statement'])

id3_full_predictions = id3_best_model.predict(full_X)
full_data['id3_predicted_category'] = id3_full_predictions

c45_full_predictions = c45_best_model.predict(full_X)
full_data['c45_predicted_category'] = c45_full_predictions

cart_full_predictions = cart_best_model.predict(full_X)
full_data['cart_predicted_category'] = cart_full_predictions

# 保存结果
csv_path = "C:/Users/tpj/Desktop/决策树/raw_last_statement_with_predictions.csv"
full_data.to_csv(csv_path, index=False)

# 可视化部分

# 1. 绘制ID3混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(id3_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non - penitent', 'Penitent'],
            yticklabels=['Non - penitent', 'Penitent'])
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix of ID3 Algorithm for Penitence Prediction')
plt.tight_layout()
plt.savefig('id3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 绘制C4.5混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(c45_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non - penitent', 'Penitent'],
            yticklabels=['Non - penitent', 'Penitent'])
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix of C4.5 Algorithm for Penitence Prediction')
plt.tight_layout()
plt.savefig('c45_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 绘制CART混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cart_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non - penitent', 'Penitent'],
            yticklabels=['Non - penitent', 'Penitent'])
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix of CART Algorithm for Penitence Prediction')
plt.tight_layout()
plt.savefig('cart_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 绘制ID3决策树
plt.figure(figsize=(20, 10))
id3_simplified_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
id3_simplified_tree.fit(X_train, y_train)
plot_tree(id3_simplified_tree,
          feature_names=vectorizer.get_feature_names_out(),
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)
plt.title('Simplified ID3 Decision Tree (Max Depth = 3)')
plt.tight_layout()
plt.savefig('id3_simplified_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 绘制C4.5决策树
plt.figure(figsize=(20, 10))
c45_simplified_tree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state=42)
c45_simplified_tree.fit(X_train, y_train)
plot_tree(c45_simplified_tree,
          feature_names=vectorizer.get_feature_names_out(),
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)
plt.title('Simplified C4.5 Decision Tree (Max Depth = 3)')
plt.tight_layout()
plt.savefig('c45_simplified_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 绘制CART决策树
plt.figure(figsize=(20, 10))
cart_simplified_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
cart_simplified_tree.fit(X_train, y_train)
plot_tree(cart_simplified_tree,
          feature_names=vectorizer.get_feature_names_out(),
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)
plt.title('Simplified CART Decision Tree (Max Depth = 3)')
plt.tight_layout()
plt.savefig('cart_simplified_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出结果
print('\nID3算法混淆矩阵：')
print(id3_cm)
print('\nID3算法分类报告：')
print(id3_report)
print(f'\nID3算法混淆矩阵计算的准确度: {id3_accuracy:.4f}')

print('\nC4.5算法混淆矩阵：')
print(c45_cm)
print('\nC4.5算法分类报告：')
print(c45_report)
print(f'\nC4.5算法混淆矩阵计算的准确度: {c45_accuracy:.4f}')

print('\nCART算法混淆矩阵：')
print(cart_cm)
print('\nCART算法分类报告：')
print(cart_report)
print(f'\nCART算法混淆矩阵计算的准确度: {cart_accuracy:.4f}')

print(f'\n带有预测结果的完整遗言数据保存路径：{csv_path}')
print('ID3算法混淆矩阵图片已保存为: id3_confusion_matrix.png')
print('C4.5算法混淆矩阵图片已保存为: c45_confusion_matrix.png')
print('CART算法混淆矩阵图片已保存为: cart_confusion_matrix.png')
print('ID3简化决策树图片已保存为: id3_simplified_decision_tree.png')
print('C4.5简化决策树图片已保存为: c45_simplified_decision_tree.png')
print('CART简化决策树图片已保存为: cart_simplified_decision_tree.png')