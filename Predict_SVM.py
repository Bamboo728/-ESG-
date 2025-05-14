import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
)

# 读取数据
train_df = pd.read_excel(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\ESG_mark2.xlsx')
test_df = pd.read_excel(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\HK_ESG_mark2.xlsx')

X_train = train_df[['E', 'S', 'G']]
y_train = train_df['ESG']
X_test = test_df[['E', 'S', 'G']]
y_test_true = test_df['ESG']

from sklearn.preprocessing import OneHotEncoder

# 替代 StandardScaler 的预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['E', 'S', 'G'])
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

# 模型训练与预测
pipeline.fit(X_train, y_train)
y_test_pred = pipeline.predict(X_test)
test_df['predicted_ESG_label'] = y_test_pred

# 模型评估
accuracy = accuracy_score(y_test_true, y_test_pred)
f1_macro = f1_score(y_test_true, y_test_pred, average='macro')
f1_micro = f1_score(y_test_true, y_test_pred, average='micro')
recall = recall_score(y_test_true, y_test_pred, average='macro')  # 或者 'micro'
print(f"召回率: {recall:.4f}")
print(f"准确率: {accuracy:.4f}")
print(f"Macro F1 分数: {f1_macro:.4f}")
print(f"Micro F1 分数: {f1_micro:.4f}")
print("\n分类报告:")
print(classification_report(y_test_true, y_test_pred))

# 混淆矩阵
cm = confusion_matrix(y_test_true, y_test_pred)
labels = sorted(test_df['ESG'].unique())

# 混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('SVM 混淆矩阵 Heatmap')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 标签分布对比图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
y_test_true.value_counts().sort_index().plot(kind='bar', title='真实标签分布')
plt.subplot(1, 2, 2)
pd.Series(y_test_pred).value_counts().sort_index().plot(kind='bar', title='SVM预测标签分布')
plt.tight_layout()
plt.show()
