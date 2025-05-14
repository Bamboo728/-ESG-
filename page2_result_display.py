import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 设置字体避免中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 ESG 多模型预测结果对比")

@st.cache_data
def load_data():
    train_df = pd.read_excel(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\ESG_mark2.xlsx')
    test_df = pd.read_excel(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\HK_ESG_mark2.xlsx')
    return train_df, test_df

train_df, test_df = load_data()
X_train = train_df[['E', 'S', 'G']]
y_train = train_df['ESG']
X_test = test_df[['E', 'S', 'G']]
y_test = test_df['ESG']

def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['E', 'S', 'G'])]
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale')
}

results = {}

for name, model in models.items():
    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        "y_pred": y_pred,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall": recall,
        "confusion_matrix": cm
    }

# 展示结果对比表格
st.subheader("📈 指标对比")
metric_df = pd.DataFrame({
    model: {
        "准确率": f"{res['accuracy']:.4f}",
        "召回率": f"{res['recall']:.4f}",
        "Macro F1": f"{res['f1_macro']:.4f}"
    }
    for model, res in results.items()
}).T
st.dataframe(metric_df)

# 混淆矩阵可视化对比
st.subheader("📌 混淆矩阵对比")
cols = st.columns(len(results))
labels = sorted(y_test.unique())

for i, (model_name, res) in enumerate(results.items()):
    with cols[i]:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        st.pyplot(fig)

# 标签分布图对比
st.subheader("📊 标签分布对比")
cols2 = st.columns(len(results))
for i, (model_name, res) in enumerate(results.items()):
    with cols2[i]:
        fig, ax = plt.subplots(figsize=(4, 3))
        pd.Series(res["y_pred"]).value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title(f"{model_name} 预测标签分布")
        st.pyplot(fig)

# 数据预览
st.subheader("🔍 标签预测数据预览")
preview_df = test_df.copy()
for model_name, res in results.items():
    preview_df[f"{model_name}_预测"] = res["y_pred"]
st.dataframe(preview_df[['E', 'S', 'G', 'ESG'] + [f"{name}_预测" for name in results.keys()]].head(20))
