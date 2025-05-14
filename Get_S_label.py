import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
from Clean_data import clean_text, split_sentences, Get_text_from_pdf
from G_match import semantic_match_by_keyword
from sklearn.metrics import accuracy_score
import pandas as pd
import gc

gc.collect()

# 关键词列表
keywords = [
    "氣候變化",
    "資源利用",
    "環境汙染",
    "環境友好",
    "環境管理"
]

# 模型路径
model_path = "D:\Pycharm\Project\learning\Graduation_F\model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 加载 label encoder
with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


df = pd.read_excel(r"D:\Pycharm\Project\learning\Graduation_F\ESG_data\predict.xlsx")
        # 提取前两列数据
result = list(zip(df.iloc[:, 0], df.iloc[:, 2]))
# 测试数据列表：文件名, 真实等级
file_info_list = result


true_labels = []
pred_labels = []

# 循环处理每个文件
for idx, (filename, true_label) in enumerate(file_info_list):
    print(f"\n处理文件：{filename}")
    pdf_path = rf"D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF\{filename}"

    try:
        # 文本抽取与处理
        pdf_text = Get_text_from_pdf(pdf_path)
        pdf_text = clean_text(pdf_text)
        sentences = split_sentences(pdf_text)



        # 语义匹配
        all_sentences = sentences
        semantic_result = semantic_match_by_keyword(all_sentences, keywords)
        if not semantic_result:
            print("未匹配到有效内容，跳过")
            continue

        # 模型预测
        inputs = tokenizer(semantic_result, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_label_id = torch.argmax(logits, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]

        print(f"预测：{predicted_label}")
        pred_labels.append(predicted_label)
        true_labels.append(true_label)
    except Exception as e:
        print(f"处理 {filename} 出错：{e}")
        continue

# 准确率评估
if true_labels and pred_labels:
    acc = accuracy_score(true_labels, pred_labels)
    print("\n=== 批量预测结果 ===")
else:
    print("没有有效的预测结果。")
