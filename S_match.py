
from Clean_data import clean_text, split_sentences, Get_text_from_pdf
from sentence_transformers import SentenceTransformer, util
from typing import List
from collections import defaultdict
from Get_tabels import convert_excel_to_sentences,extract_tables_from_pdf,save_all_tables_to_excel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import json
import csv
import torch
import os


pdf_list = []
with open('D:\Pycharm\Project\learning\Graduation_F\ESG_data\Match_pdf_test.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)
    for row in reader:
        pdf_list += row

keywords = [
    "人力資源",
    "產品責任",
    "供應鏈",
    "社會貢獻",
    "數據安全與隱私"

]


Labels_list = []
with open('D:\Pycharm\Project\learning\Graduation_F\ESG_data\Match_mark_test.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)
    for row in reader:
        Labels_list.append(row[4])


def semantic_match_by_keyword(
        sentences: List[str],
        keywords: List[str],
        model_name: str = r'D:\Anaconda\models\text2vec-base-chinese',
        top_k: int = 10,
        similarity_threshold: float = 0.35
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)

    if not sentences:
        print(" 警告：传入的句子列表为空，跳过匹配。")
        return ""

    if not keywords:
        print(" 警告：关键词列表为空，跳过匹配。")
        return ""

    try:
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True).to(device)
        keyword_embeddings = model.encode(keywords, convert_to_tensor=True).to(device)
    except Exception as e:
        print(" [向量编码失败]：", e)
        return ""

    keyword_to_sentences = defaultdict(list)

    for k_idx, k_emb in enumerate(keyword_embeddings):
        try:
            sims = util.cos_sim(k_emb.unsqueeze(0), sentence_embeddings)[0]
            top_indices = sims.topk(k=top_k).indices.tolist()

            matched = []
            for i in top_indices:
                if sims[i].item() >= similarity_threshold:
                    matched.append(sentences[i])

            if matched:
                keyword_to_sentences[keywords[k_idx]] = "。".join(matched) + "。"
        except Exception as e:
            print(f" [处理关键词失败]：{keywords[k_idx]} -> {e}")

    result = []
    for keyword, matched_text in keyword_to_sentences.items():
        result.append(f"【{keyword}】\n{matched_text}")

    return "\n\n".join(result)


def model_transform():

    # 本地模型路径
    local_model_path = r"D:\Anaconda\models\text2vec-base-chinese"

    # 自动加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path)

    # 使用 sentence-transformers 包装成 SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling

    word_embedding_model = Transformer(local_model_path)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 保存为 SentenceTransformer 能识别的格式
    sbert_model.save(local_model_path)

    print("模型已转换为 SentenceTransformer 格式，可直接使用。")



if __name__ == '__main__':
    model_transform()
    dataset = []
    # === 断点读取：跳过已处理过的PDF文件 ===
    processed_file_path = "processed_files2.txt"
    processed_files = set()
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f if line.strip())
    for i in range(len(pdf_list)):
        pdf_name = pdf_list[i]
        if pdf_name in processed_files:
            print(f" 跳过第 {i + 1} 个：{pdf_name}（已处理）")
            continue
        print(f"\n 正在处理第 {i + 1}/{len(pdf_list)} 个 PDF：{pdf_name}")
        try:
            pdf_path = rf"D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF\{pdf_name}"
            Pdf_text = Get_text_from_pdf(pdf_path)
            text = clean_text(Pdf_text)
            if not text.strip():
                print("清洗后的文本为空，跳过此文件")
                continue
            sentences = split_sentences(text)
            tabels = extract_tables_from_pdf(pdf_path, i + 1)
            save_all_tables_to_excel(tabels, i + 1)
            table_path = rf"D:\Pycharm\Project\learning\Graduation_F\Tabels_To_xlsx\company{i + 1}_tables.xlsx"
            tabel_sentence = convert_excel_to_sentences(table_path)
            all_sentence = sentences + tabel_sentence
            if not all_sentence:
                print("句子集合为空，跳过此文件")
                continue
            matched_text = semantic_match_by_keyword(all_sentence, keywords)
            if not matched_text.strip():
                print("无匹配结果，跳过此文件")
                continue
            label = Labels_list[i] if i < len(Labels_list) else "未知"
            dataset.append({"text": matched_text.strip(), "label": label})
            # === 保存处理记录 ===
            with open(processed_file_path, "a", encoding="utf-8") as f:
                f.write(f"{pdf_name}\n")
            print(f"第 {i + 1} 个 PDF 处理完成，并记录断点。")
        except Exception as e:
            print(f" 错误：第 {i + 1} 个 PDF 处理失败 -> {e}")
            continue
    # === 输出最终数据集 ===
    with open("S_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("\n 所有未处理 PDF 处理完毕，数据集已保存为 S_dataset.json")