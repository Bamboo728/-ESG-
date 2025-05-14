# 进行文本匹配并清洗数据
import fitz
import re

def Get_text_from_pdf(pdf_path):
    Pdf_text = ""
    try:
        file = fitz.open(pdf_path)
        for page in file:
            text = page.get_text()
            # print("Page: ",page.number," Text: ",text)
            Pdf_text += text
        file.close()
    except:
        print("Error: Cannot read PDF file")
    return Pdf_text


def clean_text(text):
    # 去除所有换行符，将文本合并成一行
    text = re.sub(r'\s+', ' ', text)  # 将所有空格、换行符合并为一个空格
    text = re.sub(r'\n|\r', ' ', text)  # 删除所有换行符，合并成一行
    # 去除章节编号、表格残留等（例如“2.3”、“第4章”等）
    text = re.sub(r'\n?\s*\d+\.\d+(?=\s)', '', text)
    text = re.sub(r'第[一二三四五六七八九十]+\s*章', '', text)
    # 去除特殊符号
    text = re.sub(r'[■●◆△→◇]+', '', text)
    # 删除PDF常见残留（页眉页脚、Page XX等）
    text = re.sub(r'Page\s*\d+', '', text)
    # 去除全角空格、不可见字符
    text = re.sub(r'[\u3000\xa0]', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9。，！？：；、“”‘’（）()\[\]{}《》<>/%\-_=+…~·—]+', '', text)
    # 去除额外空格，确保段落之间有单一的空格
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d{8,}', '', text)  # 删除连续8位及以上的数字串
    #text = re.sub(r'[\d/.\-]+%?', '', text)  # 删除如 68%、80/20、2021-2023 等结构
    text = re.sub(r'^\d+-\d+', '', text)
    return text.strip()


def split_sentences(text):
    # 使用 jieba 对文本进行按句子分割
    sentences = text.split("。")  # 这里假设中文句子以“。”为结尾符号
    sentences = [f"“{sentence.strip()}。”" for sentence in sentences if sentence.strip() and len(sentence.strip()) <= 150]  # 去掉空白句子并加上“”括起来
    return sentences


if __name__ == '__main__':
    #path = "D:\Pycharm\Project\learning\Graduation_plus\Company_ESG_Data\ESG_pdf\【00001】 2023年可持續發展報告 (31MB).pdf"
    Pdf_text = Get_text_from_pdf("D:\Pycharm\Project\learning\Graduation_plus\Company_ESG_Data\ESG_pdf\【00001】 2023年可持續發展報告 (31MB).pdf")
    #Pdf_text = extract_text_by_page("D:\Pycharm\Project\learning\Graduation_plus\Company_ESG_Data\ESG_pdf\【00001】 2023年可持續發展報告 (31MB).pdf")
    text = clean_text(Pdf_text)
    sentences = split_sentences(text)
    #sentences = split_sentences_nltk(text)
    print(sentences)

