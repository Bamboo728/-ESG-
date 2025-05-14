import os
import pdfplumber
import pandas as pd
import csv


def extract_tables_from_pdf(pdf_path, i):
    output_dir = f"PDF_Tabels\company{i+1}"
    os.makedirs(output_dir, exist_ok=True)
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or not table[0]:
                    continue
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                except Exception as e:
                    print(f"表格转换失败：第{page_num+1}页 第{table_idx+1}个表格：{e}")
                    continue

                table_path = os.path.join(output_dir, f"page{page_num+1}_table{table_idx+1}.csv")
                df.to_csv(table_path, index=False, encoding='utf-8-sig')
                all_tables.append({
                    "page": page_num + 1,
                    "index": table_idx + 1,
                    "path": table_path,
                    "dataframe": df
                })

    print(f"表格提取完成，共提取 {len(all_tables)} 个表格。")
    return all_tables


def save_all_tables_to_excel(tables, i):
    output_excel_path = f"Tabels_To_xlsx\company{i}_tables.xlsx"
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        for table in tables:
            sheet_name = f"p{table['page']}_t{table['index']}"
            df = table['dataframe']
            # Excel sheet 名长度限制为 31
            writer.book.use_zip64()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"所有表格已保存到 Excel 文件：{output_excel_path}")



def convert_excel_to_sentences(excel_path: str) -> list:
    """

    """
    sentences = []
    xls = pd.ExcelFile(excel_path)

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.dropna(how='all').dropna(axis=1, how='all')  # 删除全为空的行和列

        for _, row in df.iterrows():
            row_sentences = []
            for col in df.columns:
                value = row[col]
                if pd.notnull(value):
                    row_sentences.append(f"{col}為{value}")
            if row_sentences:
                sentence = "，".join(row_sentences)
                # 清洗掉 \n 和 \r，并去除首尾空格
                sentence = sentence.replace("\n", "").replace("\r", "").strip()
                sentences.append(sentence)

    return sentences


if __name__ == "__main__":
    i = 0
    pdf_path = rf'D:\Pycharm\Project\learning\Graduation_plus\Company_ESG_Data\ESG_pdf\【00001】 2023年可持續發展報告 (31MB).pdf'
    tabels = extract_tables_from_pdf(pdf_path, i+1)
    save_all_tables_to_excel(tabels, i+1)
    all_sentence = convert_excel_to_sentences(f"Tabels_To_xlsx\company{i+1}_tables.xlsx")
    print(all_sentence)

"""
    pdf_list = []
    with open('D:\Pycharm\Project\learning\Graduation_F\ESG_data\Match_pdf.csv', 'r',
              encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            pdf_list += row

    for i in range(len(pdf_list)):
        pdf_path = rf"D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF\{pdf_list[i]}"
        tabels = extract_tables_from_pdf(pdf_path,i+1)
        save_all_tables_to_excel(tabels,i+1)
        #all_sentence = convert_excel_to_sentences(f"company{i}_tables.xlsx")



"""



