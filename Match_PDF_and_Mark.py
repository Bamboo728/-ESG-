# Step 3:匹配有ESG评级的上市公司PDF文件

import pandas as pd
import os
import re
import csv

# 路径配置
csv_path = r"D:\Pycharm\Project\learning\Graduation_F\ESG_data\HK_ESG_mark.csv"
pdf_folder_path = r"D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF"


df = pd.read_csv(csv_path,encoding = 'gbk')
stock_codes = df.iloc[:, 1].astype(str).str.zfill(5).tolist()

code_list = []
for i in stock_codes:
    code_list.append(i[2:])


# 获取所有PDF文件名
pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

# 筛选匹配的PDF文件
matched_pdfs = []
matched_infos = []
for pdf in pdf_files:
    match = re.search(r"【(\d{5})】", pdf)
    # print(match)
    if match:
        code = match.group(1)

        if code in code_list:
            matched_pdfs.append(pdf)

            row = df[df.iloc[:, 1].astype(str).str.contains(code)]
            if not row.empty:
                company_name = row.iloc[0, 0]
                stock_code = row.iloc[0, 1]
                esg_rating = row.iloc[0, 2]
                e_rating = row.iloc[0, 3]
                g_rating = row.iloc[0, 4]
                s_rating = row.iloc[0, 5]
                matched_infos.append([company_name, stock_code, esg_rating, e_rating, g_rating, s_rating])


with open('D:\Pycharm\Project\learning\Graduation_F\ESG_data\Match_pdf.csv', 'w', newline='', encoding='utf-8') as f:
    # 输出匹配的PDF文件名
    k = 0
    print("匹配成功的PDF文件：")
    for pdf in matched_pdfs:
        print(pdf)
        f.write(pdf+'\n')
        k += 1
    print(k)

f.close()

with open(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\Matched_mark.csv', 'w', newline='', encoding='utf-8-sig') as f2:
    l = 0
    writer = csv.writer(f2)
    for info in matched_infos:
        writer.writerow(info)
        l += 1
    print(l)
f2.close()




