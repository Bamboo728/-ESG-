
from DrissionPage import Chromium,ChromiumOptions
from bs4 import BeautifulSoup
import csv
import time
import re

def get_esg_mark(f):
    Tab.ele('xpath:/html/body/div[2]/div[2]/div[1]/div/div[1]/ul/li[7]').click()
    time.sleep(1)
    times = 1
    while times <= 308:
        HTML_text = Tab.html
        soup = BeautifulSoup(HTML_text, 'html.parser')
        style = soup.find('div', style='display: block;')
        divs = style.find('div', class_='s_esg_body')
        if divs is None:
            print("未找到 class_='s_esg_body' 的 div，跳过当前页")
            input("请手动翻页后按回车继续...")
            times += 1
            continue

        for row in divs.find_all('div', class_='s_esg_item s_esg_item2'):
            company_name = row.find('i').text
            company_code = row.find('b').text
            esg_mark = row.find('span', class_='w2 tac').text
            match = re.search(r'\((.*?)\)', esg_mark)
            esg_mark = match.group(1) if match else esg_mark

            E_mark = float(row.find('span', class_='w3').text)
            S_mark = float(row.find('span', class_='w4').text)
            G_mark = float(row.find('span', class_='w5').text)

            E_mark = score_to_rating(E_mark)
            S_mark = score_to_rating(S_mark)
            G_mark = score_to_rating(G_mark)

            print(company_name, company_code, esg_mark, E_mark, S_mark, G_mark)
            f.write(f'{company_name},{company_code},{esg_mark},{E_mark},{S_mark},{G_mark}\n')

        print(f'第 {times} 页处理完毕，请手动翻页后按回车继续...')
        input()  # 等待用户手动翻页后按 Enter
        times += 1


def score_to_rating(score):
    if score >= 95:
        return 'AAA'
    elif score >= 90 and score < 95:
        return 'AA'
    elif score >= 85 and score < 90:
        return 'A'
    elif score >= 80 and score < 85:
        return 'BBB'
    elif score >= 75 and score < 80:
        return 'BB'
    elif score >= 70 and score < 75:
        return 'B'
    elif score >= 65 and score < 70:
        return 'CCC'
    elif score >= 60 and score < 65:
        return 'CC'
    else:
        return 'C'


def get_hk(csv_path_before, csv_path_after):
    # 指定输入和输出文件路径
    input_file = csv_path_before
    output_file = csv_path_after

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8',
                                                                     newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 写入标题行
            headers = next(reader)
            writer.writerow(headers)

            # 筛选并写入符合条件的行
            for row in reader:
                if len(row) > 1 and row[1].lower().startswith('hk'):
                    writer.writerow(row)

        print(f"筛选完成，结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except Exception as e:
        print(f"处理文件时出错：{e}")




if __name__ == '__main__':
    co = ChromiumOptions()
    co.set_argument('--start-maximized')
    page = Chromium(addr_or_opts=co)
    Tab = page.new_tab('https://finance.sina.com.cn/esg/grade.shtml')
    with open('ESG_Data\ESG_mark.csv', 'a', encoding='utf-8-sig', newline='') as f:
        get_esg_mark(f)
    f.close()
    print("done")
    get_hk('D:\Pycharm\Project\learning\Graduation_F\ESG_data\ESG_mark.csv', 'D:\Pycharm\Project\learning\Graduation_F\ESG_data\HK_ESG_mark.csv')