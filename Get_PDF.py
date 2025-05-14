from DrissionPage import Chromium,ChromiumOptions
from bs4 import BeautifulSoup
import aiohttp
import aiofiles
import asyncio
import os


def Get_download_link():
    name_list = []
    link_list = []
    co = ChromiumOptions()
    co.set_argument('--start-maximized')
    ESG_web = Chromium(addr_or_opts=co)
    Tab = ESG_web.new_tab('https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=zh')


    Tab.ele('xpath://*[@id="tier1-select"]/div/div').click()
    Tab.ele('xpath://*[@id="hkex_news_header_section"]/section/div[1]/div/div[2]/ul/li[4]/div/div[2]/div[1]/div[2]/div/div/div/div[1]/div[2]/a').click()
    Tab.ele('xpath://*[@id="rbAfter2006"]/div[1]/div/div').click()
    Tab.ele('xpath://*[@id="rbAfter2006"]/div[2]/div/div/div/ul/li[5]/a').click()
    Tab.ele('xpath://*[@id="rbAfter2006"]/div[2]/div/div/div/ul/li[5]/div/div/ul/li[3]/a').click()
    Tab.ele('xpath://*[@id="searchDate-From"]').click()
    Tab.ele('xpath://*[@id="date-picker"]/div[1]/b[1]/ul/li[3]/button').click()

    ele1 = Tab.ele('xpath://*[@id="date-picker"]/div[1]/b[2]/ul/li[7]/button')
    Tab.actions.click(ele1,times=2)
    Tab.ele('xpath://*[@id="searchDate-To"]').click()
    Tab.ele('xpath://*[@id="date-picker"]/div[1]/b[1]/ul/li[2]/button').click()

    ele2 = Tab.ele('xpath://*[@id="date-picker"]/div[1]/b[2]/ul/li[7]/button')
    Tab.actions.click(ele2,times=2)
    Tab.ele('xpath://*[@id="hkex_news_header_section"]/section/div[1]/div/div[3]/a[2]').click()
    for i in range(24):
        Tab.ele('xpath://*[@id="recordCountPanel2"]/div[1]/div/div[1]/ul/li/a').click()
        Tab.wait(3)

    print("页面加载完毕！")

    page_source = Tab.html
    soup = BeautifulSoup(page_source, 'html.parser')
    tbody = soup.find('tbody',attrs={'aria-live': 'polite', 'aria-relevant': 'all'})
    for tr in tbody.find_all('tr'):
        stock_code = tr.find('td',class_ = 'text-right stock-short-code').text.strip()[5:]
        pdf_link = tr.find('a')['href']
        pdf_name = tr.find('div',class_ = 'doc-link').text.strip()
        complete_link = 'https://www1.hkexnews.hk' + pdf_link
        complete_name = f"【{stock_code}】 {pdf_name}"
        name_list.append(complete_name)
        link_list.append(complete_link)
        # print(complete_name,complete_link)

    return name_list,link_list



async def Download_PDF(i,url,name):
    heard = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
    }
    semaphore = asyncio.Semaphore(5)
    async with semaphore:
        try:
            async with aiohttp.ClientSession(headers=heard) as session:
                response = await asyncio.wait_for(session.get(url),timeout=120)
                if response.status == 200:
                    text = await response.read()
                    os.makedirs(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF', exist_ok=True)

                    valid_name = name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace(
                        '?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

                    async with aiofiles.open(rf'D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF\{valid_name}.pdf', 'wb') as f:
                        await f.write(text)
                        print(f'第{i+1}个文件下载完成')
                else:
                    print(f'第{i+1}个文件下载失败')
        except asyncio.TimeoutError:
            print(f'第{i+1}个文件下载超时,跳过该文件')


async def main(name_list,url_list):
    tasks = []
    for i in range(len(link_list)):
        task = asyncio.create_task(Download_PDF(i,url_list[i],name_list[i]))
        tasks.append(task)
    await asyncio.wait(tasks)


async def Download_PDF_one(i, url, name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
    }

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            response = await asyncio.wait_for(session.get(url), timeout=120)
            if response.status == 200:
                content = await response.read()
                os.makedirs(r'D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF', exist_ok=True)

                valid_name = name.replace('/', '_').replace('\\', '_').replace(':', '_') \
                                 .replace('*', '_').replace('?', '_').replace('"', '_') \
                                 .replace('<', '_').replace('>', '_').replace('|', '_')

                async with aiofiles.open(rf'D:\Pycharm\Project\learning\Graduation_F\ESG_data\PDF\{valid_name}.pdf', 'wb') as f:
                    await f.write(content)
                print(f'第{i+1}个文件下载完成')
            else:
                print(f'第{i+1}个文件下载失败，状态码: {response.status}')
    except asyncio.TimeoutError:
        print(f'第{i+1}个文件下载超时, 跳过该文件')
    except Exception as e:
        print(f'第{i+1}个文件下载出错: {e}')


async def main_one(name_list, url_list):
    for i in range(len(url_list)):
        await Download_PDF_one(i, url_list[i], name_list[i])



if __name__ == '__main__':
    name_list,link_list = Get_download_link()
    asyncio.run(main_one(name_list,link_list))
    print("所有文件下载完成")



