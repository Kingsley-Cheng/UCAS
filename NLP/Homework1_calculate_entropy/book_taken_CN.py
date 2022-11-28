from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import etree

domain = "https://www.82zw.com/"


def get_detail_href(url):
    """
    该函数负责获取到每一个详情页的值
    """
    print("开始分析主界面")
    resp = requests.get(url)
    resp.encoding = "gbk"
    et = etree.HTML(resp.text)
    hrefs = et.xpath("//span[@class='s2']/a/@href")
    print("分析完成")
    return hrefs


def get_page_srcs(url):
    print("开始抓取子页面")
    resp = requests.get(url)
    resp.encoding = "gbk"
    et = etree.HTML(resp.text)
    hrefs = et.xpath("//div[@class = 'box_con']/div/dl/dd/a/@href")
    new_hrefs = []
    for href in hrefs:
        new_hrefs.append(domain + href)
    new_hrefs = new_hrefs[12:]
    print("抓取成功")
    return new_hrefs


def download_text(src, i):
    resp = requests.get(src)
    resp.encoding = "gbk"
    print("开始下载章节")
    re = etree.HTML(resp.text)
    content_inital = re.xpath('//div[@id = "content"]/text()')[:-2]
    content = "".join(content_inital)
    with open(f"BOOK_CN/book{i}", mode="a") as f:
        f.write(content)
    print("下载完毕")


def main():
    url = "https://www.82zw.com/xuanhuanxiaoshuo/"
    # 1. 抓取到首页中详情页到href
    hrefs = get_detail_href(url)
    i = 0
    with ThreadPoolExecutor(200) as t:
        for href in hrefs:
            i = i + 1
            page_src_list = get_page_srcs(href)
            for page in page_src_list:
                t.submit(download_text, page, i)
            if i == 20:
                break

if __name__ == "__main__":
    main()
