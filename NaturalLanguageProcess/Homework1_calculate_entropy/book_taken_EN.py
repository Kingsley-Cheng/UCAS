import re
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import etree

def get_detail_href(url):
    """
    该函数负责获取到每一个详情页的值
    """
    print("开始分析主界面")
    resp = requests.get(url)
    resp.encoding = "utf-8"
    et = etree.HTML(resp.text)
    hrefs = et.xpath("//div[@class = 'each_truyen']/a/@href")
    print("分析完成")
    return hrefs


def get_page_srcs(url):
    print("开始抓取子页面")
    resp = requests.get(url)
    resp.encoding = "utf-8"
    et = etree.HTML(resp.text)
    hrefs = et.xpath("//ul[@class='list-chapter']/li/a/@href")
    hrefs = [hrefs[0], hrefs[6]]
    print("抓取成功")
    return hrefs


def download_text(src1, src2, i):
    t = 0
    print("开始下载章节")
    while True:
        resp = requests.get(src1)
        resp.encoding = "utf-8"

        obj = re.compile(r'<p>(.*?)</p>', re.S)
        content1 = obj.findall(resp.text)
        content = "".join(content1)
        content = content.replace("&#8230", "...")
        with open(f"BOOK_EN/book{i}", mode="a") as f:
            f.write(content)
        if src1 == src2 or t > 5000:
            break
        et = etree.HTML(resp.text)
        src1 = et.xpath('//a[@id = "next_chap" ]/@href')[0]
        t = t + 1
        print(t)
    print("下载本书完毕")


def main():
    url = "https://engnovel.com/adventure-novels"
    # 1. 抓取到首页中详情页到href
    hrefs = get_detail_href(url)
    i = 0

    with ThreadPoolExecutor(200) as t:
        for href in hrefs:
            i = i + 1
            page_src_list = get_page_srcs(href)
            t.submit(download_text, page_src_list[1], page_src_list[0], i)


if __name__ == "__main__":
    main()
