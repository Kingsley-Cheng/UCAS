from concurrent.futures import ThreadPoolExecutor

import requests
import re
from lxml import etree
import numpy as np
import time
domain = "https://movie.douban.com/subject/26266893/comments?"
url = "limit=20&status=P&sort=new_score"


def getPage(url):
    """
    该函数负责获取到每一个详情页的值
    """
    print("开始分析主界面")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    resp = requests.get(url, headers=headers)
    resp.encoding = "utf-8"
    text = resp.text
    et = etree.HTML(text)
    comments = np.array(et.xpath("//span[@class='short']/text()"))
    for comment in comments:
        
    obj = re.compile(r'<span class="allstar(.*?)0 rating', re.S)
    score = np.array(obj.findall(text))
    result = np.squeeze(np.dstack((score, comments)))
    time.sleep(1)
    return result

if __name__ == "__main__":
    # flag = True
    # i = 1
    # result = []
    # while flag:
    #     text = getPage(domain + url)
    #     result.append(getPageCommentScore(text))
    #     url = f"?start={20 * i}limit=20&status=P&sort=new_score"
    #     i = i + 1
    #     if text is None:
    #         flag = False
    result = getPage(domain+url)
    print(result)
