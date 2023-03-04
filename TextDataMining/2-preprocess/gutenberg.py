import requests
import re 
import numpy as np
URL = "https://www.gutenberg.org"


def BookUrl(url):
    resp = requests.get(url)
    resp.encoding = "utf-8"
    text = resp.text
    obj1 = re.compile(r'<h2 id="books-last1">Top 100 EBooks yesterday</h2>\n\n<ol>\n(.*?)</ol>', re.S)
    text = obj1.findall(text)[0]
    obj2 = re.compile(r'<li><a href="(.*?)">', re.S)
    BookURL = np.array(obj2.findall(text))
    return BookURL

def download(bookurl):
    text_url = URL+bookurl+".txt.utf-8"
    resp = requests.get(text_url)
    resp.encoding = 'utf-8'
    text = resp.text
    obj = re.compile(r'\r\n\r\n\r\n\r\n\r\n\r\n(.*?)\r\n\r\n\r\n\r\n\r\n\r\nEnd', re.S)
    text = obj.findall(text)[0]
    return text

if __name__ == "__main__":
    TopURL = URL+"/browse/scores/top"
    BookURL =BookUrl(TopURL)
    text = download(BookURL[51])
    f = open("book.txt",'w')
    f.write(text)
    f.close