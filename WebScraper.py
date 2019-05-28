import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

url = "https://fangj.github.io/friends/"
html = urlopen(url)

soup = BeautifulSoup(html, 'html.parser')

df = pd.DataFrame()

for link in soup.find_all('a'):
    childurl = url + link.get('href')
    childhtml = urlopen(childurl)

    childsoup = BeautifulSoup(childhtml, 'html.parser')
    text = childsoup.text

    begin = text.find("Scene")
    text = text[begin:]
    text = re.sub("\n", " ", text)

    #Unicode errors from user input

    re.sub(r'[^\x00-\x7f]', r'', '')
    text = re.sub("\xa0", "", text)
    text = re.sub("\u2014", "", text)
    text = re.sub("\u2018", "", text)
    text = re.sub("\u2019", "", text)
    text = re.sub("\u2026", "", text)
    text = re.sub("\u201c", "", text)
    text = re.sub("\u201d", "", text)
    text = re.sub("\u2013", "", text)
    text = re.sub("\xe0", "", text)
    text = re.sub("\xe1", "", text)
    text = re.sub("\xbf", "", text)
    text = re.sub("\xe9", "", text)
    text = re.sub("\xa1", "", text)
    text = re.sub("\x85", "", text)
    text = re.sub("\x92", "", text)
    text = re.sub("\x93", "", text)
    text = re.sub("\x94", "", text)
    text = re.sub("\x97", "", text)
    text = re.sub("\xbb", "", text)
    text = re.sub("\xad", "", text)

    text = re.sub("\xed", "", text)
    text = re.sub("\xe7", "", text)
    text = re.sub("\xe8", "", text)
    text = re.sub("\xc9", "", text)

    re.sub('[^\x00-\x7f]', r'', '')

    text = re.sub("\'", "'", text)

    text = "["+text

    df1 = pd.DataFrame(data=[text])
    df = df.append(df1)
    
df.to_pickle('AllScripts.pkl')





