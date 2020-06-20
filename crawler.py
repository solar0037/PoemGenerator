import re
import requests
from bs4 import BeautifulSoup


def get_poem():
    title, poem = [], []
    for i in range(1, 51):
        url = requests.get('https://yoondongju.yonsei.ac.kr/poet/poem.asp?ID=' + str(i))

        bs_obj = BeautifulSoup(url.content.decode('euc-kr', 'ignore').encode('utf-8'), 'html.parser')
        data = bs_obj.find('div', {'id': 'con'})

        data_title = data.find('p', {'id': 'title'})
        title_text = data_title.text

        poem_text = re.sub('<br>', '\n', re.sub(r'<br/>', '\n', str(data)))
        poem_text = poem_text[poem_text.find('</p>') + 4:poem_text.find('<a') - 2]

        title.append(title_text)
        poem.append(poem_text)

        if i < 10:
            n = '0' + str(i)
        else:
            n = str(i)

        with open('./poems/poem' + n + '.txt', 'w', encoding='utf-8') as f:
            f.write(poem_text)
