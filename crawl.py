import requests
from bs4 import BeautifulSoup


def get_poem(path='./data'):
    for i in range(1, 51):
        url = requests.get('https://yoondongju.yonsei.ac.kr/poet/poem.asp?ID=' + str(i))

        bs_obj = BeautifulSoup(url.content.decode('euc-kr', 'ignore').encode('utf-8'), 'html.parser')
        data = bs_obj.find('div', {'id': 'con'})

        text = str(data).replace('<br>', '\n').replace('<br/>', '\n')
        text = text[text.find('</p>') + 4:text.find('<a') - 2]

        n = f'0{str(i)}' if i < 10 else f'{str(i)}'
        filename = f'{path}/poem{n}.txt'

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
