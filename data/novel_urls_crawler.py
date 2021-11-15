# 소설 본문 URL 크롤링

import datetime as dt
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import json 


class NovelUrlCrawler():

    def get_archives(self, max_page=68, debug=False):
        if debug:
            max_page = 1
        result = [] 
        for p in tqdm(range(1, max_page+1),leave=True,position=0):
            now = dt.datetime.now()
            novel_url = f"https://webzine.munjang.or.kr/archives/category/novel/page/{p}"
            # 페이지에서 사진, 제목, 지은이를 모두 포함하는 가장 큰 태그로 접근하기 -> id = main , class = site-main // data_selector = "main.site-main")
            plain_contents = requests.get(novel_url).text
            contents = BeautifulSoup(plain_contents, 'html.parser')

            for post_contents in contents.find_all('div', {'class' : 'post_content'}):
                global elements 
                elements = dict()
                elements['bookNames'] = post_contents.find('a',rel='bookmark').text # post_content에서 태그가 a 이고 rel 이 bookmark인 경우 그 값을 text 형태로 바꿔서 bookNames의 요소로 저장
                elements['bookmarks'] = post_contents.find('a',rel='bookmark')['href'] # post_content에서 태그가 a이고 rel이 bookmark 인 경우 
                elements['authors'] = post_contents.find('span',class_='post_au_name').a.text   #post_content에서 태그가 span이면서 클래스명이 post_au_name인 경우, 그 중 a 태그를 가지고 와서 text로  변경 
                elements['crawling_time'] = now.strftime("%Y-%m-%d %H:%M:%S")
                elements['posting_date'] = (post_contents.find_all("div",{'class' : 'post_title_mobile'})[1].text).split("/")[1]
                result.append(elements)

                # novel_contents_prac.json 파일에서 생성된 corpus 파일을 열어서 corpus의 길이를 추가 
                # with open("C:\Users\Jimin\PycharmProjects\graduation\novel/novel_contents_prac.json",encoding="utf-8") as file:
                #     file = json.load(file)
                #         # for novel in range(len(file)):
                #     elements['length_of sentences'] = len(file[post_content.index(i)]['corpus'])
                    
                # result.append(elements)
        return result


    def json_converter(self, result, filename): 
        with open(f'C:/Users/Jimin/PycharmProjects/graduation/novel/{filename}.json', 'w', encoding="UTF-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print("Created JSON File")



if __name__ == '__main__':
    crawler = NovelUrlCrawler()
    result = crawler.get_archives(debug=False)
    json_result = crawler.json_converter(result,"novel_url_test")
    print ('end')


   













   