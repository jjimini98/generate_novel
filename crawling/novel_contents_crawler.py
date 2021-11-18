import sys
pathList = ["C:/Users/Jimin/PycharmProjects/graduation","C:/Users/Jimin/PycharmProjects/graduation/data"] 
for p in pathList : 
    sys.path.append(p)
import json
from tqdm.auto import tqdm
import requests 
import time 
from bs4 import BeautifulSoup
from insert_mongo import insert_mongo
class NovelContentsCrawler():

    #novel_url.json에서 url링크 가져옴 
    def get_data(self,debug=False):
        with open("C:/Users/Jimin/PycharmProjects/graduation/data/novel_url.json", 'r',encoding='utf-8') as f:
            data = json.load(f)
        if debug: data = data[197:198]

        urls = [i.get('bookmarks') for i in data] 
        return urls 

    def refine_data(self,tag_contents):
        self.tag_contents = tag_contents
        corpus = tag_contents.text.replace(u'\xa0', '').replace('*','')
        para = (tag_contents.text.replace(u'\xa0', u'').replace(u'\n', u'')).lstrip()
        self.contents_list.append(para)
        corpus = corpus.lstrip()
        self.corpus_list.append(corpus)
        self.corpus_list = [i for i in self.corpus_list if i and i != ' ']
        self.contents_list = [i for i in self.contents_list if i and i != ' ']
   

    def crawl_novel(self,debug=False):  
        urls =  self.get_data(debug=debug) 

        novels = [] 
        fail_list = list()

        for url in tqdm(urls,leave=True,position=0): 
            novel_dict = dict() 
            self.corpus_list  = [] 
            self.contents_list = [] 

            time.sleep(0.5)

            url_contents = requests.get(url).text
            contents = BeautifulSoup(url_contents, 'html.parser')
            try:
                title =  contents.select_one('h1.entry-title').text
                novel_dict['title'] = title 
            except AttributeError:  # 해당 글이 없는 경우 
                continue
            p_tags = contents.select('div.entry-content > p')

            if len(p_tags) == 1:
                print("this")
                self.refine_data(p_tags[0])
                
            else: 
                for tag in p_tags:
                    if tag.get('align','None').find('justify') > -1 or -1 < tag.get('style','None').find('justify') or -1 < tag.get('TEXT-ALIGN','None').find('justify') : #id 나 class에 justify가 있는 경우를 대비할수 있다
                        html_contents = tag.select_one("span > font")

                        if not html_contents: #html_contents가 false면 (None)
                            html_contents = tag.select_one("span")
                            
                        if html_contents: #html_contents 가 true 면 
                            self.refine_data(html_contents)
                            
                        else:
                            self.refine_data(tag)
                    

                    elif tag.get('align','None').find('left') > -1 or tag.get('align','None').find('right') > -1:   
                            self.refine_data(tag)
                    

            if len(self.contents_list) == 0:
                print('last p tag check',len( p_tags[-1].select('span')))

            if len(self.contents_list) < 1:
                # for tag in p_tags[-1].select('span'): #기존 코드 
                for tag in p_tags:
                    if tag:
                        self.refine_data(tag)
            
            # url 저장 
            novel_dict['url'] = url

            try:
                if self.corpus_list[0].find("단편소설") > -1:
                    del self.corpus_list[0]

                if self.corpus_list[-1].find("문장웹진") > -1:
                    del self.corpus_list[-1]
            
            except IndexError:
                fail_list.append(url) 
                print(len(fail_list) ,fail_list[-1])

            if not self.contents_list: 
                if url not in fail_list:
                    fail_list.append(url)
                    print(f'fail {url} len({len(fail_list)})')
                    continue 
                
            novel_dict['contents'] = self.contents_list 
            novel_dict['corpus'] = self.corpus_list #corpus가 이상하다!~! 
            if debug : print("Length of corpus : " , len(novel_dict['corpus']))
            
            quotation_mark_list, dialogic_style_list = self.get_utterance(novel_dict,debug=True,get_json_dump=False)

            novel_dict['quotation_mark'] = quotation_mark_list
            novel_dict['dialogic_style'] = dialogic_style_list

            # insert_mongo("contents",novel_dict)
            novels.append(novel_dict)
            

        return novels
        

    def convert_json(self, result,file_name): 
        with open(f'C:/Users/Jimin/PycharmProjects/graduation/data/{file_name}.json', 'w', encoding="UTF-8") as f:
            json_file = json.dump(result, f, indent=4, ensure_ascii=False)  #json 파일로 만들기 
            print("==========================Created JSON Files==========================")
        return json_file

    def get_utterance(self, input_novels,debug=False,get_json_dump=False):

        quotation_mark_list= [] 
        dialogic_style_list= []
    
        paragraphs = input_novels['corpus'] #corpus의 value 값들만 가지고 옴 .  
        

        for index,sentences in enumerate(paragraphs): 
        
            sentences = sentences.split("\n")
            for sen in sentences:
                
                if "“" in sen and "”" in sen :
                    start = sen.index("“") 
                    end = sen.index("”") 
                    quotation_mark_list.append((index, sen[start:end+1]))
    
                elif "다." not in sen:
                    # print(sen)
                    if sen in paragraphs :
                        dialogic_style_list.append((index,sen))

                elif "다. " not in sen:
                    # print(sen)
                    if sen in paragraphs:
                        dialogic_style_list.append((index,sen))

                else : continue


        return quotation_mark_list , dialogic_style_list




# 클래스 실행부분 
if __name__ == '__main__':
    crawler = NovelContentsCrawler()
    crawl_result = crawler.crawl_novel(debug=False) 
    result = crawler.convert_json(crawl_result,"novel_contents_test")  #664개 크롤링 시 16분 소요 
    print("end")