import os
import datetime as dt
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm 
from dotenv import load_dotenv
import pymongo
import pprint
load_dotenv(verbose=True)

IP = os.getenv("IP")
PORT = int(os.getenv("PORT"))
DATABASE = os.getenv("DATABASE")
COL = os.getenv("META_COLLECTION")


conn = pymongo.MongoClient("mongodb://{}:{}".format(IP,PORT))
db = conn[DATABASE]
col = db[COL]

def main(): 
    now = dt.datetime.now()
    year = now.year
    month = now.month

    new = [] # 새로운 소설 정보 저장
    exception = [] # 저장이 안된 소설들 

    url = "https://munjang.or.kr/board.es?mid=a20103000000&bid=0003&nPage=1&b_list_cnt=12&ord=&dept_cd=&tag=&list_no=&listNo=&act=list&view_sdate_param=&keyField=&keyWord="
    plain = requests.get(url).text
    contents = BeautifulSoup(plain,'html.parser')


    for c in contents.find_all("li", {"class" : "list_li"}):
        uploadDate = list(map(int,c.find("span",{"class":"date"}).find("span").text.split("-")))

        if uploadDate[0] == year and uploadDate[1] == month: #업로드 일자가 해당 년, 월과 같은 경우 크롤링 

            global novels
            novels = {}

            try: 
                info = c.find('strong').text.split(" - ")
                novels['title'] = info[1]
                novels["author"] = info[0]
            except:
                exception.extend(info)
                novels['title'] = info[0]
                novels['author'] = "None"
                print("info : ", info[0] , "status : FINISHED")


            novels["links"] = "https://munjang.or.kr" + c.find("a")["href"]
            novels["crawling_date"] = now.strftime("%Y-%m-%d %H:%M:%S")

            if col.find_one({'$and' : [{'title':novels['title']},{'author':novels['author']}] }): # 중복제거 
                pass
            else:
                col.insert_one(novels)
                print("INSERT NOVELS : ", novels.get("title"))



if __name__ == "__main__":
    main()