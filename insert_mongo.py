# mongoDB에 데이터 추가하기 

import sys
path = "/home/jimin/workspace/leevi-python-base"
sys.path.append(path)

import json 
from leevi_common.database.mongodb import MongoDB


with open("voucher/whowant/novel_url.json", "r") as f : 
    url_dict = json.load(f)



# # mongoDB에 소설 데이터 추가하기 
mongo = MongoDB(host="office.leevi.co.kr", port=35005, id ="leevi", password = "qlenfrl999", database="whowant" )
# mongo.insert ("url_data", url_dict, database="whowant")

for url in url_dict:
    mongo.insert ("url_data", url, database="whowant")
    print("insert finished")