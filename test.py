# pymongo로 DB연동 되는지 test   

import pymongo
import json 

with open("C:/Users/Jimin/PycharmProjects/graduation/server_info.json","r",encoding="utf-8") as f :
    info = json.load(f)
# user = info.get("user")
# password = info.get("password")
ip = info.get("ip")
port = info.get("port")


conn = pymongo.MongoClient("mongodb://{}:{}".format(ip,port))
print(conn.list_database_names())


