import pymongo
import json



def connect_mongo():
    with open("C:/Users/Jimin/PycharmProjects/graduation/server_info.json","r",encoding="utf-8") as f :
        info = json.load(f)
    ip = info.get("ip")
    port = info.get("port")

    conn = pymongo.MongoClient("mongodb://{}:{}".format(ip,port))
    novel = conn['novel']
    
    return novel