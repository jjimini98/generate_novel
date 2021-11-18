import pymongo
import json 

def insert_mongo(collection,data):
    with open("C:/Users/Jimin/PycharmProjects/graduation/server_info.json","r",encoding="utf-8") as f :
        info = json.load(f)
    ip = info.get("ip")
    port = info.get("port")

    conn = pymongo.MongoClient("mongodb://{}:{}".format(ip,port))
    novel = conn['novel']

    conn_collection = novel[collection]

    conn_collection.insert(data)    

    print("============INSERT DATA============")




# 어쩔 수 없는 선택이었따..
def insert_mongo_meta(): 
    with open("C:/Users/Jimin/PycharmProjects/graduation/server_info.json","r",encoding="utf-8") as f :
        info = json.load(f)
    ip = info.get("ip")
    port = info.get("port")


    conn = pymongo.MongoClient("mongodb://{}:{}".format(ip,port))
    novel = conn['novel']
    novel_meta = novel['meta']

    with open("C:/Users/Jimin/PycharmProjects/graduation/data/novel_url.json","r", encoding='utf-8') as f:
        meta_data = json.load(f)

    for data in meta_data:
        novel_meta.insert(data)
    print(f"==========META FINISH==========")



def insert_mongo_contents(): 
    with open("C:/Users/Jimin/PycharmProjects/graduation/server_info.json","r",encoding="utf-8") as f :
        info = json.load(f)
    ip = info.get("ip")
    port = info.get("port")


    conn = pymongo.MongoClient("mongodb://{}:{}".format(ip,port))
    novel = conn['novel']
    novel_meta = novel['contents']

    with open("C:/Users/Jimin/PycharmProjects/graduation/data/novel_contents.json","r", encoding='utf-8') as f:
        contents_data = json.load(f)

    for data in contents_data:
        novel_meta.insert(data)
    print(f"==========CONTENTS FINISH==========")



if __name__ == "__main__": 
    insert_mongo_meta()
    insert_mongo_contents()

