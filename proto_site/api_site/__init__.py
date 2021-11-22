import sys,os

proto_path  = "/root/graduation/proto_site"
path_list = [proto_path]
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

for path in path_list:
    if path not in sys.path: 
        sys.path.append(path)
        