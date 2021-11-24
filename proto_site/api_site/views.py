from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
from transformers import PreTrainedTokenizerFast , pipeline 
import os , time
import torch
from concurrent.futures import ThreadPoolExecutor

# Create your views here.

# 모델의 경로 및 이름 지정 
path = "/root/graduation/gpt_models"
model_name_or_path = os.path.join(path, "kogpt2-dialog/checkpoint-140000") 
# model_name_or_path = os.path.join(path, "skt/kogpt-base-v2") 


config =os.path.join( model_name_or_path, "config.json") 
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

model = pipeline('text-generation',model=os.path.join(path,model_name_or_path), tokenizer=tokenizer,config=config)
# model = pipeline('text-generation',model=os.path.join(path,model_name_or_path), tokenizer=tokenizer,device=-1)


#클래스 뷰로 GET 방식 처리 
class ThreadResource(object):
    def __init__(self):
        self.result_list = list()
        self.flag = True

    def len_result(self):
        return len(self.result_list)

    def thread_generate(self,input_value, total, max_sentence=2 , debug = True):
        """ 
        input_value : 사용자로부터 원하는 단어 또는 문장을 입력받는다 
        total : 사용자로부터 원하는 문장 개수를 입력받는다 (최대 5)
        max_sentence : 사용자로부터 한 줄에 출력될 문장 개수를 입력받는다 (최대 5) 
        """

        count = 0 
        
        # total, max_sentence 값이 5보다 큰경우, 5로 제한해준다. 
        total = 5 if int(total) >5 else int(total)
        max_sentence = 5 if int(max_sentence) >5 else int(max_sentence)


        while self.flag :
            start = time.time()
            count += 1 

            if self.len_result() < total:  
                output=  model(input_value)[0].get('generated_text')

                # generated_sentences = split_sentences(output) #split_sentences로 문장 쪼개기(문장 1개가 들어왔을 경우)
                # result = generated_sentences[:max_sentence]

                result = output.split(". ")[:max_sentence]

                if debug : print("문장 생성!")
                
            # 중복 제거 (생성된 문장이 result_list 안에 없으면서 result_list의 길이가 total보다 작을때)
            if result not in self.result_list and self.len_result() < total:
                self.result_list.append(result)

            # 반복(while)의 횟수가 10번이거나 결과의 총 길이가 5보다 크거나 같아질때 
            if count == 5 or self.len_result()>=5: 
                if debug: 
                    print("total in Th seconds : ", time.time()-start)
                self.flag = False

        if debug: print("최종 생성 문장 리스트 : " , self.result_list)
    
class HtmlView(View):
    def get(self,request): 
        # ThreadResource 객체 생성
        thread_resource = ThreadResource()

        result_list = list()
        keyword = request.GET.get('keyword')
        max_sentence = request.GET.get('max_sentence')
        if keyword is None:
            return render(request,"api_site/home.html",{'data':None}) 

        total_num = request.GET.get('total_num',5)
        worker = int(total_num)
        request_time =time.time()

        with ThreadPoolExecutor(max_workers=worker) as executor:
            for _ in range(worker):
                executor.submit(thread_resource.thread_generate, keyword,total_num, max_sentence)  

        # print(f'result time {time.time()-request_time}')
        result_list =thread_resource.result_list
        result_list.insert(0,{'keyword':keyword})
        thread_resource.__init__()

        return render(request,"api_site/home.html",{"data":result_list}) 

#TODO 1) 키워드없이 문장이 생성되는 기능 추가  2) 단어 추천 
