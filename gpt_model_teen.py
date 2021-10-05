import os,sys
path = '/home/cos2745/workspace/kostat/leevi-python-base'
if path not in sys.path: sys.path.append(path)
# path = '/home/cos2745/workspace/kosis/leevi-python-base'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from leevi_common.database.mongodb import MongoDB
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import pipeline,AutoConfig
from TextDataset import TextDataset
import time
import json

class gpt_model:    

    # def __init__(self,device='cpu',model_name = "skt/kogpt2-base-v2",basepath ="/home/jimin/workspace/voucher/whowant/"):
    def __init__(self,device='cpu',model_name = "skt/kogpt2-base-v2",basepath =None):
        if basepath == None:
            basepath = '/mnt/workspace/voucher/whowant/'
        self.basepath = basepath
        self.modelpath = os.path.join(self.basepath, 'gpt_models')
        self.datapath = os.path.join(self.basepath, 'data')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
        self.device = device
        # 사용할 모델 load 
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    #원본코드 
    def get_contents(self, all_data:list, debug=False):
        # 모델에 사용할 데이터 가지고 오기 
        contents = list()
        modified_contents = list()

        if debug: 
            all_data = all_data[:10]

        for data in all_data:
            content = data.get('contents')
            if len(content) != 1:
                contents.extend(content) 
        
        
        if debug: print("length of contents : " , len(contents))


        if debug: 
            print("length of modified contents : " , len(modified_contents))

        return contents

    # corpus로 수정한 코드 
    # def get_contents(self, all_data:list, debug=False):
    #     # 모델에 사용할 데이터 가지고 오기 
    #     contents = list()
    #     if debug: 
    #         all_data = all_data[:5]

    #     for data in all_data:
    #         content = data.get('corpus')
    #         if len(content) != 1:
    #             contents.extend(content) 
    #     return contents

    #원본코드 
    def split_dataset(self,original_contents,testfilename = "teen_test_dataset.txt", trainfilename = "teen_train_dataset.txt" ,ratio = 0.2, debug = False):

        # train dataset / test dataset 으로 나누기 
        train,test =train_test_split(original_contents,test_size=ratio)
        if debug:  
            print("length of train :", len(train), "length of test : " , len(test))

        with open(os.path.join(self.datapath, testfilename) ,'w',encoding='utf-8') as f :
            for t in test:
                text = t.strip()
                f.write(text+'\n')
            
        with open(os.path.join(self.datapath, trainfilename),'w',encoding='utf-8') as f :
            for t in train:
                text = t.strip()
                f.write(text+'\n')


    # corpus로 수정한 코드 
    # def split_dataset(self,contents,testfilename = "test_corpus.txt", trainfilename = "train_corpus.txt" , ratio=0.5, debug = False):
    #      # train dataset / test dataset 으로 나누기 
    #     train,test =train_test_split(contents,ratio)
    #     if debug: print("length of train :", len(train), "length of test : " , len(test))

    #     with open(os.path.join(self.datapath, testfilename) ,'w',encoding='utf-8') as f :
    #         for t in test:
    #             text = t.strip()
    #             f.write(text)
            
    #     with open(os.path.join(self.datapath, trainfilename),'w',encoding='utf-8') as f :
    #         for t in train:
    #             text = t.strip()
    #             f.write(text)


    def train_save_model (self, testfilename = "teen_test_dataset.txt", trainfilename = "teen_train_dataset.txt",output_dir_name = "kogpt2_teen_contents" , debug = False, device = None):
        if device == None:  
            device = self.device
        
        # 모델 training 이후 save 
        train_dataset = TextDataset(tokenizer=self.tokenizer , file_path=os.path.join(self.datapath,testfilename),block_size=128,device = device)
        if debug: print("======== finished train dataset ========")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        if debug: print("======== finished data_collator ========") 
        test_dataset = TextDataset(tokenizer=self.tokenizer , file_path=os.path.join(self.datapath,trainfilename),block_size=128,device = device)
        if debug: print("======== finished test_dataset ========") 


        training_args = TrainingArguments(
        output_dir = os.path.join(self.modelpath,output_dir_name), #The output directory >> 해당 이름으로 모델이 저장됨 
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=150, # number of training epochs
        per_device_train_batch_size=16, # batch size for training
        per_device_eval_batch_size=8,  # batch size for evaluation
        eval_steps = 10000, # Number of update steps between two evaluations.
        save_steps=20000, # after # steps model is saved
        warmup_steps=500,# number of warmup steps for learning rate scheduler
        # dataloader_pin_memory =False,
        logging_steps = 1000,
        )
        

        trainer = Trainer(
        model=self.model, 
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #  prediction_loss_only=True,
        )

        if debug: print("======== define trainer ========") 

        trainer.train()
        if debug: print("======== finished train ======== ") 

        trainer.save_model()
        print("======== save model finished ========")


def generate_novel(input_keyword="여름방학" , basepath = None , model_name = 'kogpt2_teen_contents', debug=False):
    if basepath == None:
        basepath = '/mnt/workspace/voucher/whowant/gpt_models/'

    try:
        config = AutoConfig.from_pretrained(os.path.join(basepath,model_name))
        model_name_or_path = os.path.join(basepath,model_name)
    except Exception as e:
        print(e)
        config = AutoConfig.from_pretrained(model_name)
        # config.__dict__['task_specific_params']['text-generation']['max_length'] = 800
        model_name_or_path = model_name

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    if debug: print(model_name_or_path,'---',model_name)

    # config.max_length=900
    start = time.time()
    chef = pipeline('text-generation',model=model_name_or_path, tokenizer=tokenizer,config=config)
    print("load_model >>" , time.time()-start)
    # print(chef.model.config.max_length)

    # 학습한 모델에 keyword를 입력하면 소설 return 
    start = time.time()
    generate_text = chef(input_keyword)
    print("generate_text >>" , time.time()-start)

    #소설을 리스트에 저장 
    if debug: 
        print(generate_text[0].get('generated_text'))

    return generate_text




if __name__ == "__main__":

    # device ='cuda:1'
    cuda_device ='cuda'

    gptmodel = gpt_model(device=cuda_device)

  
    mongo = MongoDB(host="office.leevi.co.kr", port=40005, id ="leevi", password = "qlenfrl999", database="whowant" )
    all_data = mongo.find("teen_contents",{})

    # with open("/home/jimin/workspace/voucher/whowant/teen/teen_contents_th_submit.json", "r") as f:
    #     total_json = json.load(f)


    # content = gptmodel.get_contents(all_data,debug=False)

    # gptmodel.split_dataset(content,ratio = 0.2, debug=True)
    # gptmodel.train_save_model(output_dir_name = "kogpt2_teen" )

    # base 모델로 generate 
    generate_novel(model_name='kogpt2-dialog', input_keyword = "여름 어느날", debug=True)

    # train_save 모델로 generate 
    start = time.time() 
    generate_novel(input_keyword = "여름 어느날", model_name="kogpt2_teen",debug=True)  #max_length = 30 이면 9.4초 
    print("소요시간 >> " , time.time()-start) 

    # for _ in range(5):
    #     #키워드
    #     start = time.time() 
    #     new_novel = generate_novel(input_keyword= "가을 아침", model_name= "kogpt2-dialog",debug=False)[0].get('generated_text')
    #     # print("생성된 문장 >> ",new_novel.split(". ")[0]) 
    #     print("소요시간 >> ",time.time()-start)



    print("========end!========")

