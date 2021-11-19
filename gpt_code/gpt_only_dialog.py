# kogpt2-dialog_remove_quotation(소설 대화체) 모델을 학습하고 결과 확인

import os 
from gpt_model import * 
from gpt_model import generate_novel
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#gpt_model 클래스 상속 -> get_contents 부분만 메소드 오버라이딩 
import time 

class gpt_only_dialog(gpt_model):
    
    def __init__(self,device = "cuda:0"):
        super().__init__(device = device, model_name = "skt/kogpt2-base-v2", basepath ="/home/cos2745/workspace/voucher/whowant")

    # DB에서 데이터 받아오는 메소드 
    def get_contents(self, all_data : list, debug = False):
        print("length of all data : " , len(all_data))
        contents = list()
        dialog = list()

        if debug: 
            all_data = all_data[:5]
            print("========debug mode========")

        for data in all_data:
            quotation = data.get('quotation_mark') # 쌍따옴표가 있는 대화체 
            
            dialogic = data.get('dialogic_style') # 쌍따옴표가 없는 대화체 

            if len(quotation) != 0 or len(dialogic) != 0: 
                contents.extend(quotation)
                contents.extend(dialogic)

        for sentence in contents:
            dialog.append(sentence[1])

        # 쌍따옴표 있는 경우 
        # new_dialog = self.preprocessing(dialog,remove_quotation_mark = False, debug=False)
        # 쌍따옴표 없는 경우
        # new_dialog = self.preprocessing(dialog , remove_quotation_mark = True , debug = True)
        new_dialog = self.preprocessing(dialog , remove_quotation_mark = True , debug = True)


        if debug: 
            # print("테스트",''.join(new_dialog[:5]))
            print("length of dialog : " , len(dialog))
            print("length of new_dialog : ", len(new_dialog))
        return  new_dialog

    def split_dataset(self, dialog, testfilename, trainfilename, ratio = 0.2, debug=False):
        return super().split_dataset(dialog, testfilename=testfilename, trainfilename=trainfilename, ratio = ratio, debug=debug)

    def train_save_model(self, testfilename, trainfilename, output_dir_name, device=None, debug = False):

        if device == None:  device = self.device
        # 모델 training 이후 save 
        train_dataset = TextDataset(tokenizer=self.tokenizer , file_path=os.path.join(self.datapath,testfilename),block_size=128,device = device)
        if debug: print("======== finished train dataset ========")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        if debug: print("======== finished data_collator ========") 
        test_dataset = TextDataset(tokenizer=self.tokenizer , file_path=os.path.join(self.datapath,trainfilename),block_size=128,device = device)
        if debug: print("======== finished test_dataset ========") 

        training_args = TrainingArguments(
        output_dir = os.path.join(self.modelpath,output_dir_name), #The output directory >> 해당 이름으로 모델이 저장됨 
        overwrite_output_dir = True, #overwrite the content of the output directory
        num_train_epochs = 300, # number of training epochs
        per_device_train_batch_size = 16, # batch size for training
        per_device_eval_batch_size = 16,  # batch size for evaluation
        eval_steps = 400, # Number of update steps between two evaluations.
        save_steps = 800, # after # steps model is saved () 
        warmup_steps = 500,# number of warmup steps for learning rate scheduler
        logging_steps = 100,
        # dataloader_pin_memory =False,
        )

        trainer = Trainer(  
        model = self.model, 
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        # traininig_step = 200, 
        #  prediction_loss_only=True,
        )

        # train_step = trainer.training_step(self.model, training_args)
        # print("train_step : " , train_step)

        if debug: print("======== define trainer ========") 

        trainer.train()
        if debug: print("======== finished train ======== ") 

        trainer.save_model()
        print("======== save model finished ========")

        # return super().train_save_model(testfilename=testfilename, trainfilename=trainfilename, output_dir_name=output_dir_name, debug=debug, device=self.device)

    # regex = r"《문장\s?웹진.*?》|[\-\–\—\―\…\⋯\·]|\n"

    # 쌍따옴표 없는 경우에 대한 코드 수정 필요
    # def reg_fun(self, text,  debug=True):
    #     if debug: print(text)
    #     return re.sub(r'《문장\s?웹진.*?》|[\-\–\—\―\…\⋯]|\n','',text.group())
    #     return re.sub(r'《문장\s?웹진.*?》|[\-\–\—\―\…\⋯\·\“\”\"]|\n','',text.group())

    # 원본코드 
    # def reg_fun_remove(self, text,  debug=False):
    #     if debug: print(text)
    #     return re.sub(r'《문장\s?웹진.*?》|[\-\–\—\―\…\⋯\·\“\”\"\’\‘\']|\n','',text.group())

    def reg_fun(self, text,  debug=False):
        if debug: print(text)
        return re.sub(r'《문장\s?웹진.*?》|[\-\–\—\―\…\⋯\·\“\”\"\’\‘\']|\n','',text.group())

        
    # 쌍따옴표 없는 경우에 대한 수정 필요 #모델을 재학습해야만 따옴표가 사라진 것을 확인할 수 있음 
    def preprocessing(self, data_list : list, remove_quotation_mark = False, debug = False):
        modified_data_list = list()

        for data in data_list:
            # 1. 문장 웹진 , 특수문자 , 줄바꿈 문자 제거 
            # if remove_quotation_mark: 
            data = re.sub(r"《문장\s?웹진.*?》|[\-\–\—\…\⋯\·\“\”\"\’\‘\']|\n" ,self.reg_fun, data)
            # else: 
            #     data = re.sub(r"《문장\s?웹진.*?》|[\-\–\—\…\⋯\·]|\n" ,self.reg_fun, data)

            # 2. 대화체 데이터에 포함되어 있는 비대화체 제거 
            if "다." in data:
                continue  
            # 3. 문장의 길이가 너무 짧은 경우 제거  
            elif len(data) <= 3:
                continue
            else: 
                modified_data_list.append(data)

        return modified_data_list

if __name__ == "__main__":
    device = 'cuda'
    gptmodel = gpt_only_dialog(device)
    
    mongo = MongoDB(host = "office.leevi.co.kr", port = 35005, id = "leevi", password = "qlenfrl999", database = "whowant" )
    all_data = mongo.find("test" , {})

    result = gptmodel.get_contents(all_data , debug = False)

    # 특수문자를 포함하는 경우  
    # gptmodel.split_dataset(result, "test_dialog_quotation.txt", "train_dialog_quotation.txt", ratio=0.2)
    # gptmodel.train_save_model(testfilename = "test_dialog_quotation.txt" , trainfilename = "train_dialog_quotation.txt" , output_dir_name = "kogpt2-dialog_quotation" , debug = False, device = device)
    # generate_novel(input_keyword = "철수야 어디니?", model_name = "kogpt2-dialog_quotation", basepath = "/mnt/workspace/voucher/whowant/gpt_models",debug = True) 
    
    # 특수문자를 포함하지않는 경우
    # gptmodel.split_dataset(result, "test_dialog_remove_quotation.txt", "train_dialog_remove_quotation.txt", ratio=0.2)
    # gptmodel.train_save_model(testfilename = "test_dialog_remove_quotation.txt" , trainfilename = "train_dialog_remove_quotation.txt" , output_dir_name = "kogpt2-dialog_remove_quotation_" , debug = True, device = device)
    start = time.time()
    generate_novel(input_keyword = "겨울 아침 출근길", model_name = "kogpt2-dialog_remove_quotation", basepath = "/mnt/workspace/voucher/whowant/gpt_models",debug = True) 
    print(time.time()- start)

    print("========Finished!========")
