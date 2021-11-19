import os, sys
pathList = ["C:/Users/Jimin/PycharmProjects/graduation","C:/Users/Jimin/PycharmProjects/graduation/data"] 
for p in pathList : 
    sys.path.append(p)

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from TextDataset import TextDataset
from connect_mongo import connect_mongo



def get_contents():
    novelDB = connect_mongo()
    contents = novelDB['contents']
    all_data = contents.find()

    # training_corpus_list=list()
    for x in all_data:
        # training_corpus = dict() 
        # training_corpus['title'] = x.get('title') 
        # training_corpus['contents'] = 
        content = x.get('contents')
        if len(content) != 1:
            contents.extend(content)
    return contents

contents = get_contents()

train,test =train_test_split(contents,test_size=0.15)

with open('test.txt','w',encoding='utf-8') as f :
    data = ''
    for i in test:
        text = i.strip()
        # data += text + ' '
        f.write(text+'\n')
    
with open('train.txt','w',encoding='utf-8') as f :
    data = ''
    for i in train:
        text = i.strip()
        # data += text + ' '
        f.write(text+'\n')
    # f.write(data)



tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

train_dataset = TextDataset(tokenizer=tokenizer,file_path='test.txt',block_size=128)
test_dataset = TextDataset(tokenizer=tokenizer,file_path='train.txt',block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
#모델로드

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

training_args = TrainingArguments(
    output_dir="./gpt_models/kogpt2-train_save", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=30, # number of training epochs
    per_device_train_batch_size=16, # batch size for training
     per_device_eval_batch_size=16,  # batch size for evaluation
     eval_steps = 400, # Number of update steps between two evaluations.
     save_steps=800, # after # steps model is saved
     warmup_steps=500,# number of warmup steps for learning rate scheduler
     )
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
#     prediction_loss_only=True,
)

trainer.train()
trainer.save_model()