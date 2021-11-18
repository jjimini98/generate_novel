import sys 
sys.path.append("C:/Users/Jimin/PycharmProjects/graduation")
import os 
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
import re 
from make_spm_model import MakeSpmModel 
from connect_mongo import connect_mongo


class Novel_Preprocessing: 
    """ 크롤링한 소설 데이터를 전처리 

    '__init__'에 함수들에서 필요한 변수들 초기화 

    """
    def __init__(self):
        self.all_data = None
        self.filepath = "C:/Users/Jimin/PycharmProjects/graduation/data/"
        self.corpus_list = [] 
        self.vocab_size = '8000'
    

    def get_data(self,all_data:list,keys=['title','corpus'],debug=False): 
        """ DB에 저장된 데이터를 가지고 온다. 
        
        all_data : 매개변수로 리스트 형태의 소설 데이터를 받는다. 
        title : 소설 데이터에서 제목에 해당하는 key값을 전달
        corpus : 소설 데이터에서 말뭉치나 문장에 해당하는 key값을 전달 
        debug : True 일 경우, 데이터 전체 중 10개만 가지고 온다. 
        
        """
        training_corpus_list = list() 

        if debug: 
            all_data = all_data[:10]

        for data in all_data:
            training_corpus = dict() 

            for key in keys:
                training_corpus[key] = data.get(key) 
            training_corpus_list.append(training_corpus)

        return training_corpus_list


    def remove_special_char(self, output_filename, special_characters=["■","※"], debug=False): 
        """ 문장단위로 본문을 분리하고 특수문자를 제거한다. 

        filename : 함수 실행 결과를 저장할 파일이름 지정
        special_characters : 리스트로 특수문자를 주면 삭제됨 default : ["■","※"]
        debug : True 일 경우,  수정된 전체 corpus 길이 출력  

        """
        modified_corpus = list()
        for novel in self.get_data(all_data,debug=False):

            for sentence in novel.get('corpus'):
                sentence = sentence.split(". ")
                join_sentence = ".\n".join(sentence)
                join_sentence = re.sub(rf"[{''.join(special_characters)}]",'',join_sentence) # 특수문자 제거
                modified_corpus.append(join_sentence)

            modified_corpus.append("\n")
            modified_corpus.append("\n")

        if debug: 
            print(len(modified_corpus))

        with open(os.path.join(self.filepath,output_filename), 'w' , encoding = 'utf-8') as f:
            for cor in modified_corpus:
                f.write(cor)

        print(f"========{output_filename} created========")


    def convert_unk(self, input_filename, output_filename, debug = False):
        """ unk 에 해당하는 값을 원소설데이터에서 찾는다. 
        
        토큰화의 결과로 생기는 <unk>:unknown token 값 찾아내기 
        input_filename : 사용할 txt 파일 이름 지정 
        output_filename : unk 결과를 저장할 파일 이름 지정
        debug : True 일 경우, 수정된 unk 값들의 목록과 전체길이 출력

        """
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 
        
        with open(os.path.join(self.filepath,input_filename),'r',encoding='utf-8') as f: 
            corpus = f.readlines()

        result = list() 
        numUNK = 0 

        for cor in tqdm(corpus,position=0,leave=False): 
            result_token = tokenizer.tokenize(cor)
            if "<unk>" not in result_token:
                continue  
            #토큰화 결과로 생기는 "_" 제거 후, 원본 소설 문장과 비교해 있으면 없애는 작업 : 결론적으로 <unk>에 해당하는 값만 남게 된다. 
            else:
                tokens = [token.replace("▁" , '') for token in result_token if token != '▁']
                for token in tokens:
                    cor = cor.replace(token, '',1)
                value = cor.strip() + "\t" + str(numUNK) + "\n"
                numUNK += 1 
                result.append(value)

        test = [re.sub(r'\t\d+\n','',i) for i in result]
        renew_unk = list(set([re.search('[가-힣 ]+',i).group().strip() for i in test if re.search('[가-힣 ]+',i)]))
        renew_result = list()

        for text in renew_unk[1:]:
            text = text.split("\t")[0] #문자만 가지고 온다.
            if " " in text:
                renew_text = text.split()
                first = renew_text[0]
                end = renew_text[-1]
                if first==end : 
                    renew_result.append(first)
                else : 
                    renew_result.append(first)
                    renew_result.append(end)
            else: 
                renew_result.append(text)     
        renew_result = set(renew_result)

        if debug: 
            print(renew_result, len(renew_result))

        else: 
            with open(os.path.join(self.filepath+output_filename),'w',encoding='utf-8') as f: 
                for idx, unk in enumerate(renew_result):
                    f.write(unk+"\t"+str(idx)+"\n")

            print(f"========{output_filename} created========")


if __name__ == "__main__":
    pre_novel = Novel_Preprocessing() 
    novelDB = connect_mongo()
    contents = novelDB['contents']
    all_data = contents.find()

    pre_novel.remove_special_char("final_corpus.txt")
    pre_novel.convert_unk("final_corpus.txt", "final_unk.txt")

    # import spm model
    spmModel =  MakeSpmModel()
    spmModel.makeSpmModel("final_corpus.txt","final_novel")

    print("========end!========")