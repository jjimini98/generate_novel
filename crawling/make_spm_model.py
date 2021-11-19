import os 
import sentencepiece as spm


class MakeSpmModel:
    def __init__(self):
        self.vocab_size = '8000'
        self.filepath = "C:/Users/Jimin/PycharmProjects/graduation/data/"
      
    def makeSpmModel(self,input_filename,prefixname):
        parameter =  '--input={} --model_prefix={} --vocab_size={} --model_type={} --character_coverage={}'

        vocab_size = self.vocab_size
        prefix =  os.path.join(self.filepath,"spm/",prefixname)
        input_file = os.path.join(self.filepath,input_filename)

        user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK]'
        pad_id=0
        unk_id=0
        eos_id=-1
        bos_id= -1
        model_type = "bpe"
        character_coverage = 0.9995

        cmd = parameter.format(input_file, prefix, vocab_size,model_type,character_coverage)
        spm.SentencePieceTrainer.Train(cmd)
        print(f"========{prefixname} finished========")


if __name__ == '__main__':
    spmModel =  MakeSpmModel()
    spmModel.makeSpmModel("corpus_test.txt","final_novel")