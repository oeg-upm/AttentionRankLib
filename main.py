from transformers import BertTokenizer, TFBertModel

from src.attentionrank.attentions import ModelEmbedding, step_5, step6,step7,step8,step9,step10
from src.attentionrank.preprocessing import preprocessing_module
from src.attentionrank.eval import evaluate_results





bertemb= ModelEmbedding('bert-base-uncased')
#bertemb(["feature dollar",'Wolverstein'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
root_folder = './SemEval2017/'
dataset_name = 'SemEval2017'


## step 1-4
preprocessing_module(root_folder,dataset_name,tokenizer,model)


## step 5
step_5(root_folder,dataset_name)

'''
## step 6
step6(tokenizer,0,0)
## step 7
step7()
## step 8
step8(bertemb)
## step 9
step9(bertemb)
## step 10
step10()
## step 11

evaluate_results(root_folder,dataset_name)
'''










