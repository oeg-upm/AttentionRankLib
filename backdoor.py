from transformers import BertTokenizer, TFBertModel, AutoModel, AutoTokenizer,RobertaTokenizer,RobertaModel

from src.attentionrank.attentions import step_5, step6,step7,step8,step9,step10
from src.attentionrank.ModelEmbedding import ModelEmbedding,NounPhrasesIdentifier
from src.attentionrank.preprocessing import preprocessing_module
from src.attentionrank.eval import evaluate_results, generate_results

#FacebookAI/roberta-base

modelname= 'PlanTL-GOB-ES/roberta-base-bne' #'roberta-base'#PlanTL-GOB-ES/roberta-base-bne' #roberta-base' #'roberta-base'  #'bert-base-uncased'
type='roberta'
lang='es'
#dataset_name = 'SemEval2018'
dataset_name = 'SemEval2010_GTranslate'
dataset_name = 'SemEval2010_GPT3'

if type== 'bert':
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = TFBertModel.from_pretrained(modelname)

else:
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname, output_attentions=True)




bertemb= ModelEmbedding(modelname,type, tokenizer, model)
nounmodel = NounPhrasesIdentifier(lang)

'''


## step 1-4
preprocessing_module(dataset_name,bertemb,type,lang)  #,tokenizer,model
## step 5
print('STEP 5')
step_5(dataset_name,lang,bertemb,nounmodel)
## step 6
print('STEP 6')
step6(dataset_name, 512,2000,)
## step 7
print('STEP 7')
step7(dataset_name)
## step 8

print('STEP 8')
step8(bertemb,dataset_name,nounmodel,lang)
## step 9
print('STEP 9')
'''
step9(bertemb,dataset_name)
## step 10
print('STEP 10')
step10(dataset_name)
## step 11



print('EVALUATION')

n= 5
generate_results(dataset_name,lang,n)
print(n)
evaluate_results(dataset_name,n)


















