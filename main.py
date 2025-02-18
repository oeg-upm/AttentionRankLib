from transformers import BertTokenizer, TFBertModel, AutoModel, AutoTokenizer,RobertaTokenizer,RobertaModel

from src.attentionrank.attentions import step_5, step6,step7,step8,step9,step10
from src.attentionrank.ModelEmbedding import ModelEmbedding,NounPhrasesIdentifier
from src.attentionrank.preprocessing import preprocessing_module
from src.attentionrank.eval import evaluate_results, generate_results

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_dir",
    #                    default=None,
    #                    type=str,
    #                    required=True,
    #                    help="The input dataset.")
    parser.add_argument("--dataset_name",
                        default='example',
                        type=str,
                        required=True,
                        help="The input dataset name.")

    parser.add_argument("--model_type",
                        default='roberta',
                        type=str,
                        help="Model type: bert or roberta")

    parser.add_argument("--lang",
                        default='es',
                        type=str,
                        required=True,
                        help="language")

    parser.add_argument("--type_execution",
                        default='exec',
                        type=str,
                        required=True,
                        help="Type of execution: eval or exec")

    parser.add_argument("--k_value",
                        default='15',
                        type=str,
                        required=True,
                        help="K-elements to return")

    parser.add_argument("--model_name_or_path",
                        default='roberta-base',
                        type=str,
                        help="model used")

    #parser.add_argument("--local_rank",
    #                    default=-1,
    #                    type=int,
    #                    help="local_rank for distributed training on gpus")

    #parser.add_argument("--no_cuda",
    #                    action="store_true",
    #                    help="Whether not to use CUDA when available")
    args = parser.parse_args()

    #start = time.time()
    #log = Logger(args.log_dir + args.dataset_name + '.kpe.' + args.doc_embed_mode + '.log')




    '''
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    '''



    modelname= args.model_name_or_path# 'roberta-base'# 'PlanTL-GOB-ES/roberta-base-bne' #'roberta-base'#PlanTL-GOB-ES/roberta-base-bne' #' #'roberta-base'  #'bert-base-uncased'
    type= args.model_type #'roberta'
    lang= args.lang #'en'
    #dataset_name = 'SemEval2018' #dataset_name = 'SemEval2010_GTP3'
    dataset_name = args.dataset_name # 'example'

    k_val = int(args.k_value)

    if type== 'bert':
        tokenizer = BertTokenizer.from_pretrained(modelname)
        model = TFBertModel.from_pretrained(modelname)

    else:
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AutoModel.from_pretrained(modelname, output_attentions=True)




    bertemb= ModelEmbedding(modelname,type, tokenizer, model)
    nounmodel = NounPhrasesIdentifier(lang)




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

    step9(bertemb,dataset_name)
    ## step 10
    print('STEP 10')

    step10(dataset_name)
    ## step 11

    generate_results(dataset_name,lang,k_val)

    if args.type_execution == 'eval':
        print('EVALUATION')
        evaluate_results(dataset_name,k_val)
    





















