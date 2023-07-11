
#from attentionrank import step11
from src.attentionrank.eval import evaluate_results


datasetname = 'SemEval2017'
datasetpath = "./" + datasetname
evaluate_results(datasetpath,datasetname)