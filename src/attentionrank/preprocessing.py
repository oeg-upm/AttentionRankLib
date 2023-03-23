

from transformers import BertTokenizer, TFBertModel

import numpy
import os
import pickle
import json
from .utils import clean_folder
import csv
import glob




#### STEP 1-4 ####






def preprocess_file(root_folder, file_name,tokenizer,model,dataset_name):
    file_identifier = file_name[:-4]
    file_path = root_folder + 'docsutf8/' + file_name  # './SemEval2017/docsutf8/S0010938X1500195X.txt'
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', '')

    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input, output_attentions=True)
    attentions = output.attentions

    array_map = []
    for map in attentions:
        array_map.append(map)
    array_map = numpy.array(array_map)
    array_map = array_map[:, 0, :, :, :]

    # with open('./SemEval2017/docsutf8/S0010938X1500195X.txt', 'r') as file:
    #  text = file.read().replace('\n', '')

    delimiter = "."
    text = [x + delimiter for x in text.split(delimiter) if x]
    tokens = []
    for line in text:
        if not line == "":
            result = tokenizer(line)
            input_ids = result['input_ids']
            tokens = tokens + tokenizer.convert_ids_to_tokens(input_ids)

    # Data to be written
    dictionary = {
        'tokens': tokens,
        'attns': array_map,
    }

    feature_dicts_with_attn = [dictionary]

    output_path = root_folder + '/processed_' + dataset_name + '/'
    save_path = output_path + 'sentence_paired_text/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pickle.dump(feature_dicts_with_attn, open(save_path + file_identifier + '_orgbert_attn.pkl', 'wb'))

    with open(save_path + file_identifier + "_sentence_paired.txt", "w") as outfile:
        outfile.write(str(feature_dicts_with_attn))

    # Serializing json
    dictionary = {
        'tokens': tokens,
    }
    json_object = json.dumps([dictionary])

    # print(json_object)
    # Writing to sample.json
    with open(save_path + file_identifier + "_orgbert_attn.json", "w") as outfile:
        outfile.write(json_object)





def preprocessing_module(root_folder,dataset_name,tokenizer,model):
    """
    """
    reading_path = root_folder + 'docsutf8/'
    processing_path = root_folder + 'processed_' + dataset_name + '/'
    if not os.path.exists(reading_path):
        print('Error, there is no reading process path')

    if os.path.exists(processing_path):
        clean_folder(processing_path + 'sentence_paired_text/')
        os.rmdir(processing_path + 'sentence_paired_text/')
        clean_folder(processing_path)
    else:
        os.makedirs(processing_path)

    files = os.listdir(reading_path)
    for fi in files:
        print('Processing file: ' + fi)
        preprocess_file(root_folder, fi,tokenizer,model,dataset_name)


'''
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = TFBertModel.from_pretrained("bert-base-uncased")

root_folder = './SemEval2017/'
dataset_name = 'SemEval2017'
preprocessing_module(root_folder)

'''

