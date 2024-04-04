import pickle
from .utils import clean_folder
import numpy
from transformers import BertTokenizer, TFBertModel
import json
import os
import nltk
from nltk.tokenize import sent_tokenize

#### STEP 1-4 ####.  #### PABLO VERSION



model_type=''

root_folder = './SemEval2017/'
dataset_name = 'SemEval2017'
nltk.download('punkt')
#nltk.download('spanish')

import re



def process_sentence(sentence):


    attentions,encoded_input = ModelEmb.getAttentions(sentence)



    #print(encoded_input)

    array_map = []
    for mapa in attentions:
        if model_type== 'bert':
            array_map.append(mapa)# TF
        else:
            array_map.append(mapa.detach().numpy())
    array_map = numpy.array(array_map)
    array_map = array_map[:, 0, :, :, :]

    input_ids = encoded_input['input_ids']
    tokens = ModelEmb.tokenizer.convert_ids_to_tokens(input_ids[0])

    # Data to be written
    dictionary = {
        'tokens': tokens,
        'attns': array_map,
    }

    feature_dicts_with_attn = dictionary
    return feature_dicts_with_attn


def dividir_frases(lista_frases):
    nuevas_frases = []
    for frase in lista_frases:
        tokens = frase.split()  # Dividir la frase en tokens
        if len(tokens) > 400:
            # Si la longitud de tokens supera los 400, dividir la frase en dos partes aproximadamente iguales
            mitad = len(tokens) // 2
            primera_parte = " ".join(tokens[:mitad])
            segunda_parte = " ".join(tokens[mitad:])
            # Agregar las dos nuevas frases a la lista
            nuevas_frases.append(primera_parte)
            nuevas_frases.append(segunda_parte)
        else:
            # Si la longitud de tokens no supera los 400, agregar la frase original a la lista
            nuevas_frases.append(frase)
    return nuevas_frases

lang='es'
def separate_sentences(text):

    # Tokenización de la oración en frases
    if lang=='es':
        sentences = sent_tokenize(text, language='spanish')
        sentences= dividir_frases(sentences)
    else:
        sentences = nltk.sent_tokenize(text)
    return sentences


def preprocess_file(root_folder, file_name):
    file_identifier = file_name[:-4]
    file_path = root_folder + 'docsutf8/' + file_name  # './SemEval2017/docsutf8/S0010938X1500195X.txt'
    output_path = root_folder + '/processed_' + dataset_name + '/'
    save_path = output_path + 'sentence_paired_text/'


    if os.path.exists(save_path + file_identifier + '_orgbert_attn.pkl'):
        print('already')
        return
    # READ THE FILE
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', '')

    # SEPARATE INTO SENTENCES
    sentences = separate_sentences(text)

    feature_dicts_with_attn = []

    for sentence in sentences:
        feature_dicts_with_attn_sent = process_sentence(sentence)
        feature_dicts_with_attn.append(feature_dicts_with_attn_sent)



    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pickle.dump(feature_dicts_with_attn, open(save_path + file_identifier + '_orgbert_attn.pkl', 'wb'))

    with open(save_path + file_identifier + "_sentence_paired.txt", "w") as outfile:
        outfile.write(str(feature_dicts_with_attn))

    # Serializing json

    token_sentences = []
    for features in feature_dicts_with_attn:
        token_sentences.append(features['tokens'])


    json_object = json.dumps(token_sentences)  # [dictionary]

    #print(json_object)
    # print(json_object)
    # Writing to sample.json
    with open(save_path + file_identifier + "_orgbert_attn.json", "w") as outfile:
        outfile.write(json_object)


def preprocessing_module(datasetname, bertemb, type,lan):

    global ModelEmb
    ModelEmb=bertemb

    global model_type
    model_type= type
    global lang
    lang=lan

    rootfolder = './' + datasetname + '/'
    global root_folder
    root_folder=rootfolder
    global dataset_name
    dataset_name=datasetname

    reading_path = root_folder + 'docsutf8/'
    processing_path = root_folder + 'processed_' + dataset_name + '/'
    if not os.path.exists(reading_path):
        print('Error, there is no reading process path')

    if os.path.exists(processing_path):
        clean_folder(processing_path + 'sentence_paired_text/')
        # os.rmdir(processing_path+'sentence_paired_text/')
        # clean_folder(processing_path)
    else:
        os.makedirs(processing_path)

    files = os.listdir(reading_path)
    for fi in files:
        print('Processing file: ' + fi)
        if fi.endswith('.txt'):
            print(fi)
            preprocess_file(root_folder, fi)





