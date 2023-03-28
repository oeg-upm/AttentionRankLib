
from transformers import pipeline ,AutoTokenizer ,AutoModel
from .utils import clean_folder,write_csv_file
import numpy as np
import pickle
import os
import glob
import time
import random

import torch
from torch import nn
from string import punctuation
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import sys


#import sys
#sys.path.append(os.path.abspath("/content/drive/MyDrive/AttentionRank-main"))
from bert import tokenization
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates

import spacy
import csv

#### STEP 10 ####
from nltk.corpus import stopwords
nltk.download('stopwords')

class ModelEmbedding():

    def __init__(self ,model_name):
        self.extractor = pipeline(model=model_name, task="feature-extraction")
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)

    def __call__(self, data, oov_way='avg', filter_spec_tokens=True):
        result = self.extractor(data, return_tensors=True)
        ids = self.tokenizer(data)
        lis= []
        for res ,input in zip(result ,ids['input_ids']):
            lis.append((input ,res[0].cpu().detach().numpy()))
        return self.embedding_constructor(lis)

    def embedding_constructor(self, batches, oov_way='avg', filter_spec_tokens=True):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.
        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.
        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        # for token_ids, sequence_outputs in batches:
        for batch in batches:
            tokens = []
            tensors = []
            oov_len = 1

            for token_id, sequence_output in zip(batch[0] ,batch[1]):
                token = self.tokenizer.decode(token_id)
                if token == '[PAD]':
                    # [PAD] token, sequence is finished.
                    break
                if token == '[CLS]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '[SEP]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


#### STEP 5 ####
def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding="latin1")  # add, encoding="latin1") if using python3 and downloaded data


def weights_comb(input_weights, strategy=3):
    if strategy == 1:
        comb_weight = np.array(input_weights).mean()
    elif strategy == 2:
        comb_weight = np.array(input_weights).max()
    elif strategy == 3:
        comb_weight = np.array(input_weights).sum()
    else:
        comb_weight = np.array(input_weights).prod() / np.array(input_weights).sum()
    return comb_weight


'''

def data_iterator():
    for i, doc in enumerate(data):
        # if i % 100 == 0 or i == len(data) - 1:
        # print("{:.1f}% done".format(100.0 * (i + 1) / len(data)))
        yield doc["tokens"], np.array(doc["attns"])
'''


def get_data_points(head_data):
    xs, ys, avgs = [], [], []
    for layer in range(12):
        for head in range(12):
            ys.append(head_data[layer, head])
            xs.append(1 + layer)
        avgs.append(head_data[layer].mean())
    return xs, ys, avgs


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def merge_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] = y[k] + v
        else:
            y[k] = v


def map_attn(example, heads, sentence_length, layer_weight, record):
    counter_12 = 0  # check head index

    current_sum_attn = []
    for ei, (layer, head) in enumerate(heads):
        attn = example["attns"][layer][head]  # [0:sentence_length, 0:sentence_length]
        attn = np.array(attn)
        attn /= attn.sum(axis=-1, keepdims=True)  # norm each row
        attn_sum = attn.sum(axis=0, keepdims=True)  # add up 12 heads # np.shape(attn_sum) = (1,sentence length)
        words = example["tokens"]  # [0:sentence_length]
        weights_list = attn_sum[0]

        single_word_dict = {}
        for p in range(len(words)):
            hashtag_lead = words[p]
            hash_weights = [weights_list[p]]
            weight_new = weights_comb(hash_weights, 3) * layer_weight
            single_word_dict[hashtag_lead + "_" + str(p)] = weight_new  # p is the word position in sentence

        current_sum_attn.append(np.array(list(single_word_dict.values())))  # dict.values() keep the entries order
        # shape(current_sum_attn) = (12, words number), each item in it is a list of words attn in current sentence
        counter_12 += 1  # check head index

        if counter_12 % 12 == 0:  # if head number get 12, sum all 12 heads attn and output
            head = np.sum(current_sum_attn, axis=0)  # dict zip can read array, do not need to list it
            # double check
            # print(sum([item[0] for item in current_sum_attn]))
            # print(head[0])
            longer_key_list = []
            current_key_list = list(single_word_dict.keys())  # [words_positions] # dict.keys() keep the entries order
            for each_key in current_key_list:
                longer_key_list.append(each_key + "_" + str(record))  # word_positionInSentence_sentenceNumber
            current_dict = dict(zip(longer_key_list, head))  # head = word_p_s attn, heads are already summed here

            return current_dict


def clean_id(file_name):
    return file_name[:-4]


def step_5(root_folder,dataset_name):
    start_time = time.time()

    dataset = 'SemEval2017'
    text_path = root_folder + '/docsutf8/'
    # print(text_path)
    output_path = root_folder + '/processed_' + dataset_name + '/'

    files = os.listdir(text_path)
    # for i, file in enumerate(files):
    #    files[i] = file[:-4]

    # files = files[:]

    save_path = output_path + "token_attn_paired/attn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        clean_folder(save_path)

    bert_name = "orgbert"

    print('Save Path:' + save_path)
    for n, f in enumerate(files):

        ide = clean_id(f)
        attn_extracting_dir = output_path + "sentence_paired_text/" + ide + '_' + bert_name + '_attn.pkl'
        data = load_pickle(attn_extracting_dir)
        print(save_path + ide + "token_attn_paired.csv")
        # w = csv.writer(open(save_path + file + "token_attn_paired.csv", "w"))
        rows = []
        for r in range(len(data)):
            record = r
            sentence_length = len(data[record]['tokens'])

            # consider the 12th layer
            weight = 1
            layer = 11
            sentence_dict = map_attn(data[record],
                                     [(layer, 0), (layer, 1), (layer, 2), (layer, 3),
                                      (layer, 4), (layer, 5), (layer, 6), (layer, 7),
                                      (layer, 8), (layer, 9), (layer, 10), (layer, 11)],
                                     sentence_length, weight, record)

            # print('words:', len(document_dict))

            for k, v in sentence_dict.items():
                cut = k.find(
                    "_")  # although we do not keep word position, word position helps to keep its original order
                k_short = k[0:cut]
                # w.writerow([k_short, v])
                rows.append([k_short, v])

        write_csv_file(save_path + ide + "token_attn_paired.csv", rows, 'w')
        run_time = time.time()
        print(n + 1, "th file", ide, "running time", run_time - start_time)




def step6(tokenizer,max_sequence_length,num_docs):
  start_time = time.time()
  with open('./SemEval2017/docsutf8/S0010938X1500195X.txt', 'r') as file:
      text = file.read().replace('\n', '')

  dataset = 'SemEval2017'
  text_path = './' + dataset + '/docsutf8/'
  output_path = './' + dataset + '/processed_' + dataset + '/'
  save_path = output_path + 'candidate_tokenizing/'
  if not os.path.exists(save_path):
      os.makedirs(save_path)

  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)

  tagged = []
  for token in doc:
      tagged.append((str(token), token.tag_))
  tagged = [tagged]
  print(tagged)
  text_obj = InputTextObj(tagged, 'en')
  candidates = extract_candidates(text_obj)
  #print(candidates)

  file = "S0010938X1500195X"
  #w = csv.writer(open(save_path + file + "_candidate_tokenized.csv", "w"))
  rows=[]
  # tokenize candidates
  raw_lines = []
  current_doc_tokens = []
  df_dict = {}
  for l, line in enumerate(candidates):
      raw_lines.append(line)
      print('---')
      print(line)
      line = tokenization.convert_to_unicode(line).strip()
      print(line)
      segments=[]
      if not line:  # line is empty, deal the 2 segments in the [current doc tokens]
          if current_doc_tokens:
              #REVISAR ESTO for segment in prep_document(current_doc_tokens, args.max_sequence_length):
              for segment in prep_document(current_doc_tokens, max_sequence_length):
                  if (len(segment) - 1) / 2 != segment.index("[SEP]"):
                      cline = raw_lines[l - 1].replace("),", "),$$$$$").split("$$$$$")
                  segments.append(segment)
                  if len(segments) >= num_docs:
                      break
              if len(segments) >= num_docs:
                  break

          current_doc_tokens = []  # clean up [current doc tokens]

      tokens = tokenizer.tokenize(line)
      rows.append([line, tokens])

      if tokens:  # if line is not empty, add lines to [current doc tokens]
          current_doc_tokens.append(tokens)

  write_csv_file(save_path + file + '_candidate_tokenized.csv',rows)

  # get tf
  tf_dict = {}
  for item in candidates:
      if item in tf_dict:
          tf_dict[item] += 1
      else:
          tf_dict[item] = 1

  #w0 = csv.writer(open(save_path + file + '_candidate_tf.csv', "a"))# get df
  rows=[]
  for k, v in sorted(tf_dict.items(), key=lambda item: item[1], reverse=True):
      #w0.writerow([k,v])
      rows.append([k, v])
      if k in df_dict:
          df_dict[k] += 1
      else:
          df_dict[k] = 1

  write_csv_file(save_path + file + '_candidate_tf.csv',rows)

  run_time = time.time()
  print(l, "th file", file, "running time", run_time - start_time)

  #w1 = csv.writer(open(save_path + file + '_candidate_df.csv', "a"))
  rows=[]
  for k, v in sorted(df_dict.items(), key=lambda item:item[1], reverse=True):
    #w1.writerow([k, v])
    rows.append([k, v])
  write_csv_file(save_path + file + '_candidate_df.csv',rows)


#### STEP 7 ####
"""pair candidates and their accumulated self-attention"""
def step7():
    dataset = 'SemEval2017'
    doc_path = './' + dataset + '/' + 'docsutf8/'
    output_path = './' + dataset + '/processed_' + dataset + '/'
    candidate_token_path = output_path + 'candidate_tokenizing/'
    token_attn_path = output_path + 'token_attn_paired/attn/'

    save_path = output_path + 'candidate_attn_paired/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(doc_path)
    for i, file in enumerate(files):
        files[i] = file[:-4]

    files = files[:]

    start_time = time.time()

    for n, file in enumerate(files):
        print(file)

        # read token attn to list
        token_list = []
        attn_list = []
        with open(token_attn_path + file + "token_attn_paired.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                k = row[0]
                v = float(row[1])
                token_list.append(k)
                attn_list.append(v)

        # read candidate tokens to dict
        candidate_token_dict = {}
        print(candidate_token_path + file + "_candidate_tokenized.csv")
        with open(candidate_token_path + file + "_candidate_tokenized.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                k = row[0]
                v = row[1][2:-2].split("', '")
                candidate_token_dict[k] = v
                print(k)

        # candidate attn pairing
        candidate_attn_dict = {}
        for k, v in candidate_token_dict.items():
            window = len(v)
            matched = []
            for t, token in enumerate(token_list):
                if token_list[t:t + window] == v:
                    local_attn = sum(attn_list[t:t + window])
                    if k in candidate_attn_dict.keys():
                        candidate_attn_dict[k] += local_attn
                    else:
                        candidate_attn_dict[k] = local_attn
        print(candidate_attn_dict)

        # w = csv.writer(open(save_path + file + "_attn_paired.csv", "w"))
        rows = []
        for k, v in candidate_attn_dict.items():
            # w.writerow([k, v])
            rows.append([k, v])

        write_csv_file(save_path + file + "_attn_paired.csv", rows)
        run_time = time.time()
        print(n, "th file", file, "running time", run_time - start_time)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def prep_document(document, max_sequence_length):
    """Does BERT-style preprocessing on the provided document."""
    max_num_tokens = max_sequence_length - 3
    target_seq_length = max_num_tokens

    # We DON"T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, random)

                if len(tokens_a) == 0 or len(tokens_b) == 0:
                    break
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                tokens.append("[CLS]")
                for token in tokens_a:
                    tokens.append(token)
                tokens.append("[SEP]")

                for token in tokens_b:
                    tokens.append(token)
                tokens.append("[SEP]")

                instances.append(tokens)

            current_chunk = []
            current_length = 0
        i += 1
    # print(instances)
    return instances

#step7()



#### STEP 8 ####
#from bert_embedding import BertEmbedding
def step8(bertemb):
  start_time = time.time()

  dataset = 'SemEval2017'
  text_path = './' + dataset + '/docsutf8/'
  output_path = './' + dataset + '/processed_' + dataset + '/'

  # set save path for embeddings
  save_path = output_path + 'candidate_embedding/'
  if not os.path.exists(save_path):
      os.makedirs(save_path)

  # set save path for document frequency dictionary
  df_dict = {}
  save_path_tfdf = output_path + 'df_dict/'
  if not os.path.exists(save_path_tfdf):
      os.makedirs(save_path_tfdf)

  # load BERT embedding generator
  #ctx = mx.cpu(0)
  #bert = BertEmbedding(ctx=ctx, max_seq_length=512, batch_size=4)


  # read files name
  #files = glob.glob(os.path.join(text_path, '*'))
  #for i, file in enumerate(files):
  #    files[i] = file[:-4]


  files = os.listdir(text_path)
  for i, file in enumerate(files):
      files[i] = file[:-4]

  print(dataset, 'docs:', len(files))


  # run all files
  for n, file in enumerate(files):
      text = ''
      my_file = text_path + file + '.txt' #text_path +
      print(my_file)
      with open(my_file, "r") as f:
          for line in f:
              if line:
                  print(text)
                  text += line
      print(text)
      text = text.replace('$$$$$$', ' ')
      print(text)
      nlp = spacy.load("en_core_web_sm")
      doc = nlp(text)

      tagged = []
      for token in doc:
          tagged.append((str(token), token.tag_))
      tagged = [tagged]
      print(tagged)
      text_obj = InputTextObj(tagged, 'en')
      candidates = extract_candidates(text_obj)
      print("hola", candidates)
      rows=[]
      #w1 = csv.writer(open(save_path + file + '_candidate_embedding.csv', "a"))

      # get raw embedding
      candidates_with_embeddings = bertemb(candidates) #embeddign_generation('bert-base-uncased',candidates)#bert(candidates)  # this bert handle [list of candidates]
      for c, can_with_word_embed in enumerate(candidates_with_embeddings):

          can_words = candidates[c]  # important
          can_word_raw_embeddings = can_with_word_embed[1]
          #w1.writerow([can_words, can_word_raw_embeddings])
          rows.append([can_words, can_word_raw_embeddings])

      write_csv_file(save_path + file + '_candidate_embedding.csv',rows)
      # get df
      tf_dict = {}

      for item in candidates:
          item = item.lower()
          if item in tf_dict.keys():
              tf_dict[item] += 1
          else:
              tf_dict[item] = 1

      for k, v in sorted(tf_dict.items(), key=lambda item: item[1], reverse=True):
          if k in df_dict.keys():
              df_dict[k] += 1
          else:
              df_dict[k] = 1

      crt_time = time.time()
      print(dataset, n + 1, "th file", file, "running time", crt_time - start_time)

  print(tf_dict)

  # save df dictionary
  '''
  w1 = csv.writer(open(save_path_tfdf + dataset + '_candidate_df.csv', "a"))
  for k, v in sorted(df_dict.items(), key=lambda item:item[1], reverse=True):
      w1.writerow([k, v])

  '''
  '''
  with open(save_path_tfdf + dataset + '_candidate_df.csv', 'a') as outfile:
      for k, v in sorted(df_dict.items(), key=lambda item:item[1], reverse=True):
        print(k)
        w1.writerow([k, v])
  '''
  rows=[]
  #f = open(save_path_tfdf + dataset + '_candidate_df.csv', 'w')
  #writer = csv.writer(f)
  for k, v in sorted(df_dict.items(), key=lambda item:item[1], reverse=True):
    print(k)
    print(v)
    #writer.writerow([k, v])
    rows.append([k, v])
    #f.flush()
  #f.close()

  write_csv_file(save_path_tfdf + dataset + '_candidate_df.csv',rows)

#step8(bertemb)




#### STEP 9 ####

"""
get document word embedding by sentences
"""

def step9(bertemb):
  start_time = time.time()

  problem_files = []

  dataset = 'SemEval2017'
  text_path = './' + dataset + '/docsutf8/'
  output_path = './' + dataset + '/processed_' + dataset + '/'


  files = os.listdir(text_path)
  for i, file in enumerate(files):
      files[i] = file[:-4]
  files = files[:]
  print('docs:', len(files), files)

  for n, file in enumerate(files):
      print(file)
      fp = open(text_path + file + '.txt')
      # print("hola", fp.read().split('$$$$$$'))
      #print(fp.read())
      #sentences = [a for a in fp.read().split('$$$$$$')]
      sentences = fp.read().split('$$$$$$')
      print("adio", sentences)
      save_path = output_path + 'doc_word_embedding_by_sentences/' + file + '/'
      if not os.path.exists(save_path):
          os.makedirs(save_path)

      sentences_with_embeddings = bertemb(sentences)#embeddign_generation('bert-base-uncased',sentences)#bert(sentences)  # this bert handle [list of sentences]

      for l, sentence_with_embeddings in enumerate(sentences_with_embeddings):
          words = sentence_with_embeddings[0]
          print(words)
          embeddings = sentence_with_embeddings[1]
          print(save_path + file + '_sentence' + str(l) + '_word_embeddings.csv')
          w0 = csv.writer(open(save_path + file + '_sentence' + str(l) + '_word_embeddings.csv', "a"))
          for i in range(len(words)):
              w0.writerow([words[i], embeddings[i]])

      print(n + 1, "th file", file, "running time", time.time() - start_time)

#step9(bertemb)



#### STEP 10 ####
def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # [0, 1]


def self_attn_matrix(embedding_set):
    ls = np.shape(embedding_set)[0]
    Q = torch.tensor(embedding_set)
    K = torch.tensor(embedding_set)
    V = torch.tensor(embedding_set)
    attn = torch.matmul(Q, K.transpose(-1,-2))
    attn = nn.Softmax(dim=1)(attn)
    V = torch.matmul(attn, V)
    V = torch.sum(V, dim=0)
    V = V/ls
    return V

def cross_attn_matrix(D,Q):
    D = torch.tensor(D)
    Q = torch.tensor(Q)
    attn = torch.matmul(D, Q.transpose(-1,-2))
    S_d2q = nn.Softmax(dim=1)(attn)  # S_d2q : softmax the row; shape[len(doc), len(query)]
    S_q2d = nn.Softmax(dim=0)(attn)  # S_q2d : softmax the col; shape[len(doc), len(query)]
    A_d2q = torch.matmul(S_d2q, Q)
    A_q2d = torch.matmul(S_d2q, torch.matmul(S_q2d.transpose(-1,-2), D))
    V = (D+A_d2q+torch.mul(D, A_d2q)+torch.mul(D, A_q2d))
    V = V/4
    return V

def f1(a, b):
    return a*b*2/(a+b)


def mean_f_p_r(actual, predicted, best=10, pr_plot=False):
    list_f1 = []
    list_p = []
    list_r = []
    for r in range(len(actual)):
        y_actual = actual[r]
        y_predicted = predicted[r][:best]
        y_score = 0
        for p, prediction in enumerate(y_predicted):
            if prediction in y_actual and prediction not in y_predicted[:p]:
                y_score += 1
        if not y_predicted:
            y_p = 0
            y_r = 0
        else:
            y_p = y_score / len(y_predicted)
            y_r = y_score / len(y_actual)
        if y_p != 0 and y_r != 0:
            y_f1 = 2 * (y_p * y_r / (y_p + y_r))
        else:
            y_f1 = 0
        list_f1.append(y_f1)
        list_p.append(y_p)
        list_r.append(y_r)
    if pr_plot:
        return list_f1, list_p, list_r
    else:
        return np.mean(list_f1), np.mean(list_p), np.mean(list_r)


def step10():
  punctuations = []
  for punc in range(len(punctuation)):
      punctuations.append(punctuation[punc])
  stop_words_list = stopwords.words('english')
  maxInt = sys.maxsize

  while True:
      # decrease the maxInt value by factor 10
      # as long as the OverflowError occurs.
      try:
          csv.field_size_limit(maxInt)
          break
      except OverflowError:
          maxInt = int(maxInt/10)


  start_time = time.time()

  dataset = 'SemEval2017'

  text_path = './' + dataset + '/docsutf8/'
  output_path = './' + dataset + '/processed_' + dataset+ '/'
  save_path = output_path + 'candidate_cross_attn_value/'
  if not os.path.exists(save_path):
      os.makedirs(save_path)

  # get files name
  files = os.listdir(text_path)
  for i, file in enumerate(files):
      files[i] = file[:-4]


  # run all files
  for n, file in enumerate(files):

      # doc embedding set
      embedding_path = output_path + 'doc_word_embedding_by_sentences/' + file + '/'
      sentence_files = os.listdir(embedding_path)  # get sentence list
      print(sentence_files)

      all_sentences_word_embedding = []
      for sentence_file in sentence_files:  # do not need to sort
          sentence_word_embedding = []
          print(embedding_path + sentence_file)
          with open(embedding_path + sentence_file, newline='') as csvfile:
              spamreader = csv.reader(csvfile, delimiter=',')
              for row in spamreader:
                  # print(row[1])
                  value_list = row[1][1:-1]
                  if value_list[0] == ' ':
                      value_list = value_list[1:]
                  if value_list[-1] == ' ':
                      value_list = value_list[:-1]
                  value_list = value_list.replace('\n', '').replace('  ', ' ').replace('  ', ' ').split(' ')
                  #print(sentence_file, value_list)
                  k = row[0]
                  if k not in stop_words_list + punctuations:
                      temp_list = []
                      for item in value_list:
                        if item != '':
                          temp_list.append(float(item))
                      v = np.array(temp_list)
                      sentence_word_embedding.append(v)
          all_sentences_word_embedding.append(sentence_word_embedding)

      print(all_sentences_word_embedding)
      print('----')
      # print(np.shape(all_sentences_word_embedding))  # sentence number ex. 19
      # print(np.shape(all_sentences_word_embedding[0]))  # sentence 0 words number ex. (51,768)

      # get querys embeddings path
      querys_name_set = []
      querys_embedding_set = []
      querys_embeddings_path = output_path + 'candidate_embedding/'
      with open(querys_embeddings_path + file + "_candidate_embedding.csv", newline='') as csvfile:
          spamreader = csv.reader(csvfile, delimiter=',')
          for row in spamreader:
              k = row[0]
              v = row[1].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
              v = v.replace(', dtype=float32', '')[9:-3].split(']), array([')
              candidate_embeddings_set = []
              for l in range(len(v)):
                  candidate_embeddings_set.append(np.array([float(item) for item in v[l].split(', ')]))
              querys_name_set.append(k)
              querys_embedding_set.append(candidate_embeddings_set)
              #print(k, candidate_embeddings_set)

      # print(len(querys_embedding_set))  # 29 querys embeddings
      # print(len(querys_embedding_set[0]))  # how many embeddings in the query0

      # main
      ranking_dict = {}
      for w in tqdm(range(len(querys_embedding_set))):
          query_inner_attn = self_attn_matrix(querys_embedding_set[w])  # shape = len(query words)*786

          sentence_embedding_set = []
          for sentence_word_embeddings in all_sentences_word_embedding:  # ex. (19, n, 768)
              print(sentence_word_embeddings)
              cross_attn = cross_attn_matrix(sentence_word_embeddings, querys_embedding_set[w])
              sentence_embedding = self_attn_matrix(cross_attn)  # shape = (n, 768)
              sentence_embedding_set.append(sentence_embedding)  # shape = (1, 768)

          doc_inner_attn = torch.stack(sentence_embedding_set, dim=0)  # shape = (19, 768)
          doc_inner_attn = self_attn_matrix(doc_inner_attn)  # shape = (1, 768)
          output = cosine_similarity(query_inner_attn.cpu().numpy(), doc_inner_attn.cpu().numpy())
          ranking_dict[querys_name_set[w]] = float(output)
      print(ranking_dict)
      w0 = csv.writer(open(save_path + file + '_candidate_cross_attn_value.csv', "a"))
      for k, v in sorted(ranking_dict.items(), key=lambda item:item[1], reverse=True):
          # print(k,v)
          w0.writerow([k,v])
      crt_time = time.time()
      print(n + 1, "th file", "running time", crt_time - start_time)

#step10()


#### STEP 11 ####
def f1(a, b):
    return a * b * 2 / (a + b)


def mean_f_p_r(actual, predicted, best=10, pr_plot=False):
    list_f1 = []
    list_p = []
    list_r = []
    for r in range(len(actual)):
        y_actual = actual[r]
        y_predicted = predicted[r][:best]
        y_score = 0
        for p, prediction in enumerate(y_predicted):
            if prediction in y_actual and prediction not in y_predicted[:p]:
                y_score += 1
        if not y_predicted:
            y_p = 0
            y_r = 0
        else:
            y_p = y_score / len(y_predicted)
            y_r = y_score / len(y_actual)
        if y_p != 0 and y_r != 0:
            y_f1 = 2 * (y_p * y_r / (y_p + y_r))
        else:
            y_f1 = 0
        list_f1.append(y_f1)
        list_p.append(y_p)
        list_r.append(y_r)
    if pr_plot:
        return list_f1, list_p, list_r
    else:
        return np.mean(list_f1), np.mean(list_p), np.mean(list_r)


def step11():
    # load stopwords list
    mystopwords = []
    my_file = './UGIR_stopwords.txt'
    with open(my_file, "r") as f:
        for line in f:
            if line:
                mystopwords.append(line.replace('\n', ''))

    dataset = 'SemEval2017'
    text_path = "./" + dataset + '/docsutf8/'
    output_path = "./" + dataset + '/processed_' + dataset + '/'
    accumulated_self_attn_path = output_path + 'candidate_attn_paired/'

    # get files name
    files = os.listdir(text_path)
    for i, file in enumerate(files):
        files[i] = file[:-4]

    # load candidate df
    df_dict = {}
    df_path = output_path + 'df_dict/'
    with open(df_path + dataset + "_candidate_df.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            k = row[0].lower()
            v = float(row[1])
            df_dict[k] = v
    print('-----')
    print(df_dict)
    # collect
    f1_top = 10
    predicted = []
    actual = []

    start_time = time.time()

    for n, file in enumerate(files):
        print('file', file, '\n')

        # load cross attn dict
        cross_attn_dict_first = {}
        cross_attn_dict_path = output_path + 'candidate_cross_attn_value/'
        tail = "_candidate_cross_attn_value.csv"
        print(cross_attn_dict_path + file + tail)
        with open(cross_attn_dict_path + file + tail, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                k = row[0].lower()
                if k.find('.') == -1:
                    print(df_dict)
                    df = df_dict[k]
                    if df_dict[k]:  # < 44:
                        v = float(row[1])
                        cross_attn_dict_first[k] = v
        print('***')
        print(cross_attn_dict_first)
        cross_attn_dict = {}
        for k, v in cross_attn_dict_first.items():
            if k[-1] == "s" and k[:-1] in cross_attn_dict:
                cross_attn_dict[k[:-1]] = max(v, cross_attn_dict[k[:-1]])
            elif k + 's' in cross_attn_dict:
                cross_attn_dict[k] = max(v, cross_attn_dict[k + 's'])
                cross_attn_dict.pop(k + 's')
            else:
                cross_attn_dict[k] = v

        # norm cross attn dict
        print('-------')
        print(cross_attn_dict)
        a0 = min(cross_attn_dict.values())
        b0 = max(cross_attn_dict.values())
        for k, v in cross_attn_dict.items():
            cross_attn_dict[k] = (v - a0) / (b0 - a0)

        # load accumulated self attn ranking
        accumulated_self_attn_dict_first = {}
        tail0 = "_attn_paired.csv"

        with open(accumulated_self_attn_path + file + tail0, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                k = row[0].lower()
                if k.find('.') == -1:
                    v = float(row[1])  # /len(k.split(' '))
                    accumulated_self_attn_dict_first[k] = v

        accumulated_self_attn_dict = {}
        for k, v in accumulated_self_attn_dict_first.items():
            if k[-1] == "s" and k[:-1] in accumulated_self_attn_dict:
                accumulated_self_attn_dict[k[:-1]] = v + accumulated_self_attn_dict[k[:-1]]
            elif k + 's' in accumulated_self_attn_dict:
                accumulated_self_attn_dict[k] = v + accumulated_self_attn_dict[k + 's']
                accumulated_self_attn_dict.pop(k + 's')
            else:
                accumulated_self_attn_dict[k] = v
        print('--->')
        print(accumulated_self_attn_dict)

        # norm attn-candidate dict
        t = 8
        ranking_dict = {}
        a1 = min(accumulated_self_attn_dict.values())
        b1 = max(accumulated_self_attn_dict.values())
        for k, v in accumulated_self_attn_dict.items():
            if k in cross_attn_dict.keys() and k.split(' ')[0] not in mystopwords:
                accumulated_self_attn_dict[k] = (v - a1) / (b1 - a1)
                ranking_dict[k] = accumulated_self_attn_dict[k] * (t) / 10 + cross_attn_dict[k] * (10 - t) / 10

        f1_k = 0
        print('Prediction:')
        pred_single = []
        for k, v in sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True):
            if f1_k < f1_top:
                print(k, v)
                pred_single.append(k)
                f1_k += 1

        # load keys
        label_path = './' + dataset + '/keys/'
        my_key = label_path + file + '.key'
        print('\n Truth keys:')
        actual_single = []
        with open(my_key, "r") as f:
            for line in f:
                k = line.replace("\n", '').replace("  ", " ").replace("  ", " ")
                for i in range(99):
                    if k[-1] == " ":
                        k = k[:-1]
                    else:
                        break
                print(k)
                actual_single.append(k.lower())

        actual_single = list(set(actual_single))

        actual.append(actual_single)
        predicted.append(pred_single)

    mean_f1, mean_p, mean_r = mean_f_p_r(actual, predicted, f1_top)
    straight_f1 = f1(mean_p, mean_r)
    print(mean_p, mean_r, straight_f1, mean_f1)
    print('ssdddasdsdsd222')


