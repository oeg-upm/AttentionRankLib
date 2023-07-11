import csv
import numpy as np
import os
import time


def read_term_list_file(filepath):
    lst=[]
    with open(filepath, "r",encoding='utf-8') as f:
        for line in f:
            k = line.replace("\n", '').replace("  ", " ").replace("  ", " ")
            k=k.strip()
            lst.append(k.lower())
    return lst

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


def get_file_ids(text_path):
    # get files name
    files = os.listdir(text_path)
    for i, file in enumerate(files):
        files[i] = file[:-4]
    return files

def evaluate_results(datasetpath,datasetname):
    # load stopwords list
    my_file = './src/attentionrank/UGIR_stopwords.txt'
    #dataset = 'SemEval2017'
    #text_path = "./" + dataset + '/docsutf8/'
    #output_path = "./" + dataset + '/processed_' + dataset + '/'

    dataset=datasetname
    text_path =   datasetpath + '/docsutf8/'
    output_path = datasetpath+ '/processed_' + dataset + '/'

    accumulated_self_attn_path = output_path + 'candidate_attn_paired/'



    files= get_file_ids(text_path)
    print('Files to process:', len(files))

    mystopwords = read_term_list_file(my_file)


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
            print('term',k)
            if k in cross_attn_dict.keys() and k.split(' ')[0] not in mystopwords:
                print('passterm', k)
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
        actual_single = read_term_list_file(my_key)
        actual_single = list(set(actual_single))


        actual.append(actual_single)
        predicted.append(pred_single)

    print('Len actual',len(actual),len(predicted))
    mean_f1, mean_p, mean_r = mean_f_p_r(actual, predicted, f1_top)
    straight_f1 = f1(mean_p, mean_r)
    print('Precission, recall, f1, mean_f1')
    print(mean_p, mean_r, straight_f1, mean_f1)



