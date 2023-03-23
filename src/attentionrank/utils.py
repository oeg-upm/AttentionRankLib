
import glob
import os
import csv

def clean_folder(path):
    files = glob.glob(path + '*')  # /YOUR/PATH/
    for f in files:
        os.remove(f)

def write_csv_file(file_name, content, mode='w'):
    # content is list of lists
    # w = csv.writer(open(file_name, "w"))
    print('Writting file: ' + file_name)
    with open(file_name, mode) as myfile:
        wrtr = csv.writer(myfile)  # , delimiter=',', quotechar='"'
        for row in content:
            wrtr.writerow([row[0], row[1]])
            myfile.flush()  # whenever you want

    print('-')

