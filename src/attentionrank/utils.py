import glob
import os
import csv
import six
import json
import pickle
import time

import tensorflow as tf


def get_files_ids(files):
    for i, file in enumerate(files):
        files[i] = file[:-4]
    return files


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
def write_list_file(filepath, list, mode='w'):
    # content is list of lists
    # w = csv.writer(open(file_name, "w"))
    print('Writting file: ' + filepath)
    with open(filepath, mode) as archivo:
        for element in list:
            archivo.write(str(element) + '\n')



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif str(type(text)) == "<type 'unicode'>":  # isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def write_json(o, path):
    tf.io.gfile.makedirs(path.rsplit('/', 1)[0])
    with tf.io.gfile.GFile(path, 'w') as f:
        json.dump(o, f)


def load_pickle(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(o, path):
    if '/' in path:
        tf.io.gfile.makedirs(path.rsplit('/', 1)[0])
        print(path)
    with tf.io.gfile.GFile(path, 'wb') as f:
        pickle.dump(o, f, -1)


def logged_loop(iterable, n=None, **kwargs):
    if n is None:
        n = len(iterable)
    ll = LoopLogger(n, **kwargs)
    for i, elem in enumerate(iterable):
        ll.update(i + 1)
        yield elem


class LoopLogger(object):
    """Class for printing out progress/ETA for a loop."""

    def __init__(self, max_value=None, step_size=1, n_steps=25, print_time=True):
        self.max_value = max_value
        if n_steps is not None:
            self.step_size = max(1, max_value // n_steps)
        else:
            self.step_size = step_size
        self.print_time = print_time
        self.n = 0
        self.start_time = time.time()

    def step(self, values=None):
        self.update(self.n + 1, values)

    def update(self, i, values=None):
        self.n = i
        if self.n % self.step_size == 0 or self.n == self.max_value:
            if self.max_value is None:
                msg = 'On item ' + str(self.n)
            else:
                msg = '{:}/{:} = {:.1f}%'.format(self.n, self.max_value,
                                                 100.0 * self.n / self.max_value)
                if self.print_time:
                    time_elapsed = time.time() - self.start_time
                    time_per_step = time_elapsed / self.n
                    msg += ', ELAPSED: {:.1f}s'.format(time_elapsed)
                    msg += ', ETA: {:.1f}s'.format((self.max_value - self.n)
                                                   * time_per_step)
            if values is not None:
                for k, v in values:
                    msg += ' - ' + str(k) + ': ' + ('{:.4f}'.format(v)
                                                    if isinstance(v, float) else str(v))
            print(msg)
