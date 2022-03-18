import logging
import gzip
import numpy as np

def read_gz_file(filename):
    with gzip.open(filename,'rb') as infile:
        data = np.load(infile, allow_pickle=False)

    return data

def write_gz_file(data, filename):
    with gzip.open(filename, 'wb') as outfile:
        np.save(outfile, data, allow_pickle=False)


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

