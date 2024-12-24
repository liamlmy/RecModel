# coding:utf-8

import sys
import tensorflow as tf
import numpy as np
import datetime
import time
import yaml
import gc
from scipy.sparse import csr_matrix

def auc(lis, scale=1e6):
    def add(d, k, n):
        if k in d:
            d[k] += n
        else:
            d[k] = n

    s = {}
    c = {}
    for x in lis:
        label = 1 if x[0] > 0 else 0
        score = x[1]
        q = int(score * scale)
        add(s, q, 1)
        add(c, q, label)

    k = sorted(s.keys(), reverse=True)
    num_pos = 0
    num_neg = 0
    w = 0.0
    for q in k:
        pos = c[q]
        neg = s[q] - c[q]
        w = w + num_pos * neg + 0.5 * pos * neg
        num_pos += pos
        num_neg += neg
    if num_pos == 0 or num_neg == 0:
        return 0
    return w / (num_pos * num_neg)


def load_yml_file(yml_file):
    f = open(yml_file)
    ss = f.read()
    f.close()
    x = yaml.load(ss, Loader=yaml.FullLoader)
    return x


def save_yml_file(x, yml_file):
    fw = open(yml_file, 'w')
    yaml.dump(x, fw)
    fw.close()


def parse_line(line):
    lis = line.split('#', 1)[0].strip(' \n').split(' ')
    label = int(lis[0])
    fea_index = []
    fea_value = []
    for i, x in enumerate(lis[1:]):
        y = x.split(':')
        ind = int(y[0])
        val = float(y[1])
        fea_index.append(ind)
        fea_value.append(val)
    return fea_index, fea_value, label


def parse_record(value):
    features = {
        "label": tf.io.FixedLenFeature([], tf.float32),
        "fea_ids": tf.io.VarLenFeature(tf.int64),
        "fea_vals": tf.io.VarLenFeature(tf.float64)
    }
    parsed =tf.io.parse_single_example(value, features)
    return parsed


def WriteTFRecord(fs, outfile_name):
    writer = tf.io.TFRecordWriter(outfile_name)
    for line in fs:
        fea_index, fea_value, label = parse_line(line)
        tfrecord_feature = {
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "fea_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=fea_index)),
            "fea_vals": tf.train.Feature(float_list=tf.train.FloatList(value=fea_value))
        }
        example = tf.train.Example(features=tf.train.Featrues(feature=tfrecord_feature))
        writer.write(example.SerializeToString())
    writer.close()


def ReadTFRecord(files, shuffle_size=1, batch_size=1, fetch_size=1, num_parallel=1):
    def parse_record_batch(value):
        if isinstance(value, list):
            return map(parse_record, value)
        return parse_record(value)

    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)
    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset

