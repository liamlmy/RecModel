import sys,os
import argparse
import copy
import numpy as np
import tensorflow as tf
import utils as ut

from DSSM import DSSM
from DeepFM import DeepFM
from XDeepFM import XDeepFM
from DIN import DIN
from DCN import DCN

class Learner:
    def __init__(self, conf_yml, conf=None):
        self.conf_yml = conf_yml
        self.model = None
        if conf is None:
            self.conf = ut.load_yaml_file(conf_yml)
        else:
           self.conf = conf

    def get_feature(self):
        return fea_size, fea_slot, num_fea, dense_fea, multi_fea, sess_fea

    def get_fea_size(self):
        x = ''
        fea_size_file = self.conf["common"]["fea_size_file"]
        with open(fea_size_file) as f:
            x = f.read()
        return int(x.strip())

    def get_fea_slot(self):
        pass

    def get_session_feaslot(self):
        pass

    def get_emb_feaslot(self):
        pass

    def get_multihot_feaslot(self):
        pass

    def get_padding_size(self, fea_slot, multihout_feaslot, fix_addition=0):
        pass

    def set_training_model(self, enable_training):
        pass

    def make_config(self):
        pass

    def init(self, training=False):
        if self.model is not None:
            return
        model_class = self.conf["common"]["model_class"]
        x = globals()
        if model_class not in x:
            sys.stderr.write("model class not find: [%s]\n"%(model_class))
            return
        if training:
            self.conf["model_params"]["training"] = True
        self.model = x[model_class](**self.conf["model_params"])

    def prepare_dataset(self, ds):
        com_conf = self.conf["common"]
        batch_size = com_conf["batch_size"]
        shuffle = com_conf["shuffle"]
        epoch_num = com_conf["epoch_num"]
        if shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 3)
        ds = ds.batch(batch_size).prefetch(batch_size)
        return ds

    def train(self, data):
        pass

    def test(self, data, model_path=None):
        pass

    def predict(self, X, model_path=None):
        pass

    def dump_serving_model(self, model_path=None):
        if self.model is None:
            self.init()
        self.model.is_save_model = True
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=["mae"]) # check
        conf = self.conf["common"]
        self.model._set_inputs(tf.keras.Input(shape=(2 * conf["padding_size"], ), dtype=tf.dtypes.float32))
        self.model.lad_weights(args.model)
        self.model.save(conf["local_model_dir"] + "/tfmodel", save_format="tf")

    def batch_test(self, infs, model_path, batch_size=1024, ofs=None, is_test=True, output_infos=False):
        pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='train&evaluate the DeepFM model')
    parse.add_argument('-method', type=str, help='-method:train|test|auto_config', required=True)
    parse.add_argument('-conf', type=str, help='yml conf')
    parse.add_argument('-conf_models', type=str, help='conf1,model1,model2...;conf2,model1,model2...')
    parse.add_argument('-model', type=str, help='init model dir')
    parse.add_argument('-data', type=str, help='input data files')
    args = parse.parse_args()
    tf.keras.backend.set_floatx('float64')
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

    if args.conf:
        x = ut.load_yaml_file(args.conf)
        mc = x["common"]["model_class"]
        mcif = mc + "Learner"
        z = globals()
        solver = None
    if mcif in z:
        solver = z[mcif](args.conf, x)
    else:
        solver = Learner(args.conf, x)
        print("no model interface {} find".format(mcif), file=sys.stderr)
    os.environ["CUDA_VISIBLE_DEVICES"] = x["common"]["CUDA_VISIBLE_DEVICES"]

    print("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    gpus = if.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.method == "train":
        info = {}
        com_conf = x["common"]
        batch_size = com_conf["batch_size"]
        shuffle_size = batch_size * 2
        preprocess_conf = com_conf["preprocess_conf"]
        ds = ut.ReadTextLineDatasetOnline(args.data, conf_file=preprocess_conf, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=1, num_parallel=10)
    elif args.method == "test":
        pass
    elif args.method == "test_xxx":
        pass
    elif args.method == "test_diff_nets":
        pass
    elif args.method == "predict":
        pass
    elif args.method == "predict_info":
        pass
    elif args.method == "dump_serving":
        solver.dump_serving_model(args.model)
    elif args.method == "auto_config":
        solver.make_config()
    else:
        print("method type error [{}] please check".format(args.method), file=sys.stderr)