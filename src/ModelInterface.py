# coding:utf-8
"""
Author:
    Mingyang Li: liamlmy@gmail.com
"""

import sys
import os
import argparse
import copy
import numpy as np
import tensorflow as tf
import utils as ut

#from DSSM import DSSM
#from WideDepp import WideDeep
from DeepFM import DeepFM
#from XDeepFM import XDeepFM
#from DIN import DIN
#from DCN import DCN
#from SENet import SENet
#from ESSM import ESSM
#from MMOE import MMOE
#from CGC import CGC
#from PLE import PLE
#from PEPNet import PEPNet

class Learner:
    """
    Class Learner is the basic class for model training, test etc.
    Attributes
        model:
        conf_yml:
        conf:
    """
    def __init__(self, conf_yml, conf=None):
        """init"""
        self.conf_yml = conf_yml
        self.model = None
        if conf is None:
            self.conf = ut.load_yaml_file(conf_yml)
        else:
           self.conf = conf

    def get_feature(self):
        """
        get feature
        Args:
        Returns:
            fea_slot: a map of feature which key is feature name and value is [start index, end index] of feature slot
            slot_size: the size of feature slot
            fea_map: a map for different special feature which key is type of special feature and value is the list of slot
        Raises:
        """
        fea_slot, slot_size = self.get_fea_slot()
        fea_map = {}
        fea_map["num_fea"] = self.get_target_fea_slot("num_fea_conf", fea_slot)
        fea_map["dense_fea"] = self.get_target_fea_slot("dense_fea_conf", fea_slot)
        fea_map["multi_fea"] = self.get_target_fea_slot("multi_fea_conf", fea_slot)
        fea_map["seq_fea"] = self.get_target_fea_slot("seq_fea_conf", fea_slot)
        return fea_slot, slot_size, fea_map

    def get_fea_slot(self):
        """
        get feature
        Args:
        Returns:
            fea_slot: a map of feature which key is feature name and value is [start index, end index] of feature slot
            slot_size: the size of feature slot
        Raises:
        """
        feat_slot = {}
        feat_size = 0
        fea_slot_file = self.conf["feature"]["fea_slot_file"]
        if fea_slot_file is None:
            print("fea_slot file is error, please check", file=sys.stderr)
        with open(fea_slot_file) as f:
            for line in f:
                spline = line.strip().split('\t')
                fea_name = spline[0]
                start = int(spline[1])
                length = int(spline[2])
                feat_size = start + length
                feat_slot[fea_name] = (start, length)
        return feat_slot, feat_size

    def get_target_feaslot(self, target_fea, fea_slot=None):
        """
        get feature
        Args:
            target_fea: special type feature
            fea_slot: a map of feature which key is feature name and value is [start index, end index] of feature slot
        Returns:
            res: slot list of target_fea
        Raises:
        """
        if fea_slot is None:
            fea_slot, _ = self.get_fea_slot()
        target_fea_conf = self.conf["feature"][target_fea]
        res = []
        if target_fea_conf is None or target_fea_conf == '':
            return res
        with open(target_fea_conf) as f:
            for line in f:
                k = line.strip()
                if k not in fea_slot:
                    continue
                start = fea_slot[k][0]
                end = start + fea_slot[k][1]
                res.append([start, end])
        return res

    def get_padding_size(self, fea_slot, multihout_feaslot, fix_addition=0):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        fea_size = len(fea_slot)
        if fix_addition > 0:
            return fea_size + fix_addition
        for m in multi_fea_slot:
            fea_size = fea_size + m[1] - m[0] - 1
        return fea_size

    def set_training_model(self, enable_training):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        self.model.training = enable_training

    def make_config(self):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        pass

    def init(self, training=False):
        """
        get feature
        Args:
        Returns
        Raises:
        """
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
        """
        get feature
        Args:
        Returns
        Raises:
        """
        common_conf = self.conf["common"]
        batch_size = common_conf["batch_size"]
        shuffle = common_conf["shuffle"]
        epoch_num = common_conf["epoch_num"]
        if shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 3)
        ds = ds.batch(batch_size).prefetch(batch_size)
        return ds

    def train(self, train_data):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        if self.model is None:
            self.init(training=True)
        common_conf = self.conf["common"]
        batch_size = common_conf["batch_size"]
        epoch_num = common_conf["epoch_num"]
        model = self.model

        for epo in range(epoch_num):
            cnt, pos = 0, 0
            for step, data in enumerate(train_data):
                with tf.GradientTape() as tape:
                    label, feature_ids, feature_vals = data
                    cnt += label.shape[0]
                    pos += sum(label)
                    pred = model([feature_ids, feature_vals])
                    loss = model.loss(tf.expand_dims(label, 1), pred)
                    gradients = tape.gradient(loss, model.trainable_weights)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    if step % 100 == 0:
                        print("{}\tepoch:{}\tstep No.{}\tloss:{}\tpos:{}\tcnt:{}".format(datetime.datetime.now(), epo, step, round(tf.reduce_mean(loss), 6), pos, cnt))
            print("{}\tepoch:{}\tfinish".format(datetime.datetime.now(), epo))
            self.set_training_model(False)
            self.test(train_data)   # need to opimize
            self.set_training_model(True)
            model.save_weights("{}/tf_model_{}.h5".format(common_conf["local_model_dir"], epo))

    def test(self, test_data, model_path=None):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        res = []
        loss_sum = 0.0
        pos = 0
        for step, feat in enumerate(test_data):
            label, feature_ids, feature_vals = data
            pred = self.model([feature_ids, feature_vals])
            loss = self.model.loss(label, pred)
            loss_sum += loss
            pred = tf.squeeze(pred, 1).numpy().tolist()
            label = label.numpy().tolist()
            pos += sum(label)
            res.extend(zip(label, pred))
            if step % 100 == 0:
                print("step No.{}\tloss\tpos:{}\tcnt:{}".format(step, round(loss_sum / len(res), 6), pos, cnt))
        print("{}\ttest_auc:{}\tsize:{}\tloss:{}\tpos:{}\tcnt:{}".format(model_path, ut.auc(res), len(res), round(loss_sum / len(res), 6), pos, cnt))

    def predict(self, data, model_path=None):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        if self.model is None:
            self.init()
            if model_path is None:
                self.model._set_inputs(data)
                self.model.load_weights(model_path)
        pred = self.model(data)
        return tf.squeeze(pred, 1).numpy().tolist()

    def dump_serving_model(self, model_path=None):
        """
        get feature
        Args:
        Returns
        Raises:
        """
        if self.model is None:
            self.init()
        self.model.is_save_model = True
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=["mae"]) # check
        conf = self.conf["common"]
        self.model._set_inputs(tf.keras.Input(shape=(2 * conf["padding_size"], ), dtype=tf.dtypes.float32))
        self.model.lad_weights(args.model)
        self.model.save(conf["local_model_dir"] + "/tfmodel", save_format="tf")


def DeepFMLearner(Learner):
    def __init__(self, conf_file_path="dfm.yml", conf=None):
        super().__init__(conf_file_path, conf)

    def make_config(self):
        if "model_params" not in self.conf:
            print("please config model_params in {}".format(self.conf_file_path), file=sys.stderr)
            return
        fea_slot, slot_size, fea_map = self.get_feature()
        model_params = self.conf["model_params"]
        model_params["slot_size"] = slot_size
        model_params["field_size"] = len(fea_slot)
        model_params["num_fea"] = num_fea
        model_params["dense_fea"] = dense_fea
        model_params["multi_fea"] = multi_fea
        model_params["seq_fea"] = seq_fea
        if self.conf["switch"]["enable_calc_padding"]:
            self.conf["feature"]["padding_size"] = self.get_padding_size(fea_slot, multi_fea, self.conf["feature"]["fix_addition_pad"])
        ut.save_yaml_file(self.conf, self.conf_file_path)


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
        solver.train(ds)
    elif args.method == "test":
        common_conf = conf["common"]
        batch_size = common_conf["batch_size"]
        preprocess_conf = common_conf["preprocess_conf"]
        if args.data:
            data = ud.ReadTextLineDatasetOnline(args.data, conf_file=preprocess_conf, batch_size=batch_size, fetch_size=1, num_parallel=10)
            solver.batch_test_local(data, args.model)
        else:
            solver.batch_test(sys.stdin, args.model)
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