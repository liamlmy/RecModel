# coding: utf-8

import utils as ut
import argparse
import copy
import sys
import os
import numpy as np
import tensorflow as tf
import datetime
from DeepFM import DeepFM

class Model:
    def __init__(self, conf_yml, conf=None):
        self.conf_yml = conf_yml
        self.model = None
        if conf is None:
            self.conf = ut.load_yml_file(conf_yml)
        else:
            self.conf = conf

    def get_fea_slot(self):
        x = {}
        feature_size = 0
        fea_slot_file = self.conf["common"]["fea_slot_file"]
        with open(fea_slot_file) as f:
            for line in f:
                lis = line.strip().split('\t')
                a = int(lis[1])
                b = int(lis[2])
                feature_size = a + b
                x[lis[0]] = (a, b)
        return x, feature_size

    def get_target_fea_slot(self, fea_name, fea_slot_dic=None):
        if fea_slot_dic is None:
            fea_slot_dic, _ = self.get_fea_slot()
        target_fea_conf = self.conf["common"][fea_name]
        res = []
        with open(target_fea_conf) as f:
            for line in f:
                k = line.strip()
                if k not in fea_slot_dic:
                    continue
                x = fea_slot_dic[k][0]
                y = x + fea_slot_dic[k][1]
                res.append([x, y])
        return res

    def get_padding_size(self, fea_slot, multihot_fea_slot, fix_addition=0):
        x = len(fea_slot)
        if fix_addition > 0:
            return x + fix_addition
        for y in multihot_fea_slot:
            x = x - 1 + y[1] - y[0]
        return x

    def set_training_mode(self, enable_training):
        pass

    def make_config(self):
        pass

    def init(self, training=False):
        if self.model is not None:
            return
        model_class = self.conf["common"]["model_class"]
        x = globals()
        if model_class not in x:
            sys.stderr.write("model_class not find: [%s]\n"%(model_class))
            return
        if training:
            self.conf["model_params"]["training"] = True
        self.model = x["model_params"](**self.conf["model_params"])

    def prepare_dataset(self, ds):
        common_conf = self.conf["common"]
        batch_size = common_conf["batch_size"]
        shuffle = common_conf["shuffle"]
        epoch_num = com_conf["epoch_num"]
        if shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 3)
        ds = ds.batch(batch_size).prefetch(batch_size)
        return ds

    def train(self, train_data):
        if self.model is None:
            self.init(training=True)
        common_conf = self.conf["common"]
        batch_size = common_conf["batch_size"]
        epoch_num = common_conf["epoch_num"]
        model = self.model
        for epo in range(epoch_num):
            for step, feat in enumerate(train_data):
                with tf.GradientTape() as tape:
                    pred = model([feat["fea_ids"], feat["fea_vals"]])
                    loss = model.loss(tf.expand_dims(feat["label"], 1), pred)
                    gradients = tape.gradient(loss, model.trainable_weights)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    if step % 100 == 0:
                        print("step:%08d\tloss:%04f"%(step, tf.reduce_mean(loss)))
            print(datetime.datetime.now(), "epo no:%d finish"%(epo))
            self.set_training_mode(False)
            self.test(train_data)
            self.set_training_mode(True)
            model.save_weights("%s/tfmodel_%d.h5"%(common_conf["local_model_dir"], epo))
        conf = self.conf["common"]
        model.save_weights(conf["local_model_dir"] + "/tfmodel.h5")

    def test(self, test_data, model_path=''):
        res = []
        loss_sum = 0.0
        for step, feat in enumerate(test_data):
            pred = self.model([feat["fea_ids"], feat["fea_vals"]])
            loss = self.model.loss(tf.expand_dims(feat["label"], 1), pred)
            loss_sum += tf.reduce_sum(loss, 0)
            pred = tf.squeeze(pred, 1).numpy().tolist()
            label = feat["label"].numpy().tolist()
            res.extend(zip(label, pred))
            if step % 10000 == 0:
                print("step:%d loss:%04f"%(step, loss_sum / len(res)))
        print(model_path, "test auc:%f size:%d loss:%f"%(ut.auc(res), len(res), loss_sum / len(res)))

    def dump_serving_model(self, model_path):
        if self.model is None:
            self.init()
        self.model.is_save_model = True
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=["mae"])
        common_conf = self.conf["common"]
        self.model._set_inputs(tf.keras.Input(shape=(2 * conf["padding_size"],), dtype=tf.dtypes.float32))
        self.model.load_weights(args.model)
        self.model.save(common_conf["local_model_dir"] + "/tfmodel", save_format="tf")

    def predict(self, X, model_path=None):
        if self.model is None:
            self.init()
            if model_path is not None:
                self.model._set_inputs(X)
                self.model.load_weights(model_path)
        pred = self.model(X)
        return tf.squeeze(pred, 1).numpy().tolist()

    def batch_test_local(self, data_set, model_path, batch_size=1024):
        pass

    def batch_test(self, infs, model_path, batch_size=1024, ofs=None, is_test=True, output_infos=False):
        pass


class DeepFMModel(Model):
    def __init__(self, conf_yml="dfm.yml", conf=None):
        super().__init__(conf_yml, conf)

    def make_config(self):
        if "model_params" not in self.conf:
            print("please config model_params in %s"%(self.conf_yml), file=sys.stderr)
            return
        fea_slot, feature_size = self.get_fea_slot()
        multi_fea = self.get_target_fea_slot("multihot_fea_conf", fea_slot)
        preemb_fea = self.get_target_fea_slot("preemb_fea_conf", fea_slot)
        session_fea = self.get_target_fea_slot("session_fea_conf", fea_slot)
        sid_fea = self.get_target_fea_slot("sid_fea_conf", fea_slot)
        x = self.conf["model_params"]
        x["feature_size"] = feature_size
        x["field_size"] = len(fea_slot)
        x["multi_fea"] = multi_fea
        x["preemb_fea"] = preemb_fea
        x["session_fea"] = session_fea
        x["sid_fea"] = sid_fea
        if self.conf["switch"]["enable_calc_padding"]:
            self.conf["common"]["padding_size"] = self.get_padding_size(fea_slot, multi_fea, self.conf["common"]["fix_addition_pad"])
        ut.save_yaml_file(self.conf, self.conf_yml)

    def set_training_mode(self, enable_training):
        self.model.training = enable_training


class XDeepFMModel(Model):
    def __init__(self, conf_yml="xdfm.yml", conf=None):
        super().__init__(conf_yml, conf)

    def make_config(self):
        if "model_params" not in self.conf:
            print("please config model_params in %s"%(self.conf_yml), file=sys.stderr)
            return
        fea_slot, feature_size = self.get_fea_slot()
        multi_fea = self.get_target_fea_slot("multihot_fea_conf", fea_slot)
        preemb_fea = self.get_target_fea_slot("preemb_fea_conf", fea_slot)
        session_fea = self.get_target_fea_slot("session_fea_conf", fea_slot)
        sid_fea = self.get_target_fea_slot("sid_fea_conf", fea_slot)
        x = self.conf["model_params"]
        x["feature_size"] = feature_size
        x["field_size"] = len(fea_slot)
        x["multi_fea"] = multi_fea
        x["preemb_fea"] = preemb_fea
        x["session_fea"] = session_fea
        x["sid_fea"] = sid_fea
        if self.conf["switch"]["enable_calc_padding"]:
            self.conf["common"]["padding_size"] = self.get_padding_size(fea_slot, multi_fea, self.conf["common"]["fix_addition_pad"])
        ut.save_yaml_file(self.conf, self.conf_yml)

    def set_training_mode(self, enable_training):
        self.model.training = enable_training
        pass


class DINModel(Model):
    def __init__(self, conf="din.yml", conf=None):
        super().__init__(conf_yml, conf)

    def make_config(self):
        if "model_params" not in self.conf:
            print("please config model_params in %s"%(self.conf_yml), file=sys.stderr)
            return
        fea_slot, feature_size = self.get_fea_slot()
        multi_fea = self.get_target_fea_slot("multihot_fea_conf", fea_slot)
        preemb_fea = self.get_target_fea_slot("preemb_fea_conf", fea_slot)
        session_fea = self.get_target_fea_slot("session_fea_conf", fea_slot)
        sid_fea = self.get_target_fea_slot("sid_fea_conf", fea_slot)
        x = self.conf["model_params"]
        x["feature_size"] = feature_size
        x["field_size"] = len(fea_slot)
        x["multi_fea"] = multi_fea
        x["preemb_fea"] = preemb_fea
        x["session_fea"] = session_fea
        x["sid_fea"] = sid_fea
        if self.conf["switch"]["enable_calc_padding"]:
            self.conf["common"]["padding_size"] = self.get_padding_size(fea_slot, multi_fea, self.conf["common"]["fix_addition_pad"])
        ut.save_yaml_file(self.conf, self.conf_yml)

    def set_training_mode(self, enable_training):
        self.model.training = enable_training
        pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="train&evaluate the DNN model")
    parse.add_argument("-m", type=str, help="-method:train|test|auto_config", required=True)
    parse.add_argument("-conf", type=str, help="yml conf", required=True)
    parse.add_argument("-model", type=str, help="init model dir")
    parse.add_argument("-data", type=str, help="input data files")
    args = parse.parse_args()
    x = ut.load_yml_file(args.conf)
    mc = x["common"]["model_class"]
    mcif = mc + "Model"
    z = globals()
    solver = None
    if mcif in z:
        solver = z[mcif][args.conf, x]
    else:
        solver = Model(args.conf, x)
        print("no model interface %s find"%(mcif), file=sys.stderr)

    os.environ["CUDA_VISIBLE_DEVICES"] = x["common"]["CUDA_VISIBLE_DEVICES"]
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.m == "train":
        info = {}
        com_conf = x["common"]
        batch_size = com_conf["batch_size"]
        shuffle_size = batch_size * 10
        ds = ut.ReadTFRecord(args.data, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=1, num_parallel=10)
        solver.train(ds)
    elif args.m == "test":
        if args.data:
            ds = ut.ReadTFRecord(args.data, batch_size=1024, fetch_size=1)
            solver.batch_test_local(ds, args.model)
        else:
            solver.batch_test(sys.stdin, args.model)
    elif args.m == "predict":
        solver.batch_test(sys.stdin, args.model, ofs=sys.stdout, is_test=False)
    elif args.m == "predict_info":
        solver.batch_test(sys.stdin, args.model, ofs=sys.stdout, is_test=False, output_infos=True)
    elif args.m == "dump_serving":
        solver.dump_serving_model(args.model)
    elif args.m == "auto_config":
        solver.make_config()
    else:
        pass
