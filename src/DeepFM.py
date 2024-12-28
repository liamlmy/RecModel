# coding: utf-8
"""
Author:
    Mingyang Li: liamlmy@gmail.com
"""

import numpy as np
import tensorflow as tf

class DeepFM(tf.keras.Model):
    def __init__(self, slot_size, filed_size, embedding_size=8,
                 traning=False, l2_reg=0.001, learning_rate=0.001,
                 zscore_file=None, use_bn=False,
                 dropout_fm=[0., 0.], deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 mulit_fea=[], dense_fea=[], seq_fea=[], sid_fea_slot=[]):
        super(DeepFM, self).__init__()
        self.is_save_model = False
        self.slot_size = slot_size
        self.embedding_size = embedding_size
        self.training = training
        self.mulit_fea = mulit_fea
        self.dense_fea = dense_fea
        self.seq_fea = seq_fea
        self.sid_fea_slot = sid_fea_slot
        self.filed_size_single = filed_size - len(mulit_fea) - len(dense_fea) - len(seq_fea)
        self.loss = tf.keras.losses.binary_crossentropy
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=1000000, decay_rate=1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")

        zscore_mean_np, zscore_var_np = self.zscore_np(slot_size, zscore_file)
        self.zscore_mean = tf.keras.layers.Embedding(slot_size, 1, weights=[zscore_mean_np], trainable=False)
        self.zscore_var = tf.keras.layers.Embedding(slot_size, 1, weights=[zscore_var_np], trainable=False)

        self.lr = tf.keras.layers.Embedding(slot_size, 1, embeddings_regularizer=regularizers.l2(l2_reg))
        self.fm = tf.keras.layers.Embedding(slot_size, embedding_size, embeddings_regularizer=regularizers.l2(l2_reg))
        
        self.dropout_lr = tf.keras.layers.Dropout(dropout_fm[0])
        self.dropout_fm = tf.keras.layers.Dropout(dropout_fm[1])
        self.dropout_deep = tf.keras.layers.Dropout(dropout_deep[0])

        print("slot_size={}".format(self.slot_size))

        self.deep = tf.keras.Sequential()
        for i, l in enumerate(deep_layers):
            if use_bn:
                self.deep.add(tf.keras.layers.BatchNormalization())
            self.deep.add(tf.keras.layers.Dense(l, activation=tf.nn.relu, kernal_regularizer=tf.keras.regularizers.l2(l2_reg)))
            if i + 1 < len(dropout_deep) and dropout_deep[i + 1] > 1e-6:
                self.deep.add(tf.keras.layers.Dropout(dropout_deep[i + 1]))
        self.dense_concat = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def zscore_np(self, feature_size, zscore_file=None):
        zscore_mean_np = np.zeros([feature_size, 1], dtype=np.float)
        zscore_var_np = np.ones([feature_size, 1], dtype=np.float)
        if zscore_file is not None:
            for line in open(zscore_file).readlines():
                i, m, v = line.strip().split(":")
                zscore_mean_np[int(i)][0] = float(m)
                zscore_var_np[int(i)][0] = float(v)
        return zscore_mean_np, zscore_var_np

    def transform(self, data):
        if self.is_save_model:
            feature_index = tf.cast(data[:, 0::2], tf.dtypes.int64)
            feature_value = data[:, 1::2]
        else:
            feature_index = tf.sparse.to_dense(data[0])
            feature_value = tf.sparse.to_dense(data[1])
        return feature_index, feature_value

    def get_sess_feat_index(self, seq_input):
        mask_bool = tf.fill(tf.shape(seq_input), False)
        mask_bool = mask_bool | tf.where(seq_input > 0.0, True, False)
        mask_bias = tf.where(mask_bool, float(self.sid_fea_slot[0] - 1), 0.0)
        seq_index = seq_input + mask_bias
        seq_value = tf.where(mask_bool, 1.0, 0.0)
        seq_index = self.emb_fm(seq_index)
        seq_value = tf.expand_dims(seq_value, 2)
        seq_input = tf.math.multiply(seq_input, seq_value)
        return seq_input

    def call(self, data):
        feature_index, feature_value = self.transform(data)
        candidate_input = None

        if len(self.sid_fea_slot) > 0:
            candidate_bool = tf.fill(tf.shape(feature_index), False)
            candidate_len = 0
            for x in self.sid_fea_slot:
                candidate_bool = candidate_bool | (tf.where(feature_index >= self.x[0], True, False) & tf.where(feature_index < self.x[1], True, False))
                candidate_len += (self.sid_fea_slot[1] - self.sid_fea_slot[0])
            candidate_mask = tf.where(candidate_bool, False, True)
            candidate_input = self.emb_fm(tf.boolean_mask(feature_index, candidate_bool))

        seq_input = None
        if len(self.seq_fea) > 0:
            seq_bool = tf.fill(tf.shape(feature_index), False)
            seq_len = 0
            for x in self.seq_fea:
                seq_bool = seq_bool | (tf.where(feature_index >= x[0], True, False) & tf.where(feature_index < x[1], True, False))
                seq_len += (x[1] - x[0])
                cur_seq_bool = tf.fill(tf.shape(feature_index), False)
                cur_seq_bool = cur_seq_bool | (tf.where(feature_index >= x[0], True, False) & tf.where(feature_index < x[1], True, False))
                cur_seq_len = (x[1] - x[0])
                cur_seq_mask = tf.where(seq_bool, False, True)
                cur_seq_input = tf.reshape(tf.boolean_mask(feature_value, cur_seq_bool), shape=[-1, cur_seq_len])
                cur_seq_input = self.get_sess_feat_index(cur_seq_input)
                mask = tf.cast(tf.not_equal(cur_seq_input[:, :, 0], 0), dtype=tf.float64)
                cur_seq_input = tf.reduce_sum(cur_seq_input, 1)
                if seq_input is None:
                    seq_input = cur_seq_input
                else:
                    seq_input = tf.concat([seq_input, cur_seq_input], 1)
            seq_mask = tf.where(seq_bool, False, True)
            feature_index = tf.where(seq_mask, feature_index, 0)
            feature_value = tf.where(seq_mask, feature_value, 0)

        dense_fea = None
        if len(self.dense_fea) > 0:
            dense_bool = tf.fill(tf.shape(feature_index), False)
            emb_len = 0
            for x in self.dense_fea:
                dense_bool = dense_bool | (tf.where(feature_index >= x[0], True, False) & tf.where(feature_index < x[1], True, False))
                emb_len += x[1] - x[0]
            dense_mask = tf.where(dense_bool, False, True)
            dense_input = tf.reshape(tf.boolean_mask(feature_value, dense_bool), shape=[-1, emb_len])
            feature_index = tf.where(dnese_mask, feature_index, 0)
            feature_value = tf.where(dense_mask, feature_value, 0)

        mean = self.zscore_mean(feature_index)
        var = self.zscore_var(feature_index)
        feature_value = tf.math.divide(tf.math.subtract(tf.expand_dims(feature_value, 2), mean), var)

        # lr part
        lr = tf.math.multiply(self.lr(feature_index), feature_value)
        lr = tf.reduce_sum(lr, 1)
        lr = self.dropout_lr(lr, training=self.training)

        # fm part
        full_embedding = self.emb_fm(feature_index)
        fm_embedding = tf.math.multiply(full_embedding, feature_value)
        square_sum_fm_embedding = tf.math.square(tf.reduce_sum(fm_embedding, 1))
        sum_square_fm_embedding = tf.reduce_sum(tf.math.square(fm_embedding), 1)
        fm = 0.5 * tf.math.subtract(square_sum_fm_embedding, sum_square_fm_embedding)
        fm = self.dropout_fm(fm, training=self.training)

        # deep part
        single_mask = tf.where(feature_index > 0, True, False)
        for fea in self.multi_fea:
            single_mask = single_mask & (tf.where(feature_index < fea[0], True, False) | tf.where(feature_index >= fea[1], True, False))
        single_feature_index = tf.reshape(tf.boolean_mask(feature_index, single_mask), shape=[-1, self.field_size_single])
        deep = self.emb_fm(single_feature_index)
        for fea in self.multi_fea:
            mask = tf.where(feature_index >= fea[0], 1, 0) * tf.where(feature_index < fea[1], 1, 0)
            value = feature_value * tf.cast(tf.expand_dims(mask, 2), tf.dtypes.float32)
            deep = tf.concat([deep, tf.reduce_sum(tf.math.multiply(full_embedding, value), 1, keepdims=True)], axis=1)
        deep = tf.reshape(deep, shape=[-1, (self.field_size_single + len(self.multi_fea)) * self.embedding_size])
        if dense_input is not None:
            deep = tf.concat([deep, dense_input], 1)
        if seq_input is not None:
            deep = tf.concat([deep, seq_input], 1)
        deep = self.dropout_deep(deep, training=self.training)
        deep = self.deep(deep, training=self.training)

        # concat lr & fm & deep
        concat = tf.concat([lr, fm, deep], axis=1)
        out = self.dense_concat(concat)

        return out
