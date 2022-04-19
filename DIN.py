# coding: utf-8

import numpy as np
import tensorflow as tf

class Attention_Layer(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, activation="relu"):
        super(Attention_Layer, self).__init__()
        self.att_dense = [tf.keras.layers.Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = tf.keras.layers.Dense(1)
        self.const_min = -4294967295

    def call(self, inputs):
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)
        outputs = tf.squeeze(outputs, axis=-1)

        paddings = tf.ones_like(outputs) * self.const_min
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

        outputs = tf.expand_dims(tf.nn.softmax(logits=outputs), axis=1)
        outputs = tf.squeeze(tf.matmul(outputs, v), axis=1)

        return outputs


class DIN(tf.keras.Model):
    def __init__(self, feature_size, filed_size,
                 embedding_size=8, dropout_fm=[0., 0.],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 att_layers=[20, 20], l2_reg=0.001, zscore_file=None,
                 learning_rate=0.001, training=False, use_bn=False,
                 mulit_fea=[], preemb_fea=[], session_fea=[], sid_fea_slot=[]):
        super(DIN, self).__init__()
        self.is_save_model = False
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.training = training
        self.mulit_fea = mulit_fea
        self.preemb_fea = preemb_fea
        self.session_fea = session_fea
        self.sid_fea_slot = sid_fea_slot
        self.filed_size_single = filed_size - len(mulit_fea) - len(preemb_fea) - len(session_fea)
        self.loss = tf.keras.losses.binary_crossentropy
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=1000000, decay_rate=1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
        zscore_mean_np, zscore_var_np = self.zscore_np(feature_size, zscore_file)
        self.zscore_mean = tf.keras.layers.Embedding(feature_size, 1, weights=[zscore_mean_np], trainable=False)
        self.zscore_var = tf.keras.layers.Embedding(feature_size, 1, weights=[zscore_var_np], trainable=False)
        self.emb_lr = tf.keras.layers.Embedding(feature_size, 1, embeddings_regularizer=regularizers.l2(l2_reg))
        self.emb_fm = tf.keras.layers.Embedding(feature_size, embedding_size, embeddings_regularizer=regularizers.l2(l2_reg))
        self.attention_layer = Attention_Layer(att_layers, "sigmoid")
        self.dropout_lr = tf.keras.layers.Dropout(dropout_fm[0])
        self.dropout_fm = tf.keras.layers.Dropout(dropout_fm[1])
        self.dropout_deep = tf.keras.layers.Dropout(dropout_deep[0])

        print("feature_size={}".format(self.feature_size))

        self.deep = tf.keras.Sequential()
        for i, l in enumerate(deep_layers):
            if use_bn:
                self.deep.add(tf.keras.layers.BatchNormalization())
            self.deep.add(tf.keras.layers.Dense(l, activation=tf.nn.relu, kernal_regularizer=tf.keras.regularizers.l2(l2_reg)))
            if i + 1 < len(dropout_deep) and dropout_deep[i + 1] > 1e-6:
                self.deep.add(tf.keras.layers.Dropout(dropout_deep[i + 1]))
        self.dense_concat = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def zscore_np(self, feature_size, zscore_file=None):
        zscore_mean_np = np.zeros([feature_size, 1], dtype=np.float)
        zscore_var_np = np.ones([feature_size, 1], dtype=np.float)
        if zscore_file is not None:
            for line in open(zscore_file).readlines():
                i, m, v = line.strip().split(":")
                zscore_mean_np[int(i)][0] = float(m)
                zscore_var_np[int(i)][0] = float(v)
        return zscore_mean_np, zscore_var_np

    def transform(self, feat):
        if self.is_save_model:
            feat_index = tf.cast(feat[:, 0::2], tf.dtypes.int64)
            feat_value = feat[:, 1::2]
        else:
            feat_index = tf.sparse.to_dense(feat[0])
            feat_value = tf.sparse.to_dense(feat[1])
        return feat_index, feat_value

    def get_sess_feat_index(self, session_input):
        mask_bool = tf.fill(tf.shape(session_input), False)
        mask_bool = mask_bool | tf.where(session_input > 0.0, True, False)
        mask_bias = tf.where(mask_bool, float(self.sid_fea_slot[0] - 1), 0.0)
        session_index = session_input + mask_bias
        session_value = tf.where(mask_bool, 1.0, 0.0)
        session_index = self.emb_fm(session_index)
        session_value = tf.expand_dims(session_value, 2)
        session_input = tf.math.multiply(session_input, session_value)
        return session_input

    def call(self, feat):
        feat_index, feat_value = self.transform(feat)
        candidate_input = None

        if len(self.sid_fea_slot) > 0:
            candidate_bool = tf.fill(tf.shape(feat_index), False)
            candidate_len = 0
            for x in self.sid_fea_slot:
                candidate_bool = candidate_bool | (tf.where(feat_index >= self.x[0], True, False) & tf.where(feat_index < self.x[1], True, False))
                candidate_len += (self.sid_fea_slot[1] - self.sid_fea_slot[0])
            candidate_mask = tf.where(candidate_bool, False, True)
            candidate_input = self.emb_fm(tf.boolean_mask(feat_index, candidate_bool))

        session_input = None
        if len(self.session_fea) > 0:
            session_bool = tf.fill(tf.shape(feat_index), False)
            session_len = 0
            for x in self.session_fea:
                session_bool = session_bool | (tf.where(feat_index >= x[0], True, False) & tf.where(feat_index < x[1], True, False))
                session_len += (x[1] - x[0])
                cur_session_bool = tf.fill(tf.shape(feat_index), False)
                cur_session_bool = cur_session_bool | (tf.where(feat_index >= x[0], True, False) & tf.where(feat_index < x[1], True, False))
                cur_session_len = (x[1] - x[0])
                cur_session_mask = tf.where(session_bool, False, True)
                cur_session_input = tf.reshape(tf.boolean_mask(feat_value, cur_session_bool), shape=[-1, cur_session_len])
                cur_session_input = self.get_sess_feat_index(cur_session_input)
                mask = tf.cast(tf.not_equal(cur_session_input[:, :, 0], 0), dtype=tf.float64)
                cur_session_input = self.attention_layer([candidate_input, cur_session_input, cur_session_input, mask])
                if session_input is None:
                    session_input = cur_session_input
                else:
                    session_input = tf.concat([session_input, cur_session_input], 1)
            session_mask = tf.where(session_bool, False, True)
            feat_index = tf.where(session_mask, feat_index, 0)
            feat_value = tf.where(session_mask, feat_value, 0)

        preemb_fea = None
        if len(self.preemb_fea) > 0:
            preemb_bool = tf.fill(tf.shape(feat_index), False)
            emb_len = 0
            for x in self.preemb_fea:
                preemb_bool = preemb_bool | (tf.where(feat_index >= x[0], True, False) & tf.where(feat_index < x[1], True, False))
                emb_len += x[1] - x[0]
            preemb_mask = tf.where(preemb_bool, False, True)
            preemb_input = tf.reshape(tf.boolean_mask(feat_value, preemb_bool), shape=[-1, emb_len])
            feat_index = tf.where(preemb_mask, feat_index, 0)
            feat_value = tf.where(preemb_mask, feat_value, 0)

        mean = self.zscore_mean(feat_index)
        var = self.zscore_var(feat_index)
        feat_value = tf.math.divide(tf.math.subtract(tf.expand_dims(feat_value, 2), mean), var)

        # lr part
        lr = tf.reduce_sum(tf.math.multiply(self.emb_lr(feat_index), feat_value), 1)
        lr = self.dropout_lr(lr, training=self.training)

        # fm part
        full_embedding = self.emb_fm(feat_index)
        fm_embedding = tf.math.multiply(full_embedding, feat_value)
        square_sum_fm_embedding = tf.math.square(tf.reduce_sum(fm_embedding, 1))
        sum_square_fm_embedding = tf.reduce_sum(tf.math.square(fm_embedding), 1)
        fm = 0.5 * tf.math.subtract(square_sum_fm_embedding, sum_square_fm_embedding)
        fm = self.dropout_fm(fm, training=self.training)

        # deep part
        single_mask = tf.where(feat_index > 0, True, False)
        for fea in self.multi_fea:
            single_mask = single_mask & (tf.where(feat_index < fea[0], True, False) | tf.where(feat_index >= fea[1], True, False))
        single_feat_index = tf.reshape(tf.boolean_mask(feat_index, single_mask), shape=[-1, self.field_size_single])
        deep = self.emb_fm(single_feat_index)
        for fea in self.multi_fea:
            mask = tf.where(feat_index >= fea[0], 1, 0) * tf.where(feat_index < fea[1], 1, 0)
            value = feat_value * tf.cast(tf.expand_dims(mask, 2), tf.dtypes.float32)
            deep = tf.concat([deep, tf.reduce_sum(tf.math.multiply(full_embedding, value), 1, keepdims=True)], axis=1)
        deep = tf.reshape(deep, shape=[-1, (self.field_size_single + len(self.multi_fea)) * self.embedding_size])
        if preemb_input is not None:
            deep = tf.concat([deep, preemb_input], 1)
        if session_input is not None:
            deep = tf.concat([deep, session_input], 1)
        deep = self.dropout_deep(deep, training=self.training)
        deep = self.deep(deep, training=self.training)

        # concat lr & fm & deep
        concat = tf.concat([lr, fm, deep], axis=1)
        out = self.dense_concat(concat)

        return out
