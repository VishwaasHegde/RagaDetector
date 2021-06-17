from __future__ import division
from __future__ import print_function

# import re
import sys
import math
from collections import defaultdict

from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Conv1D
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D
from tensorflow.keras.models import Model

import pyhocon
import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import recorder
from pydub import AudioSegment
from resampy import resample
import mir_eval
from data_generator import DataGenerator
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}
import data_utils

def build_and_load_model(config, task, tradition):
    """
    Build the CNN model and load the weights

    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    model_capacity = config['model_capacity']
    model_srate = config['model_srate']
    hop_size = int(config['hop_size'] * model_srate)
    sequence_length = int((config['sequence_length'] * model_srate - 1024) / hop_size) + 1
    cutoff = config['cutoff']
    n_frames = 1 + int((model_srate * cutoff - 1024) / hop_size)
    n_seq = int(n_frames // sequence_length)
    note_dim = config['note_dim']

    if task == 'pitch':
        x_batch = Input(shape=(None, 1024), name='x_input', dtype='float32')
        x = x_batch[0]
        y, note_emb = get_pitch_emb(x, n_seq, n_frames, model_capacity)
        pitch_model = Model(inputs=[x_batch], outputs=y)
        return pitch_model
    elif task == 'tonic':
        hist_cqt_batch = Input(shape=(60,4), name='hist_cqt_input', dtype='float32')
        hist_cqt = hist_cqt_batch[0]

        tonic_logits = get_tonic_emb(hist_cqt, note_dim, 0.6)

        tonic_model = Model(inputs=[hist_cqt_batch], outputs=[tonic_logits])
        tonic_model.compile(loss={'tf_op_layer_tonic': 'binary_crossentropy'},
                          optimizer='adam', metrics={'tf_op_layer_tonic': 'accuracy'},
                          loss_weights={'tf_op_layer_tonic': 1})

        return tonic_model

    elif task == 'raga':
        raga_feature_batch = Input(shape=(12, 12, 60, 4), name='raga_feature_input', dtype='float32')
        raga_feat = raga_feature_batch[0]
        raga_emb = get_raga_emb(raga_feat, note_dim)
        n_labels = config[tradition + '_n_labels']
        raga_logits = Dense(n_labels, activation='softmax', name='raga')(raga_emb)
        cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
        rag_model = Model(inputs=[raga_feature_batch], outputs=[raga_logits])
        rag_model.compile(loss={'raga': cce},
                      optimizer='adam', metrics={'raga': 'accuracy'})
        return rag_model


def get_pitch_emb(x, n_seq, n_frames, model_capacity):
    capacity_multiplier = {
        'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
    }[model_capacity]
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    z = []
    layers_cache = []
    for i in range(n_seq):
        x_pitch = x[int(i * n_frames / n_seq):int((i + 1) * n_frames / n_seq)]

        if i == 0:
            res = Reshape(target_shape=(1024, 1, 1), name='input-reshape')
            layers_cache.append(res)
            conv_layers = []
        else:
            res = layers_cache[0]
            conv_layers = layers_cache[1]
        y = res(x_pitch)
        m = 0
        for l, f, w, s in zip(layers, filters, widths, strides):
            if i == 0:
                conv_1 = Conv2D(f, (w, 1), strides=s, padding='same',
                                activation='relu', name="conv%d" % l, trainable=False)
                bn_1 = BatchNormalization(name="conv%d-BN" % l)
                mp_1 = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                                 name="conv%d-maxpool" % l, trainable=False)
                do_1 = Dropout(0.25, name="conv%d-dropout" % l)
                conv_layers.append([conv_1, bn_1, mp_1, do_1])
            else:
                conv_1, bn_1, mp_1, do_1 = conv_layers[m]

            y = conv_1(y)
            y = bn_1(y)
            y = mp_1(y)
            y = do_1(y)
            m += 1

        if i == 0:
            den = Dense(360, activation='sigmoid', name="classifier", trainable=False)
            per = Permute((2, 1, 3))
            flat = Flatten(name="flatten")
            layers_cache.append(conv_layers)
            layers_cache.append(den)
            layers_cache.append(per)
            layers_cache.append(flat)
        else:
            den = layers_cache[2]
            per = layers_cache[3]
            flat = layers_cache[4]

        y = per(y)
        y = flat(y)
        y = den(y)
        z.append(y)
    y = tf.concat(z, axis=0)

    return y, den.weights[0]



def get_hist_emb(hist_cqt, note_dim, indices, topk, drop_rate=0.2):
    hist_cc = [tf.cast(standardize(h), tf.float32) for h in hist_cqt]

    hist_cc = tf.transpose(tf.stack(hist_cc))
    hist_cc_all = []
    for i in range(topk):
        hist_cc_trans = tf.roll(hist_cc, -indices[i], axis=0)
        hist_cc_all.append(hist_cc_trans)
    hist_cc_all = tf.stack(hist_cc_all)
    hist_cc_all = Dropout(0.2)(hist_cc_all)

    d = 1
    f = 64

    z = convolution_block(d, hist_cc_all, 5, f, drop_rate)
    z = convolution_block(d, z, 3, 2*f, drop_rate)
    z = convolution_block(d, z, 3, 4*f, drop_rate)

    z = Flatten()(z)
    z = Dense(note_dim, activation='relu')(z)
    z = Dropout(0.4)(z)
    return z

def get_tonic_emb(hist_cqt, note_dim, drop_rate=0.2):
    topk=9

    indices = tf.random.uniform(shape=(topk,), minval=0, maxval=60, dtype=tf.int32)
    hist_emb = get_hist_emb(hist_cqt, note_dim, indices, topk, drop_rate)

    tonic_emb = hist_emb

    tonic_logits_norm = []
    tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)
    for i in range(topk):
        tonic_logits_norm.append(tf.roll(tonic_logits[i],indices[i], axis=0))
    tonic_logits = tf.reduce_mean(tonic_logits_norm, axis=0, keepdims=True)

    return tonic_logits

def get_raga_emb(raga_feat, note_dim):
    f = 192

    cnn_raga_emb = []
    raga_feat_dict_2 = defaultdict(list)

    lim = 55
    for i in range(0, 60, 5):
        for j in range(5, lim + 1, 5):
            s = i
            e = (i + j + 60) % 60

            rf = raga_feat[s//5,e//5,:,:]
            raga_feat_dict_2[j].append(rf)

    for k,v in raga_feat_dict_2.items():
        feat_1 = tf.stack(v, axis=0)
        feat_1 = tf.stack(feat_1)
        feat_2 = get_raga_from_cnn(feat_1, f, note_dim)
        cnn_raga_emb.append(feat_2)

    cnn_raga_emb = tf.stack(cnn_raga_emb)
    cnn_raga_emb = tf.squeeze(cnn_raga_emb,1)
    cnn_raga_emb = tf.transpose(cnn_raga_emb)
    cnn_raga_emb = Dropout(0.5)(cnn_raga_emb)
    cnn_raga_emb = Dense(1)(cnn_raga_emb)
    return tf.transpose(cnn_raga_emb)

def get_raga_from_cnn(raga_feat, f, note_dim=384):

    raga_feat = tf.expand_dims(raga_feat,0)

    y = Conv2D(f, (3, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(raga_feat)
    y = Conv2D(f, (3, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(2, 2))(y)
    # y = AvgPool2D(pool_size=(2, 2))(y)
    y = Dropout(0.7)(y)

    y = Conv2D(f*2, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(y)
    y = Conv2D(f*2, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(2, 2))(y)
    # y = AvgPool2D(pool_size=(2, 2))(y)
    y = Dropout(0.7)(y)

    y = Flatten()(y)

    y = Dense(2*note_dim, kernel_initializer='he_uniform', activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(note_dim, kernel_initializer='he_uniform', activation='relu')(y)
    return Dropout(0.2)(y)


def convolution_block(d, input_data, ks, f, drop_rate):
    if d==1:
        z = Conv1D(filters=f, kernel_size=ks, strides=1, padding='same', activation='relu')(input_data)
        z = Conv1D(filters=f, kernel_size=ks, strides=1, padding='same', activation='relu')(z)
        z = BatchNormalization()(z)
        z = MaxPool1D(pool_size=2)(z)
        z = Dropout(drop_rate)(z)
    else:
        z = Conv2D(filters=f, kernel_size=(ks, ks), strides=(1, 1), activation='relu', padding='same')(input_data)
        z = Conv2D(filters=f, kernel_size=(ks, ks), strides=(1, 1), activation='relu', padding='same')(z)
        z = BatchNormalization()(z)
        z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
        z = Dropout(drop_rate)(z)

    return z

def freq_to_cents(freq, std=25):
    frequency_reference = 10
    c_true = 1200 * math.log(freq / frequency_reference, 2)

    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
    return target

def standardize(z):
    return (z - tf.reduce_mean(z)) / (tf.math.reduce_std(z))

def normalize(z):
    min_z = tf.reduce_min(z)
    return (z - min_z) / (tf.reduce_max(z) - min_z)

def train(task, tradition):
    if task == 'tonic':
        train_tonic(tradition)
    elif task == 'raga':
        # tonic_model_path = train_tonic(tradition)
        raga_model_path = 'model/{}_raga_model.hdf5'.format(tradition)

        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
        training_generator = DataGenerator(task, tradition, 'train', config, random=True, shuffle=True, full_length=True)
        # validation_generator = DataGenerator(task, tradition, 'validate', config, random=False, shuffle=False, full_length=False)
        # validation_generator = DataGenerator(task, tradition, 'validate', config, random=False, shuffle=False, full_length=False)
        validation_generator = DataGenerator(task, tradition, 'validate', config, random=False, shuffle=False,
                                             full_length=False)
        # test_generator = DataGenerator(task, tradition, 'test', config, random=False, shuffle=False, full_length=False)
        # test_generator = DataGenerator(task, tradition, 'test', config, random=True, shuffle=False, full_length=True)

        model = build_and_load_model(config, task, tradition)
        # model.load_weights(tonic_model_path, by_name=True)
        # model.load_weights('model/model-full.h5', by_name=True)
        # model.fit(generator)
        # model.fit(x=training_generator,
        #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)

        checkpoint = ModelCheckpoint(raga_model_path, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max', period=1)

        # checkpoint = ModelCheckpoint(raga_model_path, monitor='raga_loss', verbose=1,
        #                              save_best_only=True, mode='min', period=1)
        # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, min_delta=0.0000001)
        # model.fit_generator(generator=training_generator,
        #                     validation_data=validation_generator, verbose=1, epochs=50, shuffle=True,
        #                     callbacks=[checkpoint])
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator, verbose=1, epochs=100, shuffle=True,
                            callbacks=[checkpoint])


def train_tonic(tradition):
    task = 'tonic'
    model_path = 'model/{}_tonic_model.hdf5'.format(tradition)
    # if os.path.exists(model_path):
    #     return model_path

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    training_generator = DataGenerator(task, tradition, 'train', config, random=True, shuffle=True, full_length=True)
    validation_generator = DataGenerator(task, tradition, 'validate', config, random=False, shuffle=False,
                                         full_length=False)
    test_generator = DataGenerator(task, tradition, 'test', config, random=False, shuffle=False, full_length=False)
    model = build_and_load_model(config, task, tradition)
    # model.load_weights('model/model-large.h5', by_name=True)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, verbose=1, epochs=50, shuffle=True,
                        callbacks=[checkpoint])

    return model_path


def test(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    test_generator = DataGenerator(task, tradition, 'test', config, random=True, shuffle=False, full_length=True)
    # test_generator = DataGenerator(task, tradition, 'test', config, random=False, shuffle=False, full_length=False)
    # test_generator = DataGenerator(task, tradition, 'test', config, random=True, shuffle=False, full_length=True)
    # test_generator = DataGenerator(task, tradition, 'validate', config, random=True, shuffle=False,
    #                                      full_length=True)
    # test_generator = DataGenerator(task, tradition, 'test', config, random=True)
    model = build_and_load_model(config, task, tradition)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    # model.load_weights('model/hindustani_tonic_model.hdf5', by_name=True)
    # model.load_weights('model/model-full.h5'.format(tradition, 'tonic'), by_name=True)
    # model.load_weights('model/{/}_{}_model.hdf5'.format(tradition, 'tonic'), by_name=True)
    model.load_weights('model/{}_{}_model.hdf5'.format(tradition, task), by_name=True)
    data = test_generator.data
    k=5
    if task=='raga':
        raga_pred = None
        tonic_pred = None
        all_p = []
        for i in range(k):
            p = model.predict_generator(test_generator, verbose=1)
            all_p.append(np.argmax(p, axis=1))

            # print(np.argmax(p[0], axis=1))
            if raga_pred is None:
                raga_pred = p
                # tonic_pred = p[1]
            else:
                raga_pred += p
                # tonic_pred += p[1]
        # all_p = np.array(all_p)
        # print(stats.mode(all_p, axis=0))
        # merged_raga_target, merged_raga_pred = merge_preds(raga_pred, data['labels'].values, data['mbid'].values)
        # print(merged_raga_target, merged_raga_pred)
        # data = pd.DataFrame()
        merged_raga_pred = np.argmax(raga_pred, axis=1)
        merged_raga_target = data['labels'].values
        data['predicted_raga'] = merged_raga_pred
        data['labels'] = merged_raga_target

        # data['predicted_raga'] = np.argmax(raga_pred, axis=1)
        acc = 0
        # print(data['predicted_raga'].values)
        for t,p in zip(data['labels'].values, data['predicted_raga'].values):
            acc += int(t == p)
        # tonic_pred = tonic_pred/k
        print('raga accuracy: {}'.format(acc/data.shape[0]))
        # print(data['labels'].values)
        # print(data['predicted_raga'].values)
        data.to_csv(config[tradition + '_' + 'output'], index=False, sep='\t')
    else:
        tonic_pred = None
        for i in range(k):
            p = model.predict_generator(test_generator, verbose=1)
            # print(np.argmax(p[0], axis=1))
            if tonic_pred is None:
                tonic_pred = p
            else:
                tonic_pred += p
        tonic_pred = tonic_pred/k
        # tonic_pred = tonic_pred
        # all_p = np.array(all_p)

    # model.evaluate_generator(generator=test_generator, verbose=1)
    # p = model.predict_generator(test_generator, verbose=1)
    # print(p[0, :, :, 0])
    # plt.imshow(p[1,:,:,1], cmap='hot', interpolation='nearest')
    # plt.show()

    # plt.imshow(p[1, :, :, 1], cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(p[1, :, :, 2], cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(p[1, :, :, 3], cmap='hot', interpolation='nearest')
    # plt.show()

    # print(p)
    # print(np.argmax(p[0]))
    # print(np.max(p, axis=1))
    # print(p[0])
    # print(np.argmax(p[0], axis=1))
    # print(np.max(p[1], axis=1))
    # print(np.argmax(p[1], axis=1))
    # for pi in p[0]:
    #     print(pi)
    # cents = to_local_average_cents(tonic_pred)
    # pred_frequency = 10 * 2 ** (cents / 1200)
    #
    # data['predicted_tonic'] = pred_frequency
    data.to_csv(config[tradition+'_'+'output'], index=False, sep='\t')

    # full_data = pd.read_csv(config[tradition+'_'+'output'], sep='\t')


    # if task=='tonic':
    #     for name, data in full_data.groupby('data_type'):
    #         tonic_accuracy(data)
    # else:
    #     tonic_accuracy(data)



    #

    # print('pred', frequency)

def merge_preds(raga_pred, target, mbids):
    pt = -1
    pmbid = mbids[0]
    merged_raga_pred = []
    merged_raga_target = []
    prev_raga_pred = 0
    for rp, t, mbid in zip(raga_pred, target, mbids):
        if mbid==pmbid:
            prev_raga_pred+=rp
        else:
            merged_raga_target.append(pt)
            merged_raga_pred.append(np.argmax(prev_raga_pred))
            prev_raga_pred=0
        pmbid=mbid
        pt=t

    merged_raga_target.append(pt)
    merged_raga_pred.append(np.argmax(prev_raga_pred))
    return merged_raga_target, merged_raga_pred



def test_pitch(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    model = build_and_load_model(config, task, tradition)
    model.load_weights('model/model-full.h5', by_name=True)
    model_srate = config['model_srate']
    step_size = config['hop_size']
    cuttoff = config['cutoff']
    for t in ['train', 'validate', 'test']:
        data_path = config[tradition + '_' + t]
        data = pd.read_csv(data_path, sep='\t')
        data = data.reset_index()

        slice_ind = 0
        k = 0

        while k < data.shape[0]:
            path = data.loc[k, 'path']
            # path = 'E:\\E2ERaga\\data\\RagaDataset\\audio\\cb280397-4e13-448e-9bd7-97105b2347dc.wav'
            pitch_path = path[:path.index('.wav')] + '.pitch'
            pitch_path = pitch_path.replace('audio', 'pitches')

            if os.path.exists(pitch_path):
                k+=1
                continue

            # if os.path.exists(pitch_path):
            #     pitch_file = open(pitch_path, "a")
            # else:
            #     pitch_file = open(pitch_path, "w")
            pitches = []
            while True:
                if slice_ind == 0:
                    print(pitch_path)

                frames, slice_ind = __data_generation_pitch(path, slice_ind, model_srate, step_size, cuttoff)

                p = model.predict(np.array([frames]))
                p = np.sum(np.reshape(p, [-1,6,60]), axis=1)
                # cents = to_local_average_cents(p)
                # frequency = 10 * 2 ** (cents / 1200)
                for p1 in p:
                    p1 = list(map(str, p1))
                    p1 = ','.join(p1)
                    pitches.append(p1)
                # p = ','.join(p)
                # pitches.extend(frequency)
                # pitches.extend(p)
                # pitches.append(p)
                if slice_ind == 0:
                    k += 1
                    break
                # frequency = list(map(str, frequency))
            pitches = list(map(str, pitches))
            pitch_file = open(pitch_path, "w")
            pitch_file.writelines('\n'.join(pitches))
            pitch_file.close()

def tonic_accuracy(data):
    tonic_label = []
    for a in data['tonic'].values:
        if a / 4 > 61.74:
            tonic_label.append(a / 8)
        else:
            tonic_label.append(a / 4)
    tonic_label = np.array(tonic_label)

    pred_frequency = data['predicted_tonic'].values

    ref_t = np.zeros_like(pred_frequency)
    (ref_v, ref_c,
     est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_t,
                                                     pred_frequency,
                                                     ref_t,
                                                     tonic_label)

    acc_50 = mir_eval.melody.raw_pitch_accuracy(ref_v[1:], ref_c[1:],
                                                est_v[1:], est_c[1:], cent_tolerance=50)
    acc_30 = mir_eval.melody.raw_pitch_accuracy(ref_v[1:], ref_c[1:],
                                                est_v[1:], est_c[1:], cent_tolerance=30)
    acc_10 = mir_eval.melody.raw_pitch_accuracy(ref_v[1:], ref_c[1:],
                                                est_v[1:], est_c[1:], cent_tolerance=10)

    print('tonic accuracies for {}, {}, {}'.format(acc_50, acc_30, acc_10))

def predict_run_time(tradition, seconds=60):
    pitch_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['pitch']
    pitch_model = build_and_load_model(pitch_config, 'pitch', tradition)
    pitch_model.load_weights('model/model-full.h5', by_name=True)

    tonic_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['tonic']
    tonic_model = build_and_load_model(tonic_config, 'tonic', tradition)
    tonic_model.load_weights('model/{}_tonic_model.hdf5'.format(tradition), by_name=True)

    raga_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
    raga_model = build_and_load_model(raga_config, 'tonic', tradition)
    raga_model.load_weights('model/{}_raga_model.hdf5'.format(tradition), by_name=True)

    while True:
        audio = recorder.record(seconds)
        if audio is None:
            return
        get_raga_tonic_prediction(audio, pitch_config, pitch_model, tonic_model, raga_model)


def get_raga_tonic_prediction(audio, pitch_config, pitch_model, tonic_model, raga_model):
    frames = data_utils.audio_2_frames(audio, pitch_config)
    p = pitch_model.predict(np.array([frames]))
    cents = data_utils.to_local_average_cents(p)
    frequencies = 10 * 2 ** (cents / 1200)
    pitches = [freq_to_cents(freq) for freq in frequencies]
    pitches = np.expand_dims(pitches, 0)
    hist_cqt = data_utils.get_hist_cqt(audio, pitches)

    p = tonic_model.predict(hist_cqt)
    cents = data_utils.to_local_average_cents(p)
    frequency = 10 * 2 ** (cents / 1200)
    print('tonic: {}'.format(frequency[0]))

    
    p = raga_model.predict(hist_cqt)
    cents = data_utils.to_local_average_cents(p)
    frequency = 10 * 2 ** (cents / 1200)
    print('tonic: {}'.format(frequency[0]))


def predict_run_time_file(file_path, tradition, filetype='wav'):

    pitch_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['pitch']
    pitch_model = build_and_load_model(pitch_config, 'pitch',tradition)
    pitch_model.load_weights('model/model-full.h5', by_name=True)

    raga_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
    raga_model = build_and_load_model(raga_config, 'raga', tradition)
    raga_model.load_weights('model/{}_raga_model.hdf5'.format(tradition), by_name=True)

    if filetype == 'mp3':
        audio = mp3_to_wav(file_path)
    else:
        sr, audio = wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)

    get_raga_tonic_prediction(audio, pitch_config, pitch_model, raga_model)


def mp3_to_wav(mp3_path):

    a = AudioSegment.from_mp3(mp3_path)
    y = np.array(a.get_array_of_samples())

    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(1)
    y = np.float32(y) / 2 ** 15

    y = resample(y, a.frame_rate, 16000)
    return y


def __data_generation_pitch(path, slice_ind, model_srate, step_size, cuttoff):
    # pitch_path = self.data.loc[index, 'pitch_path']
    # if self.current_data[2] == path:
    #     frames = self.current_data[0]
    #     pitches = self.current_data[1]
    #     pitches = pitches[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
    #     frames = frames[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
    #     return frames, pitches
    # else:
    #     sr, audio = wavfile.read(path)
    #     if len(audio.shape) == 2:
    #         audio = audio.mean(1)  # make mono
    #     audio = self.get_non_zero(audio)
    sr, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    # audio = self.get_non_zero(audio)

    # audio = audio[:self.model_srate*15]
    # audio = self.mp3_to_wav(path)

    # print(audio[:100])
    audio = np.pad(audio, 512, mode='constant', constant_values=0)
    audio_len = len(audio)
    audio = audio[slice_ind * model_srate * cuttoff:(slice_ind + 1) * model_srate * cuttoff]
    if (slice_ind + 1) * model_srate * cuttoff >= audio_len:
        slice_ind = -1
    # audio = audio[: self.model_srate*self.cutoff]
    hop_length = int(model_srate * step_size)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

    return frames, slice_ind + 1


def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    # z = np.reshape(activation, [-1, 6, 60])
    # z = np.mean(z, axis=1) #(None, 60)
    # z = np.reshape(z, [-1,12,5])
    # z = np.mean(z, axis=2)  # (None, 12)
    zarg = np.argmax(activation, axis=1)
    zarg = zarg % 60
    zarg = zarg / 5
    # ((((cents - 1997.3794084376191) / 20) % 60) / 5)
    return time, frequency, zarg, confidence, activation


def process_file(file, output=None, model_capacity='full', viterbi=False,
                 center=True, save_activation=False, save_plot=False,
                 plot_voicing=False, step_size=10, verbose=True):
    """
    Use the input model to perform pitch estimation on the input file.

    Parameters
    ----------
    file : str
        Path to WAV file to be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    save_activation : bool
        Save the output activation matrix to an .npy file. False by default.
    save_plot : bool
        Save a plot of the output activation matrix to a .png file. False by
        default.
    plot_voicing : bool
        Include a visual representation of the voicing activity detection in
        the plot of the output activation matrix. False by default, only
        relevant if save_plot is True.
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : bool
        Print status messages and keras progress (default=True).

    Returns
    -------

    """
    try:
        sr, audio = wavfile.read(file)
    except ValueError:
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    time, frequency, cents, confidence, activation = predict(
        audio, sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        center=center,
        step_size=step_size,
        verbose=1 * verbose)

    # write prediction as TSV
    f0_file = output_path(file, ".f0.csv", output)
    f0_data = np.vstack([time, frequency, cents, confidence]).transpose()
    np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f', '%.6f'], delimiter=',',
               header='time,frequency,cents,confidence', comments='')
    if verbose:
        print("CREPE: Saved the estimated frequencies and confidence values "
              "at {}".format(f0_file))

    # save the salience file to a .npy file
    if save_activation:
        activation_path = output_path(file, ".activation.npy", output)
        np.save(activation_path, activation)
        if verbose:
            print("CREPE: Saved the activation matrix at {}".format(
                activation_path))

    # save the salience visualization in a PNG file
    if save_plot:
        import matplotlib.cm
        from imageio import imwrite

        plot_file = output_path(file, ".activation.png", output)
        # to draw the low pitches in the bottom
        salience = np.flip(activation, axis=1)
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())

        if plot_voicing:
            # attach a soft and hard voicing detection result under the
            # salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = (
                inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])

        imwrite(plot_file, (255 * image).astype(np.uint8))
        if verbose:
            print("CREPE: Saved the salience plot at {}".format(plot_file))

