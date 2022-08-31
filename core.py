from __future__ import division
from __future__ import print_function

import sys
from collections import defaultdict

from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Conv1D
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D
from tensorflow.keras.models import Model
import raga_feature
import pyhocon
import os
import pandas as pd
from pydub import AudioSegment
from resampy import resample
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle
import matplotlib.pyplot as plt

models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}
import data_utils

standard_tonic = ['C','C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

class CRePE:
    def __init__(self):
        pitch_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['pitch']
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        # pitch_model = tf.keras.models.load_model('model/pitch/model-full.h5')
        # pitch_model = self.build_model(pitch_config)
        # pitch_model.load_weights('model/model-full.h5')
        # pitch_model.save('model/pitch/model-full.h5')
        with self.sess.as_default():
            with self.graph.as_default():
                pitch_model = self.build_model(pitch_config)
                pitch_model.load_weights('model/model-full.h5', by_name=True)

        # tf.keras.backend.clear_session()
        # self.graph_pitch = tf.Graph()
        # with self.graph_pitch.as_default():
        #     pitch_model = self.build_model(pitch_config)
        # pitch_model = self.build_model(pitch_config)
        # pitch_model.load_weights('model/model-full.h5', by_name=True)
        self.pitch_model = pitch_model
        self.pitch_config = pitch_config

    def interp1d(self, array: np.ndarray, fn, tn) -> np.ndarray:
        return np.interp(np.linspace(0, fn - 1, num=tn), np.arange(fn), array)

    def stretch(self, pitches):
        pitches_exp = np.zeros([pitches.shape[0], 120])
        for idx, p in enumerate(pitches):
            pitches_exp[idx] = self.interp1d(p, 60, 120)
        return pitches_exp

    def build_model(self, config):
        model_capacity = config['model_capacity']
        model_srate = config['model_srate']
        hop_size = int(config['hop_size'] * model_srate)
        sequence_length = int((config['sequence_length'] * model_srate - 1024) / hop_size) + 1
        cutoff = config['cutoff']
        n_frames = 1 + int((model_srate * cutoff - 1024) / hop_size)
        # n_seq = int(n_frames // sequence_length)
        n_seq = 1
        # x = x_batch[0]
        x_batch = Input(shape=(1024,), name='x_input', dtype='float32')
        x = x_batch
        y, note_emb = self.get_pitch_emb(x, n_seq, n_frames, model_capacity)
        pitch_model = Model(inputs=[x_batch], outputs=y)
        return pitch_model

    def predict_pitches(self, audio):
        frames = data_utils.audio_2_frames(audio, self.pitch_config)
        with self.sess.as_default():
            with self.graph.as_default():
                p = self.pitch_model.predict(frames)
                print("Pitch Prediction Completed")

        pitches = np.sum(np.reshape(p, [-1,6,60]),1)
        # cents = data_utils.to_local_average_cents(p)
        # frequencies = 10 * 2 ** (cents / 1200)
        # pitches = [data_utils.freq_to_cents(freq) for freq in frequencies]
        # pitches = np.sum(np.reshape(pitches, [-1,6,60]),1)
        # plt.plot(np.mean(pitches, axis=0))
        # plt.show()
        print('Argmax')
        print(np.argmax(np.mean(pitches, axis=0)))
        return pitches

    def get_pitch_emb(self, x, n_seq, n_frames, model_capacity):
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
            # x_pitch = x[int(i * n_frames / n_seq):int((i + 1) * n_frames / n_seq)]
            x_pitch = x
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


class SPD_Model:

    def __init__(self, tradition):
        self.graph_tonic = tf.Graph()
        self.sess_tonic = tf.compat.v1.Session(graph=self.graph_tonic)
        with self.sess_tonic.as_default():
            with self.graph_tonic.as_default():
                tonic_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['tonic']
                self.tonic_model = self.build_and_load_model(tonic_config, 'tonic', tradition)
                self.tonic_model.load_weights('model/{}_tonic_model.hdf5'.format(tradition))
        if tradition == 'Carnatic':
            self.models_weights = [0.21073548, 0.09824791, -0.11856023, 0.10903256,
                                   0.071041666, 0.029498437, 0.058856107, 0.005804086,
                                   0.23304237, -0.0036090536, -0.16575447, 0.1428653,
                                   -0.114484884, 0.067209795, -0.15399341, 0.06962044,
                                   0.04790523, -0.1683678, 0.067225285, -0.06486485,
                                   0.12629355, -0.06692766, 0.035410862, 0.14386775, 0.13790666]
        else:
            self.models_weights = [0.14530538, 0.16622296, 0.023467897, -0.14283751,
                                   0.28589576, -0.18367307, 0.06799893, 0.12999865,
                                   -0.50317067, 0.39374444, -0.011883757, -0.027416285,
                                   -0.10365037, 0.026902597, -0.0032952016, 0.1016536,
                                   0.09131258, 0.29101947, 0.050576165, -0.16490516,
                                   -0.09560222, -0.116639435, 0.13686526, 0.14130622, 0.2870755]

        raga_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
        self.raga_list = self.get_raga_list(raga_config, tradition)
        self.knn_models = self.build_and_load_model(raga_config, 'raga', tradition)


    def build_and_load_model(self, config, task, tradition):
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
        note_dim = config['note_dim']

        if task == 'tonic':
            hist_cqt_batch = Input(shape=(60,4), name='hist_cqt_input', dtype='float32')
            hist_cqt = hist_cqt_batch[0]

            tonic_logits = self.get_tonic_emb(hist_cqt, note_dim, 0.6)

            tonic_model = Model(inputs=[hist_cqt_batch], outputs=[tonic_logits])
            tonic_model.compile(loss={'tf_op_layer_tonic': 'binary_crossentropy'},
                              optimizer='adam', metrics={'tf_op_layer_tonic': 'accuracy'},
                              loss_weights={'tf_op_layer_tonic': 1})

            return tonic_model

        elif task == 'raga':
            knn_models = {}
            for wd in range(0,250,10):
                with open('data/RagaDataset/{}/model/spd_knn_{}.pkl'.format(tradition, wd), 'rb') as f:
                    knn_models[wd] = pickle.load(f)
            return knn_models

    def get_hist_emb(self, hist_cqt, note_dim, indices, topk, drop_rate=0.2):
        # hist_cc = [tf.cast(self.standardize(h), tf.float32) for h in hist_cqt]

        hist_cc = hist_cqt
        hist_cc_all = []
        for i in range(topk):
            hist_cc_trans = tf.roll(hist_cc, -indices[i], axis=0)
            hist_cc_all.append(hist_cc_trans)
        hist_cc_all = tf.stack(hist_cc_all)
        hist_cc_all = Dropout(0.2)(hist_cc_all)

        d = 1
        f = 64

        z = self.convolution_block(d, hist_cc_all, 5, f, drop_rate)
        z = self.convolution_block(d, z, 3, 2*f, drop_rate)
        z = self.convolution_block(d, z, 3, 4*f, drop_rate)

        z = Flatten()(z)
        z = Dense(note_dim, activation='relu')(z)
        z = Dropout(0.4)(z)
        return z

    def get_tonic_emb(self, hist_cqt, note_dim, drop_rate=0.2):
        topk=9

        indices = tf.random.uniform(shape=(topk,), minval=0, maxval=60, dtype=tf.int32)
        hist_emb = self.get_hist_emb(hist_cqt, note_dim, indices, topk, drop_rate)

        tonic_emb = hist_emb

        tonic_logits_norm = []
        tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)
        for i in range(topk):
            tonic_logits_norm.append(tf.roll(tonic_logits[i],indices[i], axis=0))
        tonic_logits = tf.reduce_mean(tonic_logits_norm, axis=0, keepdims=True)
        tonic_logits = tf.roll(tonic_logits,0,axis=1, name='tonic')
        return tonic_logits

    def convolution_block(self, d, input_data, ks, f, drop_rate):
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

    def standardize(self, z):
        return (z - tf.reduce_mean(z)) / (tf.math.reduce_std(z))

    def normalize(self, z):
        min_z = tf.reduce_min(z)
        return (z - min_z) / (tf.reduce_max(z) - min_z)

    def  predict_tonic_raga(self, crepe, audio, pitchvalue_prob, tonic=None):
        if tonic is None:
            hist_cqt = data_utils.get_hist_cqt(audio, pitchvalue_prob)
            with self.sess_tonic.as_default():
                with self.graph_tonic.as_default():
                    pred_tonic = self.tonic_model.predict(hist_cqt)[0]
                    print("Tonic Prediction Complete")

            print('pred_tonic', pred_tonic)
            print('argmax pred_tonic', np.argmax(pred_tonic))

            pred_12 = np.sum(np.reshape(pred_tonic, [12, 5]), 1)
            tonic_12 = standard_tonic[np.argmax(pred_12)]
            tonic = np.argmax(pred_tonic)
            # tonic = tonic-3

            # tonic = 20
        else:
            tonic_12 = standard_tonic.index(tonic)
            tonic = tonic_12*5
            tonic = tonic+3  # Ugly hack to fix some tonic issue

        # tonic = 20
        pitchvalue_prob = crepe.stretch(pitchvalue_prob)
        tonic = tonic*2
        print('tonic', tonic)
        pitchvalue_prob = np.roll(pitchvalue_prob, -tonic, axis=1)

        pred_proba = raga_feature.get_raga_feat_and_predict(self.knn_models, pitchvalue_prob, len(self.raga_list))
        pred_proba = pred_proba.T  # (n_labels, 25)

        models_weights = np.expand_dims(self.models_weights, 1)
        y_pred = np.matmul(pred_proba, models_weights)[:,0]
        y_pred = np.argmax(y_pred, 0)
        pred_raga = self.raga_list[y_pred]
        return tonic_12, pred_raga

    def get_raga_list(self, raga_config, tradition):
        raga_list = pd.read_csv(raga_config['{}_targets'.format(tradition)], header=None)
        raga_list = list(raga_list.iloc[:,0])
        return raga_list


    def get_raga_tonic_prediction(self, audio, pitch_config, pitch_model, tonic_model, raga_model):

        p = tonic_model.predict(hist_cqt)
        cents = data_utils.to_local_average_cents(p)
        frequency = 10 * 2 ** (cents / 1200)
        print('tonic: {}'.format(frequency[0]))
        p = raga_model.predict(hist_cqt)
        print('tonic: {}'.format(frequency[0]))


    def predict_run_time_file(self, file_path, tradition, filetype='wav'):

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


    def mp3_to_wav(self, mp3_path):

        a = AudioSegment.from_mp3(mp3_path)
        y = np.array(a.get_array_of_samples())

        if a.channels == 2:
            y = y.reshape((-1, 2))
            y = y.mean(1)
        y = np.float32(y) / 2 ** 15

        y = resample(y, a.frame_rate, 16000)
        return y


    def __data_generation_pitch(self, path, slice_ind, model_srate, step_size, cuttoff):
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


    def predict(self, audio, sr, model_capacity='full',
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


    def process_file(self, file, output=None, model_capacity='full', viterbi=False,
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
