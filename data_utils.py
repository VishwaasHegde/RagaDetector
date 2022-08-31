import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa
import librosa.display
import math
import raga_feature
from pydub import AudioSegment
from resampy import resample


def audio_2_frames(audio, config):
    # audio = audio[: self.model_srate*self.cutoff]
    hop_length = int(16000 * config['hop_size'])
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

    return frames

def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
        # product_sum = np.sum(
        #     salience * to_local_average_cents.cents_mapping)
        # return product_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

def stadardize(z):
    return (z - np.mean(z)) / (np.std(z)+0.001)

def normalize(z):
    z_min = np.min(z)
    return (z - z_min) / (np.max(z) - z_min + 0.001)


def get_cqt(audio, sr=16000):
    C = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=60, n_bins=60 * 7, pad_mode='wrap',
                           fmin=librosa.note_to_hz('C1')))
    c_cqt = librosa.amplitude_to_db(C, ref=np.max)
    c_cqt = np.reshape(c_cqt, [7, 60, -1])
    c_cqt = np.mean(c_cqt, axis=0)
    c_cqt = np.transpose(c_cqt)
    c_cqt = np.expand_dims(c_cqt,0)
    return c_cqt

def get_hist_cqt(audio, pitches):
    pitches = pitches
    cqt = get_cqt(audio)[0]
    pitches_mean = np.mean(pitches,0)
    pitches_std = np.std(pitches, 0)
    cqt_mean = np.mean(cqt, 0)
    cqt_std = np.std(cqt, 0)
    cqt_mean = np.roll(cqt_mean, 3, axis=-1)
    cqt_std = np.roll(cqt_std, 3, axis=-1)

    hist_cqt = np.stack([stadardize(pitches_mean), stadardize(pitches_std), stadardize(cqt_mean), stadardize(cqt_std)],-1)
    hist_cqt = np.expand_dims(hist_cqt,0)
    return hist_cqt

def get_raga_feat(pitches):
    return raga_feature.get_raga_feat(pitches)

def freq_to_cents(freq, std=25):
    frequency_reference = 10
    c_true = 1200 * math.log(freq / frequency_reference, 2)

    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
    return target

def mp3_to_wav(mp3_path):

    a = AudioSegment.from_mp3(mp3_path)
    y = np.array(a.get_array_of_samples())

    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(1)
    y = np.float32(y) / 2 ** 15

    y = resample(y, a.frame_rate, 16000)
    return y





