from core import CRePE, CreToRa
import recorder
import os
import data_utils
from scipy.io import wavfile
import argparse


def predict_run_time(crepe, cretora_hindustani, cretora_carnatic, tradition='h', tonic=None, seconds=60):
    while True:
        audio = recorder.record(seconds)
        if audio is None:
            return
        pitches = crepe.predict_pitches(audio)
        if tradition=='h':
            pred_tonic, pred_raga = cretora_hindustani.predict_tonic_raga(audio, pitches, tonic)
        else:
            pred_tonic, pred_raga = cretora_carnatic.predict_tonic_raga(audio, pitches, tonic)

        if tonic is None:
            print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))
        else:
            print('Predicted Raga: {}'.format(pred_raga))

def predict_on_file(crepe, cretora, file_path):
    split_tup = os.path.splitext(file_path)

    if split_tup[1] == '.mp3':
        audio = data_utils.mp3_to_wav(file_path)
    else:
        sr, audio = wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)

    pitches = crepe.predict_pitches(audio)
    print("Pitch Prediction Complete")

    # pitches = crepe.predict_pitches(audio)
    pred_tonic, pred_raga = cretora.predict_tonic_raga(audio, pitches, None)

    print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))

if __name__ == '__main__':

    # tradition = 'h'
    # runtime_file = 'data/sample_data/Bhup_25.wav'
    # crepe = CRePE()
    # cretora = CreToTa('hindustani')
    # predict_on_file(crepe, cretora, runtime_file)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--runtime', default=False,
                            help='records audio for the given amount of duration')
    arg_parser.add_argument('--runtime_file', default=None,
                            help='runs the model on the input file path')
    arg_parser.add_argument('--duration', default=60,
                            help='sets the duration for recording in seconds')
    arg_parser.add_argument('--tradition', default='h',
                            help='sets the tradition - [h]industani/[c]arnatic')
    arg_parser.add_argument('--tonic', default=None,
                            help='sets the tonic if given, otherwise tonic is predicted')

    p_args = arg_parser.parse_args()

    crepe = CRePE()

    if p_args.runtime:
        cretora_hindustani = CreToRa('hindustani')
        cretora_carnatic= CreToRa('carnatic')
        predict_run_time(crepe, cretora_hindustani, cretora_carnatic,
                         tradition=p_args.tradition, tonic=p_args.tonic, seconds=int(p_args.duration))

    elif p_args.runtime_file:
        if p_args.tradition == 'h':
            cretora = CreToRa('hindustani')
        else:
            cretora = CreToRa('carnatic')
        predict_on_file(crepe, cretora, p_args.runtime_file)
