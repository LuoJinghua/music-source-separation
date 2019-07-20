#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import os

import librosa
import numpy as np
import tensorflow as tf

from config import EvalConfig, ModelConfig
from model import Model
from preprocess import soft_time_freq_mask, to_wav, write_wav
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only


def decode_input(filename):
    data, rate = librosa.load(filename, mono=False, sr=ModelConfig.SR)

    print ('channels: %d samples: %d' % data.shape)

    n_channels = data.shape[0]
    total_samples = data.shape[1]
    result = []
    for ch in range(n_channels):
        result.append(np.array([data[ch, :]]).flatten())
    return total_samples, data, np.array(result, dtype=np.float32)


def separate(filename, channel):
    with tf.Graph().as_default():
        # Model
        model = Model(ModelConfig.HIDDEN_LAYERS, ModelConfig.HIDDEN_UNITS)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        total_samples, origin_samples, samples = decode_input(filename)
        channels = origin_samples.shape[0]
        with tf.Session(config=EvalConfig.session_conf) as sess:

            # Initialized, Load state
            sess.run(tf.global_variables_initializer())
            model.load_state(sess, EvalConfig.CKPT_PATH)

            mixed_wav, src1_wav, src2_wav = samples, samples, samples

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)
            mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
            mixed_phase = get_phase(mixed_spec)

            (pred_src1_mag, pred_src2_mag) = sess.run(model(), feed_dict={model.x_mixed: mixed_batch})

            seq_len = mixed_phase.shape[-1]
            pred_src1_mag = model.batch_to_spec(pred_src1_mag, 1)[:, :, :seq_len]
            pred_src2_mag = model.batch_to_spec(pred_src2_mag, 1)[:, :, :seq_len]

            # Time-frequency masking
            mask_src1 = soft_time_freq_mask(pred_src1_mag, pred_src2_mag)
            # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
            mask_src2 = 1. - mask_src1
            pred_src1_mag = mixed_mag * mask_src1
            pred_src2_mag = mixed_mag * mask_src2

            # (magnitude, phase) -> spectrogram -> wav
            if EvalConfig.GRIFFIN_LIM:
                pred_src1_wav = to_wav_mag_only(pred_src1_mag, init_phase=mixed_phase,
                                                num_iters=EvalConfig.GRIFFIN_LIM_ITER)
                pred_src2_wav = to_wav_mag_only(pred_src2_mag, init_phase=mixed_phase,
                                                num_iters=EvalConfig.GRIFFIN_LIM_ITER)
            else:
                pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
                pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)

            def stack(data):
                size = data.shape[0] // channels
                elements = []
                for i in range(channels):
                    elements.append(data[size * i:size * (i + 1)])
                return np.dstack(elements)[0]

            music_data = pred_src1_wav
            voice_data = pred_src2_wav

            if channel >= 0:
                def filter_samples(data):
                    for i in range(origin_samples.shape[0]):
                        if i != channel:
                            data[i, :] = origin_samples[i, 0:data.shape[1]]
                    return data

                music_data = filter_samples(music_data)
                voice_data = filter_samples(voice_data)

            music_wav = np.dstack(music_data)[0]
            voice_wav = np.dstack(voice_data)[0]
            return music_wav, voice_wav
    return None


def extract(filename, channel):
    music_wav, voice_wav = separate(filename, channel)

    base_file_name = os.path.splitext(filename)[0]
    write_wav(music_wav, base_file_name + '-h%d-music' % ModelConfig.HIDDEN_UNITS)
    write_wav(voice_wav, base_file_name + '-h%d-voice' % ModelConfig.HIDDEN_UNITS)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog music.wav')
    parser.add_option('-c', dest='channel', default=-1, type=int,
                      help="extract voice from specified channel, -1 to extract all channels")
    parser.add_option('-p', dest='check_point', default=EvalConfig.CKPT_PATH,
                      help="the path to checkpoint")
    parser.add_option('--hidden-units', dest='hidden_units', default=ModelConfig.HIDDEN_UNITS, type=int,
                      help='the hidden units per GRU cell')
    parser.add_option('--hidden-layers', dest='hidden_layers', default=ModelConfig.HIDDEN_LAYERS, type=int,
                      help='the hidden layers of network')
    parser.add_option('--case-name', dest='case_name', default=EvalConfig.CASE,
                      help='the name of this setup')
    options, args = parser.parse_args()
    if options.check_point:
        EvalConfig.CKPT_PATH = options.check_point
    ModelConfig.HIDDEN_UNITS = options.hidden_units
    ModelConfig.HIDDEN_LAYERS = options.hidden_layers
    EvalConfig.CASE = options.case_name
    for arg in args:
        extract(arg, options.channel)
