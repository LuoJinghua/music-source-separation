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
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only, soft_time_freq_mask, to_wav, write_wav


def decode_input(filename):
    data, rate = librosa.load(filename, mono=False, sr=ModelConfig.SR)

    print ('channels: %d samples: %d' % data.shape)

    total_samples = data.shape[1]
    samples = np.array([data[0, :], data[0, :]]).flatten()
    return total_samples, np.array([samples], dtype=np.float32)


def extract(filename):
    # Model
    model = Model()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    total_samples, samples = decode_input(filename)
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

        base_file_name = os.path.splitext(filename)[0]
        def stack(data):
            size = data.shape[0] // 2
            left_data = data[0:size]
            right_data = data[size:size*2]
            return np.dstack((left_data, right_data))[0]

        music_wav = stack(pred_src1_wav[0])
        voice_wav = stack(pred_src2_wav[0])

        write_wav(music_wav, base_file_name + '-music')
        write_wav(voice_wav, base_file_name + '-voice')


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog music.wav')
    options, args = parser.parse_args()

    for arg in args:
        extract(arg)
