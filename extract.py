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
    # samples = np.array([data[0, :], data[1, :]]).flatten()
    samples = np.array(data).flatten()
    return total_samples, data, np.array([samples], dtype=np.float32)


def extract(filename, channel):
    with tf.Graph().as_default():
        # Model
        model = Model()
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

            base_file_name = os.path.splitext(filename)[0]

            def stack(data):
                size = data.shape[0] // channels
                elements = []
                for i in range(channels):
                    elements.append(data[size * i:size * (i + 1)])
                return np.dstack(elements)[0]

            music_data = pred_src1_wav[0]
            voice_data = pred_src2_wav[0]
            if channel >= 0:
                def filter_samples(data):
                    data = data.reshape((origin_samples.shape[0], data.shape[0] / origin_samples.shape[0]))
                    for i in range(origin_samples.shape[0]):
                        if i != channel:
                            data[i, :] = origin_samples[i, 0:data.shape[1]]
                    return data.flatten()

                music_data = filter_samples(music_data)
                voice_data = filter_samples(voice_data)

            music_wav = stack(music_data)
            voice_wav = stack(voice_data)

            write_wav(music_wav, base_file_name + '-h%d-music' % model.hidden_size)
            write_wav(voice_wav, base_file_name + '-h%d-voice' % model.hidden_size)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog music.wav')
    parser.add_option('-c', dest='channel', default=-1, type=int,
                      help="extract voice from specified channel, -1 to extract all channels")
    options, args = parser.parse_args()

    for arg in args:
        extract(arg, options.channel)
