#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import os
import shutil

import numpy as np
import tensorflow as tf

from config import EvalConfig, ModelConfig
from data import Data
from mir_eval.separation import bss_eval_sources
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only, soft_time_freq_mask, to_wav, write_wav


def eval(n):
    overall_gnsdr, overall_gsir, overall_gsar = [], [], []
    for i in range(n):
        with tf.Graph().as_default():
            # Model
            model = Model(ModelConfig.HIDDEN_LAYERS, ModelConfig.HIDDEN_UNITS)
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            with tf.Session(config=EvalConfig.session_conf) as sess:

                # Initialized, Load state
                sess.run(tf.global_variables_initializer())
                model.load_state(sess, EvalConfig.CKPT_PATH)

                print('num trainable parameters: %s' % (
                    np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

                writer = tf.summary.FileWriter(EvalConfig.GRAPH_PATH, sess.graph)

                data = Data(EvalConfig.DATA_PATH)
                mixed_wav, src1_wav, src2_wav, wavfiles = data.next_wavs(EvalConfig.SECONDS, EvalConfig.NUM_EVAL)

                mixed_spec = to_spectrogram(mixed_wav)
                mixed_mag = get_magnitude(mixed_spec)
                mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
                mixed_phase = get_phase(mixed_spec)

                assert (np.all(np.equal(model.batch_to_spec(mixed_batch, EvalConfig.NUM_EVAL), padded_mixed_mag)))

                (pred_src1_mag, pred_src2_mag) = sess.run(model(), feed_dict={model.x_mixed: mixed_batch})

                seq_len = mixed_phase.shape[-1]
                pred_src1_mag = model.batch_to_spec(pred_src1_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]
                pred_src2_mag = model.batch_to_spec(pred_src2_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]

                # Time-frequency masking
                mask_src1 = soft_time_freq_mask(pred_src1_mag, pred_src2_mag)
                # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
                mask_src2 = 1. - mask_src1
                pred_src1_mag = mixed_mag * mask_src1
                pred_src2_mag = mixed_mag * mask_src2

                # (magnitude, phase) -> spectrogram -> wav
                if EvalConfig.GRIFFIN_LIM:
                    pred_src1_wav = to_wav_mag_only(pred_src1_mag, init_phase=mixed_phase, num_iters=EvalConfig.GRIFFIN_LIM_ITER)
                    pred_src2_wav = to_wav_mag_only(pred_src2_mag, init_phase=mixed_phase, num_iters=EvalConfig.GRIFFIN_LIM_ITER)
                else:
                    pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
                    pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)

                # Write the result
                tf.summary.audio('GT_mixed', mixed_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
                tf.summary.audio('Pred_music', pred_src1_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
                tf.summary.audio('Pred_vocal', pred_src2_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)

                if EvalConfig.EVAL_METRIC:
                    # Compute BSS metrics
                    gnsdr, gsir, gsar = bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav)

                    # Write the score of BSS metrics
                    tf.summary.scalar('GNSDR_music', gnsdr[0])
                    tf.summary.scalar('GSIR_music', gsir[0])
                    tf.summary.scalar('GSAR_music', gsar[0])
                    tf.summary.scalar('GNSDR_vocal', gnsdr[1])
                    tf.summary.scalar('GSIR_vocal', gsir[1])
                    tf.summary.scalar('GSAR_vocal', gsar[1])
                    print ('GNSDR: ', gnsdr)
                    print ('GSIR: ', gsir)
                    print ('GSAR: ', gsar)

                overall_gnsdr.append(gnsdr)
                overall_gsir.append(gsir)
                overall_gsar.append(gsar)

                if EvalConfig.WRITE_RESULT:
                    # Write the result
                    for i in range(len(wavfiles)):
                        name = wavfiles[i].replace('/', '-').replace('.wav', '')
                        write_wav(mixed_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'original'))
                        write_wav(pred_src1_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'music'))
                        write_wav(pred_src2_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'voice'))

                writer.add_summary(sess.run(tf.summary.merge_all()), global_step=global_step.eval())

                writer.close()

    if n > 1:
        overall_gnsdr = np.array(overall_gnsdr)
        overall_gsir = np.array(overall_gsir)
        overall_gsar = np.array(overall_gsar)
        overall_gnsdr = np.mean(overall_gnsdr, axis=0)
        overall_gsir = np.mean(overall_gsir, axis=0)
        overall_gsar = np.mean(overall_gsar, axis=0)

        print ('OVERALL GNSDR: ', overall_gnsdr)
        print ('OVERALL GSIR: ', overall_gsir)
        print ('OVERALL GSAR: ', overall_gsar)


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:, :len_cropped]
    src2_wav = src2_wav[:, :len_cropped]
    mixed_wav = mixed_wav[:, :len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    for i in range(EvalConfig.NUM_EVAL):
        if abs(np.sum(src1_wav[i])) < 1e-10 or abs(np.sum(src2_wav[i])) < 1e-10:
            continue
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                            np.array([pred_src1_wav[i], pred_src2_wav[i]]), False)
        sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                              np.array([mixed_wav[i], mixed_wav[i]]), False)
        nsdr = sdr - sdr_mixed
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


def setup_path():
    if EvalConfig.RE_EVAL:
        if os.path.exists(EvalConfig.GRAPH_PATH):
            shutil.rmtree(EvalConfig.GRAPH_PATH)
        if os.path.exists(EvalConfig.RESULT_PATH):
            shutil.rmtree(EvalConfig.RESULT_PATH)

    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog -n 10')
    parser.add_option('-n', dest='n', default=1, type=int,
                      help="run the evaluation N times")
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
    setup_path()
    eval(options.n)
