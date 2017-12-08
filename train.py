#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import os
import shutil
import time

import numpy as np
import tensorflow as tf

from config import EvalConfig, TrainConfig
from data import Data
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, soft_time_freq_mask, to_wav, to_wav_mag_only
from utils import Diff, bss_eval_global


def eval(model, eval_data, sess):
    mixed_wav, src1_wav, src2_wav, _ = eval_data.next_wavs(EvalConfig.SECONDS, EvalConfig.NUM_EVAL)

    mixed_spec = to_spectrogram(mixed_wav)
    mixed_mag = get_magnitude(mixed_spec)

    src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
    src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

    src1_batch, _ = model.spec_to_batch(src1_mag)
    src2_batch, _ = model.spec_to_batch(src2_mag)
    mixed_batch, _ = model.spec_to_batch(mixed_mag)

    pred_src1_mag, pred_src2_mag = sess.run(model(),
                                            feed_dict={model.x_mixed: mixed_batch})

    mixed_phase = get_phase(mixed_spec)
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
        pred_src1_wav = to_wav_mag_only(pred_src1_mag, init_phase=mixed_phase,
                                        num_iters=EvalConfig.GRIFFIN_LIM_ITER)
        pred_src2_wav = to_wav_mag_only(pred_src2_mag, init_phase=mixed_phase,
                                        num_iters=EvalConfig.GRIFFIN_LIM_ITER)
    else:
        pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
        pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)

    # Compute BSS metrics
    gnsdr, gsir, gsar = bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav,
                                        EvalConfig.NUM_EVAL)
    return gnsdr, gsir, gsar


# TODO multi-gpu
def train():
    # Model
    model = Model()

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    model.gnsdr_music = tf.placeholder(dtype=tf.float32, shape=(), name='gnsdr_music')
    model.gsir_music = tf.placeholder(dtype=tf.float32, shape=(), name='gsir_music')
    model.gsar_music = tf.placeholder(dtype=tf.float32, shape=(), name='gsar_music')

    model.gnsdr_vocal = tf.placeholder(dtype=tf.float32, shape=(), name='gnsdr_vocal')
    model.gsir_vocal = tf.placeholder(dtype=tf.float32, shape=(), name='gsir_vocal')
    model.gsar_vocal = tf.placeholder(dtype=tf.float32, shape=(), name='gsar_vocal')

    # Summaries
    summary_ops = summaries(model, loss_fn)

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, TrainConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        # Input source
        data = Data(TrainConfig.DATA_PATH)
        eval_data = Data(EvalConfig.DATA_PATH)

        loss = Diff()
        gnsdr, gsir, gsar = np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        intial_global_step = global_step.eval()
        for step in xrange(intial_global_step, TrainConfig.FINAL_STEP):
            start_time = time.time()

            bss_metric = step % 20 == 0 or step == intial_global_step
            bss_eval = ''
            if bss_metric:
                gnsdr, gsir, gsar = eval(model, eval_data, sess)
                bss_eval = 'GNSDR: {} GSIR: {} GSAR: {}'.format(gnsdr, gsir, gsar)

            mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(TrainConfig.SECONDS, TrainConfig.NUM_WAVFILE)

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)

            src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

            src1_batch, _ = model.spec_to_batch(src1_mag)
            src2_batch, _ = model.spec_to_batch(src2_mag)
            mixed_batch, _ = model.spec_to_batch(mixed_mag)

            l, _, summary = sess.run([loss_fn, optimizer, summary_ops],
                                     feed_dict={model.x_mixed: mixed_batch,
                                                model.y_src1: src1_batch,
                                                model.y_src2: src2_batch,
                                                model.gnsdr_music: gnsdr[0],
                                                model.gsir_music: gsir[0],
                                                model.gsar_music: gsar[0],
                                                model.gnsdr_vocal: gnsdr[1],
                                                model.gsir_vocal: gsir[1],
                                                model.gsar_vocal: gsar[1]})
            loss.update(l)
            writer.add_summary(summary, global_step=step)

            # Save state
            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)

            elapsed_time = time.time() - start_time
            print('step-{}\ttime={:2.2f}\td_loss={:2.2f}\tloss={:2.3f}\tbss_eval: {}'.format(step,
                                                                                             elapsed_time,
                                                                                             loss.diff * 100,
                                                                                             loss.value,
                                                                                             bss_eval))

        writer.close()


def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)

    tf.summary.scalar('GNSDR_music', model.gnsdr_music)
    tf.summary.scalar('GSIR_music', model.gsir_music)
    tf.summary.scalar('GSAR_music', model.gsar_music)
    tf.summary.scalar('GNSDR_vocal', model.gnsdr_vocal)
    tf.summary.scalar('GSIR_vocal', model.gsir_vocal)
    tf.summary.scalar('GSAR_vocal', model.gsar_vocal)
    return tf.summary.merge_all()


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


if __name__ == '__main__':
    setup_path()
    train()
