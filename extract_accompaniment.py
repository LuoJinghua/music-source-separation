#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from config import ModelConfig, EvalConfig
from extract import separate
from preprocess import write_wav


def extract(filename, acc_path, voice_path):
    if not acc_path and not voice_path:
        return
    music_wav, voice_wav = separate(filename, -1)

    if acc_path:
        write_wav(music_wav, os.path.splitext(acc_path)[0])
    if voice_path:
        write_wav(voice_wav, os.path.splitext(voice_path)[0])


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog music.mp3 -a acc.wav')
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
    parser.add_option('-v', '--vocal-path', dest='vocal_path', default='',
                      help='the path to the separated vocal wave file')
    parser.add_option('-a', '--accompaniment-path', dest='acc_path', default='',
                      help='the path to the separated accompaniment wave file')
    options, args = parser.parse_args()

    if options.check_point:
        EvalConfig.CKPT_PATH = options.check_point
    ModelConfig.HIDDEN_UNITS = options.hidden_units
    ModelConfig.HIDDEN_LAYERS = options.hidden_layers
    EvalConfig.CASE = options.case_name

    for arg in args:
        extract(arg, options.acc_path, options.vocal_path)
        break
