# -*- coding: utf-8 -*-
""" from https://github.com/microsoft/DNS-Challenge/blob/master/audiolib.py """

"""
@author: chkarada
"""
import os
import numpy as np
import soundfile as sf
from scipy import signal

EPS = np.finfo(float).eps
np.random.seed(0)

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def normalize(audio):
    scalar = 1.0/max(abs(audio)+EPS)
    audio = audio * scalar
    return audio

def unify_normalize(*args):
    max_amp = activelev(args)
    unify_scale = 0.99/(max_amp+EPS)
    return [x * unify_scale for x in args]

def activelev(args):
    return max([max(abs(x)) for x in args])


def audioread(path, start=0, stop=None):
    '''Function to read audio'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
        return (None, None)

    if len(audio.shape) == 1:  # mono
        pass
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]

    return audio, sample_rate


def add_clipping(audio, max_thresh_perc=0.8):
    '''Function to add clipping'''
    threshold = max(abs(audio))*max_thresh_perc
    audioclipped = np.clip(audio, -threshold, threshold)
    return audioclipped


def add_pyreverb(clean_speech, rir, level=0.2):
    '''Function to add reverb'''
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: clean_speech.shape[0]]
    mixed = reverb_speech * level + clean_speech * (1.0-level)
    return mixed


def snr_mixer(clean, noise, snr):
    '''Function to mix clean speech and noise at various SNR levels.
    clean and noise should be of equal length'''

    # Set the noise level for a given SNR
    noisescalar = 1.0 / (10**(snr/20)) * np.mean(np.abs(clean)) / np.mean(np.abs(noise)+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    return noisyspeech



def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''

    audio = normalize(audio, target_level)
    window_size = 50 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20*np.log10(sum(audio_win**2)+EPS)
        frame_energy_prob = 1./(1+np.exp(-(a+b*frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob*alpha_att + prev_energy_prob*(1-alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob*alpha_rel + prev_energy_prob*(1-alpha_rel)

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames/cnt
    return perc_active
