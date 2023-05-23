""" from https://github.com/haoheliu/voicefixer_main/blob/main/tools/dsp/lowpass.py """

from scipy.signal import butter
import librosa
import numpy as np

from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly

# ----------------------------------------------------------------
# README:
# Define several functions for data generation.
# ----------------------------------------------------------------

def align_length(x, y):
    """ align the length of y to that of x
    Args:
        x (np.array): reference signal
        y (np.array): the signal needs to be length aligned
    Return:
        y (np.array): signal with the same length as x
    """
    Lx = len(x)
    Ly = len(y)

    if Lx == Ly:
        return y
    elif Lx > Ly:
        # pad y with zeros
        return np.pad(y, (0, Lx-Ly), mode='constant')
    else:
        # cut y
        return y[:Lx]


def lowpass_filter(x, highcut, fs, order, ftype):
    """ process input signal x using lowpass filter

    Args:
        x (np.array): input signal
        highcut (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    hi = highcut / nyq

    if ftype == 'butter':
        sos = butter(order, hi, btype='low', output='sos')
    elif ftype == 'cheby1':
        sos = cheby1(order, 0.1, hi, btype='low', output='sos')
    elif ftype == 'cheby2':
        sos = cheby2(order, 60, hi, btype='low', output='sos')
    elif ftype == 'ellip':
        sos = ellip(order, 0.1, 60, hi, btype='low', output='sos')
    elif ftype == 'bessel':
        sos = bessel(order, hi, btype='low', output='sos')
    else:
        raise Exception(f'The lowpass filter {ftype} is not supported!')

    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)

    return y


def highpass_filter(x, lowcut, fs, order, ftype):
    """ process input signal x using highpass filter

    Args:
        x (np.array): input signal
        lowcut (float): low cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq

    if ftype == 'butter':
        sos = butter(order, low, btype='highpass', output='sos')
    elif ftype == 'cheby1':
        sos = cheby1(order, 0.1, low, btype='highpass', output='sos')
    elif ftype == 'cheby2':
        sos = cheby2(order, 60, low, btype='highpass', output='sos')
    elif ftype == 'ellip':
        sos = ellip(order, 0.1, 60, low, btype='highpass', output='sos')
    elif ftype == 'bessel':
        sos = bessel(order, low, btype='highpass', output='sos')
    else:
        raise Exception(f'The highpass filter {ftype} is not supported!')

    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)

    return y


def bandpass_filter(x, bandwidth, fs, order, ftype):
    """ process input signal x using bandpass filter

    Args:
        x (np.array): input signal
        bandwidth (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    low, hi = bandwidth
    hi = hi / nyq
    low = low / nyq

    if ftype == 'butter':
        sos = butter(order, [low, hi], btype='bandpass', output='sos')
    elif ftype == 'cheby1':
        sos = cheby1(order, 0.1, [low, hi], btype='bandpass', output='sos')
    elif ftype == 'cheby2':
        sos = cheby2(order, 60, [low, hi], btype='bandpass', output='sos')
    elif ftype == 'ellip':
        sos = ellip(order, 0.1, 60, [low, hi], btype='bandpass', output='sos')
    elif ftype == 'bessel':
        sos = bessel(order, [low, hi], btype='bandpass', output='sos')
    else:
        raise Exception(f'The lowpass filter {ftype} is not supported!')

    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)

    return y


def stft_hard_lowpass(data, lowpass_ratio, fs_ori=44100):
    fs_down = int(lowpass_ratio * fs_ori)
    # downsample to the low sampling rate
    y = resample_poly(data, fs_down, fs_ori)

    # upsample to the original sampling rate
    y = resample_poly(y, fs_ori, fs_down)

    if len(y) != len(data):
        y = align_length(data, y)
    return y


def limit(integer, high, low):
    if(integer > high):
        return high
    elif(integer < low):
        return low
    else:
        return int(integer)


def lowpass(data, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param highcut: cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :return: filtered data, (samples,)
    """
    assert _type in ["butter", "cheby1", "cheby2", "ellip", "bessel", "stft"], "Error: Unexpected filter type {}".format(_type)

    if _type != "stft":
        order = limit(order, high=10, low=2)
        return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype=_type)
    else:
        return stft_hard_lowpass(data, lowpass_ratio=highcut / int(fs / 2))

if __name__ == "__main__":
    import librosa
    import soundfile as sf

    audiopath = "/home/anonymous/Projects/diffrefiner/clean.wav"
    wav, sr = librosa.load(audiopath, sr=None)
    print(wav.shape)

    wav = lowpass_filter(wav, 3000, fs=sr, order=2, ftype="bessel")

    sf.write('/home/anonymous/Projects/diffrefiner/lowpass.wav', wav, sr)
