"""adapted from https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/inference.py"""

import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
import json
import librosa

from model import DiffRefiner
from preprocessor import Preprocessor
from data.dataset import get_mel, AudioAugDataset
from utils import save_plot

from hifigan_vocoder.env import AttrDict
from hifigan_vocoder.models import Generator as HiFiGAN
import logging
import os

import argparse


HIFIGAN_CONFIG = './hifigan_vocoder/log/cusent_vctk_lr_1e-6/config.json'
HIFIGAN_CHECKPT = './hifigan_vocoder/log/cusent_vctk_lr_1e-6/g_00220000'


parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_config', '-c', default='./configs/cusent_train.json',
   )
parser.add_argument(
    '--test_dir', default='./test_files/testsets_vctk/noisy_testset_wav',
)
parser.add_argument(
    '--restore_file', type=int, default=1000, required=True
    )
parser.add_argument(
    '-t', '--timesteps', type=int, required=False,
    default=25, help='number of timesteps of reverse diffusion'
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        s = f.read()
    config = json.loads(s)
    return config

args = parser.parse_args()
config = load_config(args.train_config)
testdir = args.test_dir
restore_file = args.restore_file

log_dir = config["training"]["log_dir"]

n_mels = config["preprocessing"]["mel"]["n_mel_channels"]
n_fft = config["preprocessing"]["stft"]["n_fft"]
sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
hop_length = config["preprocessing"]["stft"]["hop_length"]
win_length = config["preprocessing"]["stft"]["win_length"]
f_min = config["preprocessing"]["mel"]["mel_fmin"]
f_max = config["preprocessing"]["mel"]["mel_fmax"]

dec_dim = config["model"]["dec_dim"]
beta_min = config["model"]["beta_min"]
beta_max = config["model"]["beta_max"]
pe_scale = config["model"]["pe_scale"]
use_text = bool(config["model"]["use_text"])


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


mel_max = 2.5
mel_min = -12.5
mel_std = (mel_max - mel_min)/2.0
mel_mean = (mel_max + mel_min)/2.0


def mel_normalize(mel):
    return (mel - mel_mean)/mel_std


def mel_recover(normalized_mel):
    return normalized_mel * mel_std + mel_mean


if __name__ == "__main__":
    outdir = os.path.join("out", "test_epoch_{}_{}_{}_dpmsolver_retrain_vocoder".format(os.path.basename(args.train_config)[:-5], restore_file, testdir.split('/')[1]))
    os.makedirs(outdir, exist_ok=True)

    mypreprocessor = Preprocessor(config)
    mydataset = AudioAugDataset(config)

    print('Initializing model...')

    model = DiffRefiner(n_mels, use_text, dec_dim, beta_min, beta_max, pe_scale).cuda()

    print("Restoring parameters from epoch {}".format(restore_file))
    with open(f'{log_dir}/test.log', 'a') as f:
        f.write("Restoring parameters from epoch {}".format(restore_file))
    model_ckpt_path = os.path.join(log_dir, 'grad_{}.pt'.format(restore_file))
    model.load_state_dict(torch.load(model_ckpt_path))

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(
        HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    print('Start testing...')

    model.eval()
    print('Synthesis...')
        
    model.eval()
    with torch.no_grad():
        for wavfile in os.listdir(os.path.join(testdir)):
            audiopath = os.path.join(testdir, wavfile)
            index = os.path.basename(audiopath)[:-4]
            print(index)

            noisy_wav, _ = librosa.load(audiopath, sr=22050)
            noisy_mel = get_mel(noisy_wav)
            mel_len = noisy_mel.shape[1]

            y = torch.FloatTensor(noisy_mel).unsqueeze(0).cuda()
            save_plot(y.squeeze().cpu(),
                      '{}/noisy_{}.png'.format(outdir, index))
            y = mel_normalize(y)

            mu = None

            y_lengths = torch.LongTensor([y.shape[-1]]).cuda()
           
            denoised = model(
                y, y_lengths, n_timesteps=args.timesteps, temperature=1.2, mu=mu, stoc=False)

            denoised = mel_recover(denoised)
            save_plot(denoised.squeeze().cpu(),
                      '{}/denoised_{}.png'.format(outdir, index))

            audio = (vocoder.forward(denoised).cpu().squeeze(
            ).clamp(-1, 1).numpy() * 32768).astype(np.int16)

            audiopath = os.path.join(outdir, "{}.wav".format(index))
            write(audiopath, 22050, audio)
            print(audiopath)


