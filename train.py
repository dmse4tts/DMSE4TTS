""" adapted from https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/train.py """

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import json

from model import DiffRefiner, fix_len_compatibility
from data import AudioAugDataset, BatchCollate, PairedCleanNoisyDataset
from utils import save_plot
import logging
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_config', default='configs/cusent_train.json',
)

parser.add_argument(
    '--restore_file', type=int, default=0
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        s = f.read()
    config = json.loads(s)
    return config


args = parser.parse_args()
config = load_config(args.train_config)
restore_file = args.restore_file

n_epochs = config["training"]["n_epochs"]
batch_size = config["training"]["batch_size"]
out_size = fix_len_compatibility(
    int(config["training"]["out_size"])*22050//256)
learning_rate = config["training"]["learning_rate"]
random_seed = config["training"]["random_seed"]
save_every = config["training"]["save_every"]
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
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    os.makedirs(log_dir, exist_ok=True)
    # logger = SummaryWriter(log_dir=log_dir)
    set_logger(f'{log_dir}/train.log')

    print('Initializing data loaders...')
    batch_collate = BatchCollate()

    train_dataset = AudioAugDataset(config)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=batch_collate, drop_last=True, num_workers=16
    )

    test_dataset = PairedCleanNoisyDataset(config)

    print('Initializing model...')

    model = DiffRefiner(n_mels, use_text, dec_dim,
                        beta_min, beta_max, pe_scale).cuda()

    logging.info('Number of decoder parameters = %.2fm' %
                 (model.decoder.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if restore_file > 0:
        logging.info("Restoring parameters from epoch {}".format(restore_file))
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write("Restoring parameters from epoch {}".format(restore_file))
        model_ckpt_path = os.path.join(
            log_dir, 'grad_{}.pt'.format(restore_file))
        opt_ckpt_path = os.path.join(
            log_dir, 'opt_{}.pt'.format(restore_file))
        model.load_state_dict(torch.load(model_ckpt_path))
        optimizer.load_state_dict(torch.load(opt_ckpt_path))

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=4)
    for item in test_batch:
        clean, noisy, index = item['clean'], item['noisy'], item['index']
    
        index = os.path.basename(index)
        clean = mel_normalize(clean)
        noisy = mel_normalize(noisy)
        

        save_plot(clean.squeeze(), '{}/clean_{}.png'.format(log_dir, index))
        save_plot(noisy.squeeze(), '{}/noisy_{}.png'.format(log_dir, index))
        if use_text:
            text = item['text']
            text = mel_normalize(text)
            save_plot(text.squeeze(), '{}/text_{}.png'.format(log_dir, index))

    print('Start training...')
    iteration = 0
    for epoch in range(1+restore_file, n_epochs + 1):
        if epoch % 10 == 1:
            model.eval()
            print('Synthesis...')
            with torch.no_grad():
                for item in test_batch:
                    y = torch.FloatTensor(item['noisy']).unsqueeze(0).cuda()
                    y = mel_normalize(y)

                    if not use_text:
                        mu = None
                    else:
                        mu = torch.FloatTensor(
                            item['text']).unsqueeze(0).cuda()
                        mu = mel_normalize(mu)

                    y_lengths = torch.LongTensor([y.shape[-1]]).cuda()
                    index = item['index']

                    x = model(
                        y, y_lengths, n_timesteps=25, mu=mu)

                    # x = mel_recover(x)
                    save_plot(x.squeeze().cpu(),
                              '{}/denoised_{}.png'.format(log_dir, index))
                    # save_plot(y.squeeze().cpu(),
                    #          '{}/corrupted_{}.png'.format(log_dir, index))

        model.train()

        diff_losses = []
        with tqdm(train_dataloader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x = batch['clean'].cuda()
                y = batch['noisy'].cuda()

                x = mel_normalize(x)
                y = mel_normalize(y)

                y_lengths = batch['mel_lengths'].cuda()
                if not use_text:
                    mu = None
                else:
                    mu = batch['text'].cuda()
                    mu = mel_normalize(mu)

                diff_loss = model.compute_loss(
                    x, y, y_lengths, mu=mu, out_size=out_size)
                loss = diff_loss
                loss.backward()

                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                msg = f'Epoch: {epoch}, iteration: {iteration} | diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)

                diff_losses.append(diff_loss.item())
                iteration += 1

        msg = 'Epoch %d: diff loss = %.3f ' % (epoch, np.mean(diff_losses))

        logging.info(msg)

        if epoch % save_every > 0:
            continue

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
        opt_ckpt = optimizer.state_dict()
        torch.save(opt_ckpt, f=f"{log_dir}/opt_{epoch}.pt")
