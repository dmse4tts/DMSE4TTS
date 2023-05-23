import numpy as np
from tqdm import tqdm

from data import AudioAugDataset
import json


def load_config(config_path):
    with open(config_path, 'r') as f:
        s = f.read()
    config = json.loads(s)
    return config

config = load_config('configs/cusent_no_text.json')
dataset = AudioAugDataset(config)

mels_mode_dict = dict()
mels_mode = dict()

mel_max = -100
mel_min = 100

for idx in tqdm(range(len(dataset))):
    x = dataset.__getitem__(idx)
    durations = x['durations']
    mel = x['clean']
    mel_max = max(mel_max, np.max(mel))
    mel_min = min(mel_min, np.min(mel))
    phonemes = x['phonemes']
    start_frame = 0
    for i in range(len(phonemes)):
        phoneme = phonemes[i]
        end_frame = start_frame + durations[i]
        if phoneme not in mels_mode_dict:
            mels_mode_dict[phoneme] = [
                np.round(np.median(mel[:, start_frame:end_frame], 1), 1)]
        else:
            mels_mode_dict[phoneme] += [
                np.round(np.median(mel[:, start_frame:end_frame], 1), 1)]
        start_frame = end_frame


for p in mels_mode_dict:
    print(p)
    mels_mode[p] = np.mean(np.asarray(mels_mode_dict[p]), 0)
    print(mels_mode[p])

with open("train_files/mels_avg_cusent.txt", "w+") as outfile:
    for p in mels_mode:
        msg = p + '\t' + str(mels_mode[p].tolist()) + '\n'
        outfile.write(msg)

print("mels_avg saved")
print("mel max and min:", mel_max, mel_min)
