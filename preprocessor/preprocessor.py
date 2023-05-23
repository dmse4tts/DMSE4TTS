import os
import random
import math

from praatio import tgio
import random


class Preprocessor:
    def __init__(self, config):
        self.config = config
        # Path
        self.clean_flist = config["path"]["clean_flist"]
        self.tgt_dir = config["path"]["textgrid_path"]
        self.out_file = config["path"]["dataset_file_path"]
        # audio preprocessing parameters
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]

    def build_from_path(self):
        print("Processing Data ...")
        out = list()
        n_frames = 0

        with open(self.clean_flist) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                audiopath = line.strip()

                basename = os.path.basename(audiopath).split(".")[0]
                speaker = audiopath.split('/')[-2]
                tg_path = os.path.join(
                    self.tgt_dir, speaker, "{}.TextGrid".format(basename)
                )
                print(tg_path)
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, tg_path, audiopath)
                    if ret is None:
                        continue
                    else:
                        info, n = ret
                    out.append(info)

                n_frames += n

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(self.out_file, "w", encoding="utf-8") as f:
            for m in out:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, tgpath, audiopath):
        # Get alignments
        textgrid = tgio.openTextgrid(tgpath)
        phone, durations, stamps = self.get_alignment(
            textgrid.tierDict["phones"].entryList
        )
        total_seconds = sum(durations) * self.hop_length / float(self.sampling_rate)
        if total_seconds < 1.5:
            return None
        text = " ".join(phone)

        stamps_str = " ".join([str(stamp) for stamp in stamps])
        durations_str = " ".join([str(duration) for duration in durations])

        return (
            "|".join([stamps_str, durations_str,
                     text, audiopath, speaker]),
            sum(durations),
        )

    def get_alignment(self, tier):
        def time2frame(t):
            return math.floor((t+1e-9) * self.sampling_rate / self.hop_length)

        sil_phones = ["", "sp", "spn"]

        phones = []
        stamps = []
        durations = []

        for t in tier:
            s, e, p = t.start, t.end, t.label

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
            else:
                # For silent phones
                phones.append('sil')

            start_frame = time2frame(s)
            end_frame = max(time2frame(e), start_frame+1)

            stamps.append(start_frame)
            durations.append(end_frame-start_frame)

        return phones, durations, stamps
