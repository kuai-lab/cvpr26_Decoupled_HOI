import codecs as cs
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "processed_data"


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class DecHOIEvaluationDataset(Dataset):
    def __init__(self, res_npz_folder, word_vectorizer=None, data_root_folder=None):
        self.res_npz_folder = Path(res_npz_folder)
        self.data_root_folder = Path(
            data_root_folder or os.environ.get("DECHOI_PROCESSED_DATA", str(DEFAULT_DATA_ROOT))
        )
        self.language_anno_folder = self.data_root_folder / "omomo_text_anno_txt_data"
        self.window_data_dict = self.load_res_npz_files()
        self.w_vectorizer = word_vectorizer

        mean_std_jpos_path = self.data_root_folder / "t2m_mean_std_jpos.p"
        mean_std_dict = joblib.load(mean_std_jpos_path)
        self.mean_jpos = torch.from_numpy(mean_std_dict["jpos_mean"]).float()
        self.std_jpos = torch.from_numpy(mean_std_dict["jpos_std"]).float()

    def load_res_npz_files(self):
        window_data_dict = {}
        for idx, npz_path in enumerate(sorted(self.res_npz_folder.glob("*.npz"))):
            npz_data = np.load(npz_path)
            window_data_dict[idx] = {
                "global_jpos": npz_data["global_jpos"],
                "seq_name": str(npz_data["seq_name"]),
            }
        return window_data_dict

    def load_language_annotation(self, seq_name):
        txt_path = self.language_anno_folder / f"{seq_name}.txt"
        with cs.open(txt_path) as handle:
            last_line = handle.readlines()[-1]

        caption, token_string = last_line.strip().split("#")
        return {
            "caption": caption,
            "tokens": token_string.split(" "),
        }

    def __len__(self):
        return len(self.window_data_dict)

    def normalize_jpos_mean_std(self, ori_jpos):
        if ori_jpos.dim() == 3:
            return (ori_jpos - self.mean_jpos[None, None, :]) / self.std_jpos[None, None, :]
        return (ori_jpos - self.mean_jpos[None, :]) / self.std_jpos[None, :]

    def de_normalize_jpos_mean_std(self, norm_jpos):
        if norm_jpos.dim() == 3:
            return norm_jpos * self.std_jpos[None, None, :] + self.mean_jpos[None, None, :]
        return norm_jpos * self.std_jpos[None, :] + self.mean_jpos[None, :]

    def __getitem__(self, index):
        ori_jpos = self.window_data_dict[index]["global_jpos"].reshape(-1, 24 * 3)
        ori_jpos = torch.from_numpy(ori_jpos).float()
        data_input = self.normalize_jpos_mean_std(ori_jpos)

        seq_name = self.window_data_dict[index]["seq_name"]
        actual_steps = data_input.shape[0]

        text_data = self.load_language_annotation(seq_name)
        caption, tokens = text_data["caption"], text_data["tokens"]

        max_text_len = 30
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])

        return (
            np.concatenate(word_embeddings, axis=0),
            np.concatenate(pos_one_hots, axis=0),
            caption,
            sent_len,
            data_input,
            actual_steps,
            "_".join(tokens),
        )
