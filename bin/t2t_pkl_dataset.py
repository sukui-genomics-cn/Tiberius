import logging
import os
import pickle
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
import tensorflow as tf
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

from utils import tiberius_reduce_labels

baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
LABEL_NUM = 15
PADDING_TOKEN = "N"
PADDING_TOKEN_ID = 9

def random_sample_seq(seq: str, max_length: int, times: int = 1) -> Tuple[int, int]:
    if max_length // times != 0:
        max_length = (max_length // times) * times

    if len(seq) <= max_length:
        if len(seq) // times != 0:
            end_idx = (len(seq) // times) * times
        else:
            end_idx = max_length
        start_idx, end_idx = 0, end_idx
    else:
        start_idx = random.randint(0, len(seq) - max_length)
        end_idx = start_idx + max_length
    return start_idx, end_idx


class T2TTiberiusDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            dataset_name: str = "8K",
            split: str = "train",
            max_length: int = 512,
            seed: int = 42,
            **kwargs
    ):
        super(T2TTiberiusDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)

        data_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."

        with open(data_path, "r") as f:
            data = f.read().strip().split("\n")

        self.sequences = data
        self.max_length = max_length
        self.labels = None
        self.output_size = kwargs.get("output_size", LABEL_NUM)
        self.mlm = kwargs.get("mlm", False)
        self.mlm_probability = kwargs.get("mlm_probability", 0.15)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        with open(self.sequences[idx], "rb") as f:
            data = pickle.load(f)
        # TODO down sample with 0.5%
        assert "seq" in data, f"seq not in data: {data.keys()}"
        seq = data["seq"]
        label = data["annotation"].toarray()
        assert len(seq) == label.shape[0], f"seq len: {len(seq)}, label len: {label.shape[0]}"

        if len(seq) >= self.max_length:
            start_idx, end_idx = random_sample_seq(seq, self.max_length, times=9)
            seq = seq[start_idx:end_idx]
            label = label[start_idx:end_idx]
        else:
            pad_num = self.max_length - len(seq)
            seq += PADDING_TOKEN * pad_num
            if isinstance(label, np.ndarray):
                pad_label = np.zeros((pad_num, self.output_size))
                pad_label[:, 0] = 1
                label = np.concatenate([label, np.zeros((pad_num, self.output_size))], axis=0)
            elif isinstance(label, list):
                label = np.concatenate([label, np.zeros((pad_num, self.output_size))], axis=0)
                label = np.array(label, dtype=np.int64)
                label = np.eye(self.output_size)[label]
            else:
                raise ValueError(f"Unsupported label type: {type(label)}")
        input_ids = self.tokenizer(seq)
        input_ids = np.array(input_ids, dtype=np.int64)

        # reduce label
        if label.shape[-1] != self.output_size:
            label = tiberius_reduce_labels(label, self.output_size)

        return input_ids, label

    @staticmethod
    def tokenizer(sequence):
        table = np.zeros((256, 6), dtype=np.uint8)
        table[:, 4] = 1  # N is encoded as [0, 0, 0, 0, 1, 0]

        # Set specific labels for A, C, G, T
        table[ord('A'), :] = [1, 0, 0, 0, 0, 0]
        table[ord('C'), :] = [0, 1, 0, 0, 0, 0]
        table[ord('G'), :] = [0, 0, 1, 0, 0, 0]
        table[ord('T'), :] = [0, 0, 0, 1, 0, 0]
        # Set labels for a, c, g, t with softmasking indicator
        table[ord('a'), :] = [1, 0, 0, 0, 0, 1]
        table[ord('c'), :] = [0, 1, 0, 0, 0, 1]
        table[ord('g'), :] = [0, 0, 1, 0, 0, 1]
        table[ord('t'), :] = [0, 0, 0, 1, 0, 1]

        # Convert the sequence to integer sequence
        int_seq = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        # Perform one-hot encoding
        return table[int_seq]


class T2TTiberiusTfrecordDataset(T2TTiberiusDataset):
    def __getitem__(self, idx):
        with open(self.sequences[idx], "rb") as f:
            data = pickle.load(f)
        # TODO down sample with 0.5%
        assert "seq" in data, f"seq not in data: {data.keys()}"
        seq = str(data["seq"])
        if "anno" in data:
            label = data["anno"]
        else:
            label = data["annotation"]
        if getattr(label, "toarray", None) is not None:
            label = label.toarray()
        assert len(seq) == label.shape[0], f"seq len: {len(seq)}, label len: {label.shape[0]}"

        if len(seq) >= self.max_length:
            start_idx, end_idx = random_sample_seq(seq, self.max_length, times=9)
            seq = seq[start_idx:end_idx]
            label = label[start_idx:end_idx]
        else:
            logging.info(f"seq len: {len(seq)}, max_length: {self.max_length}")
            # pad_num = self.max_length - len(seq)
            seq = PADDING_TOKEN * self.max_length
            label = np.zeros(shape=(self.max_length, label.shape[1]), dtype=np.int64)
            label[:, 0] = 1
        # else:
        #     logging.warning(f"seq len: {len(seq)}, max_length: {self.max_length}")
        input_ids = self.tokenizer(seq)
        input_ids = np.array(input_ids, dtype=np.int64)

        # reduce label
        if label.shape[-1] != self.output_size:
            label = tiberius_reduce_labels(label, self.output_size)
        # label = np.array(label, dtype=np.int64)
        # label = np.eye(10)[label]
        # print(f"input_ids shape: {input_ids.shape}, label shape: {label.shape}")
        return input_ids, label

    def __len__(self):
        return len(self.sequences)


def pytorch_to_tensorflow_dataset(pytorch_dataset):
    # 使用tf.py_function来包装PyTorch Dataset的__getitem__方法
    def generator():
        for data, label in pytorch_dataset:
            yield tf.convert_to_tensor(data, dtype=tf.int32), tf.convert_to_tensor(label, dtype=tf.int32)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=[None, 6], dtype=tf.int32),
            tf.TensorSpec(shape=[None, LABEL_NUM], dtype=tf.int32)
        ))


if __name__ == '__main__':
    dest_path = "/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic"
    bend_data = T2TTiberiusDataset(
        dest_path=dest_path,
        dataset_name="homo_sapiens_tiberius_20K",
        split="test",
        max_length=9999,
        read_strand=False,
    )
    tf_dataset = pytorch_to_tensorflow_dataset(bend_data)
    tf_dataset = tf_dataset.batch(4).prefetch(tf.data.AUTOTUNE)
    first_batch = next(iter(tf_dataset.take(2)))
    features, labels = first_batch
    print(features.shape, labels.shape)

