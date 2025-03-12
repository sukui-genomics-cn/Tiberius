import logging
import os
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pysam
from tqdm.auto import tqdm

from torch.utils.data import Dataset
import tensorflow as tf
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
import h5py

baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}


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


def reverse_complement(dna_string: str):
    # """Returns the reverse-complement for a DNA string."""
    """
    Returns the reverse-complement for a DNA string.

    Parameters
    ----------
    dna_string : str
        DNA string to reverse-complement.

    Returns
    -------
    str
        Reverse-complement of the input DNA string.
    """

    complement = [baseComplement.get(base, 'N') for base in dna_string]
    reversed_complement = reversed(complement)
    return ''.join(list(reversed_complement))


class Fasta():
    """Class for fetching sequences from a reference genome fasta file."""

    def __init__(self, fasta) -> None:
        """
        Initialize a Fasta object for fetching sequences from a reference genome fasta file.

        Parameters
        ----------
        fasta : str
            Path to a reference genome fasta file.
        """

        self._fasta = pysam.FastaFile(fasta)

    def fetch(self, chrom: str, start: int, end: int, strand: str = '+', flank: int = 0) -> str:
        """
        Fetch a sequence from the reference genome fasta file.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start : int
            Start coordinate.
        end : int
            End coordinate.
        strand : str, optional
            Strand. The default is '+'.
            If strand is '-', the sequence will be reverse-complemented before returning.
        flank : int, optional
            Number of bases to add to the start and end coordinates. The default is 0.
        Returns
        -------
        str
            Sequence from the reference genome fasta file.
        """
        sequence = self._fasta.fetch(str(chrom), start - flank, end + flank).upper()

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')

        return sequence


class BendDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            tokenizer: PreTrainedTokenizerBase = None,
            dataset_name: str = "gene_finding",
            split: str = "train",
            max_length: int = 1024,
            read_strand=False,
            flank=0,
            **kwargs
    ):
        super(BendDataset, self).__init__()
        assert split in ["train", "valid", "test"], f"Split {split} not in train, valid, test."
        self.max_length = max_length
        fasta_file = os.path.join(dest_path, dataset_name, "GRCh38.primary_assembly.genome.fa")
        assert os.path.exists(fasta_file), f"File {fasta_file} does not exist."
        bed_file = os.path.join(dest_path, dataset_name, "gene_finding.bed")
        assert os.path.exists(bed_file), f"File {bed_file} does not exist."
        anna_path = os.path.join(dest_path, dataset_name, "gene_finding.bed")
        assert os.path.exists(anna_path), f"File {anna_path} does not exist."

        fasta = Fasta(fasta_file)

        seq_data = pd.read_csv(bed_file, header='infer', sep='\t', low_memory=False)
        mask = (seq_data.iloc[:, -1] == split)
        split_seq_data = seq_data[mask]

        label_path = os.path.join(dest_path, dataset_name, "gene_finding.hdf5")
        h5_file = h5py.File(label_path, "r")
        assert "labels" in h5_file.keys()
        h5_labels = h5_file["labels"]
        strand_column_idx = split_seq_data.columns.get_loc('strand') if 'strand' in split_seq_data.columns else 3

        self.sequences = []
        self.labels = []

        for idx, line in tqdm(split_seq_data.iterrows(), total=len(split_seq_data), desc=f'load {split} sequences'):
            # get bed row
            if read_strand:
                chrom, start, end, strand = line.iloc[0], int(line.iloc[1]), int(line.iloc[2]), line.iloc[
                    strand_column_idx]
            else:
                chrom, start, end, strand = line.iloc[0], int(line.iloc[1]), int(line.iloc[2]), '+'
            label = h5_labels[idx]
            sequence = fasta.fetch(chrom, start, end, strand=strand, flank=flank)
            if len(sequence) == label.size:
                self.labels.append(np.asarray(label, dtype=np.int64))
                self.sequences.append(sequence)
            else:
                print(f"idx: {idx}, chrom: {chrom}, start: {start}, end: {end}, strand: {strand}")
                print(f"seq len: {len(sequence)}, label len: {label.size}")

        self._data_collator = DataCollatorWithPadding(self.tokenizer)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx].tolist()
        assert len(seq) == len(label), f"seq len: {len(seq)}, label len: {len(label)}"

        if len(seq) > self.max_length:
            start_idx, end_idx = random_sample_seq(seq, self.max_length)
            seq = seq[start_idx:end_idx]
            label = label[start_idx:end_idx]
        else:
            pad_num  = self.max_length - len(seq)
            seq += 'N' * pad_num
            label += [9] * pad_num
        input_ids = self.tokenizer(seq)
        input_ids = np.array(input_ids, dtype=np.int64)
        label = np.array(label, dtype=np.int64)
        label = np.eye(10)[label]
        # print(f"input_ids shape: {input_ids.shape}, label shape: {label.shape}")
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

    def data_collator(self, features: List[Dict[str, Any]]):
        return self._data_collator(features)


def pytorch_to_tensorflow_dataset(pytorch_dataset):
    # 使用tf.py_function来包装PyTorch Dataset的__getitem__方法
    def generator():
        for data, label in pytorch_dataset:
            yield tf.convert_to_tensor(data, dtype=tf.int32), tf.convert_to_tensor(label, dtype=tf.int32)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=[None, 6], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 10], dtype=tf.int32)
        ))


if __name__ == '__main__':
    dest_path = "/home/share/huadjyin/home/s_liulin4/datasets/Gene_annotation_test/BEND"
    bend_data = BendDataset(
        dest_path=dest_path,
        split="test",
        dataset_name="gene_finding",
        max_length=9999,
        read_strand=False,
    )
    tf_dataset = pytorch_to_tensorflow_dataset(bend_data)
    tf_dataset = tf_dataset.batch(4).prefetch(tf.data.AUTOTUNE)
    first_batch = next(iter(tf_dataset.take(2)))
    features, labels = first_batch
    print(features.shape, labels.shape)

# find . -name 'Homo_sapiens_chr*' -print0 | xargs -0 -I {} mv {} ~/02_data/07_genomics_data/Tiberius/pkls/Homo_sapiens/