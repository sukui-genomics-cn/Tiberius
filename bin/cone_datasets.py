import sys, json, os, re, sys, csv, argparse, requests, time, logging, warnings, pickle
import subprocess as sp
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoModel
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class TrainDataset:
    def __init__(self, pkl_paths="/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic/homo_sapiens_tiberius_pkls_from_tfrecord_10K",
                 split='train', batch_size=8, buffer_size=1000, repeat=True):
        """
        Custom dataset for loading genomics data from .pkl files with generator as TensorFlow Dataset.

        Args:
            pkl_paths (string): Path to directory where splits files are.
            split (string): Which split is being used(train, val, test)
            batch_size (int): Batch size for the dataset.
            buffer_size (int): Buffer size for shuffling the dataset.
        """
        split = split + '.txt'
        file_path = os.path.join(pkl_paths, split)
        self.repeat = repeat
        
        # Open the file and read each line
        with open(file_path, 'r') as file:
            self.pkl_paths = file.readlines() 

        self.pkl_paths = [line.strip() for line in self.pkl_paths if '.pkl' in line]

        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def _parse_pkl_file(self, pkl_path):
        """Load data from a .pkl file."""
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)
        input_ids = data['input_ids'] 
        annotation = data['annotation'] 
        return input_ids, annotation

    def _generator(self):
        """Generator that loads and yields data."""
        for pkl_path in self.pkl_paths:
            input_ids, annotation = self._parse_pkl_file(pkl_path)
            yield (
                tf.convert_to_tensor(input_ids, dtype=tf.int32),
                tf.convert_to_tensor(annotation, dtype=tf.int32)
            )

    def create_dataset(self):
        """Create a TensorFlow dataset from the generator."""
        dataset = tf.data.Dataset.from_generator(
            self._generator, 
            output_signature=(
                tf.TensorSpec(shape=(9999, 6), dtype=tf.int32),   # For input_ids
                tf.TensorSpec(shape=(9999, 15), dtype=tf.int32)   # For annotation 
            )
        )
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        
        if self.repeat:
            dataset = dataset.repeat()  # Repeat indefinitely for training
        else:
            dataset = dataset.repeat(1)  # Do not repeat for validation

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.pkl_paths)

class TokenizerDataset:
    def __init__(self, max_length=9999, pkl_paths="/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic/homo_sapiens_tiberius_pkls_from_tfrecord_10K",
                 split='train', batch_size=8, buffer_size=1000, repeat=True):
        """
        Custom dataset for loading genomics data from .pkl files with generator as TensorFlow Dataset.

        Args:
            pkl_paths (string): Path to directory where splits files are.
            split (string): Which split is being used(train, val, test)
            batch_size (int): Batch size for the dataset.
            buffer_size (int): Buffer size for shuffling the dataset.
        """
        split = split + '.txt'
        file_path = os.path.join(pkl_paths, split)
        self.max_length = max_length
        self.repeat = repeat
        self.token_to_id = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.vocab_size = 6
        
        # Open the file and read each line
        with open(file_path, 'r') as file:
            self.pkl_paths = file.readlines() 

        self.pkl_paths = [line.strip() for line in self.pkl_paths if '.pkl' in line]

        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def convert_to_input_ids(self, seq):
        input_ids = []
        for c in seq:
            curr = np.zeros(6, dtype=np.int32)
            curr[self.token_to_id.get(c.upper(), 'N')] = 1
            if c != c.upper():
                curr[5] = 1
            input_ids.append(curr)
        
        input_ids = tf.constant(input_ids, dtype=tf.int32)
        return input_ids


    def _parse_pkl_file(self, pkl_path):
        """
            Load data from a .pkl file and return one-hot encoded input_ids along with annotation.
        """
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)
        
        seq = data['seq']
        annotation = data['annotation']
        annotation = annotation.toarray()  # Converts sparse to dense (NumPy array)
        

        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        
        input_ids = self.convert_to_input_ids(seq)

        return input_ids, annotation


    def _generator(self):
        """Generator that loads and yields data."""
        for pkl_path in self.pkl_paths:
            input_ids, annotation = self._parse_pkl_file(pkl_path)
            yield (
                tf.convert_to_tensor(input_ids, dtype=tf.int32),
                tf.convert_to_tensor(annotation, dtype=tf.int32)
            )

    def create_dataset(self):
        """Create a TensorFlow dataset from the generator."""
        dataset = tf.data.Dataset.from_generator(
            self._generator, 
            output_signature=(
                tf.TensorSpec(shape=(9999, 6), dtype=tf.int32),   # For input_ids
                tf.TensorSpec(shape=(9999, 15), dtype=tf.int32)   # For annotation 
            )
        )
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        
        if self.repeat:
            dataset = dataset.repeat()  # Repeat indefinitely for training
        else:
            dataset = dataset.repeat(1)  # Do not repeat for validation

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.pkl_paths)


class OneChrDataset:
    def __init__(self, pkl_dir_path="/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Tiberius/pkls/Homo_sapiens_hmm",
                 chr='chr22', batch_size=8, buffer_size=1000):
        """
        Custom dataset for loading genomics data from .pkl files with generator as TensorFlow Dataset.

        Args:
            pkl_dir_path (string): Path to directory where .pkl files are.
            chr (string): Name of chromosome that is being used for inference
            batch_size (int): Batch size for the dataset.
            buffer_size (int): Buffer size for shuffling the dataset.
        """
        self.pkl_paths = [
            os.path.join(pkl_dir_path, f) 
            for f in os.listdir(pkl_dir_path) 
            if chr in f
        ]
        self.chr = chr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def _parse_pkl_file(self, pkl_path):
        """Load data from a .pkl file."""
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)
        input_ids = data['input_id'] 
        annotation = data['annotation'] 
        return input_ids, annotation

    def _generator(self):
        """Generator that loads and yields data."""
        for pkl_path in self.pkl_paths:
            input_ids, annotation = self._parse_pkl_file(pkl_path)

            yield (
                tf.convert_to_tensor(input_ids, dtype=tf.int32), 
                tf.convert_to_tensor(annotation, dtype=tf.int32)
            )

    def create_dataset(self):
        """Create a TensorFlow dataset from the generator."""
        dataset = tf.data.Dataset.from_generator(
            self._generator, 
            output_signature=(
                tf.TensorSpec(shape=(9999,6), dtype=tf.int32),  # For input_ids
                tf.TensorSpec(shape=(9999,15), dtype=tf.int32)   # For annotation (or adjust dtype as necessary)
            )
        )
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.pkl_paths)


