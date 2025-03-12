import logging
import sys
import os

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius_backups/Tiberius/learnMSA")

from models import add_hmm_layer
import tensorflow as tf
from tensorflow.python.client import device_lib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(device_lib.list_local_devices())


# make a dummy pre-HMM model
def make_model(hmm_factor, batch_size, in_dim=5, out_dim=15):
    input = tf.keras.layers.Input(shape=(None, in_dim), name='main_input')
    output = tf.keras.layers.Dense(out_dim, activation="softmax", name='out')(input)
    pre_hmm_model = tf.keras.Model(inputs=input, outputs=output)
    model = add_hmm_layer(pre_hmm_model,
                          hmm_factor=hmm_factor,
                          output_size=out_dim,
                          batch_size=batch_size,
                          share_intron_parameters=False,
                          use_border_hints=False
                          )
    model.compile()
    return model, pre_hmm_model

def create_dummy_data(B=2, L=4):
    seqs = np.arange(B * L) % 5
    dummy_hmm_input = np.arange(B * L * 15) % 15
    seqs = seqs.reshape(B, L)
    seqs = np.eye(5)[seqs].astype(np.float32)
    dummy_hmm_input = dummy_hmm_input / np.max(abs(dummy_hmm_input))
    dummy_hmm_input = dummy_hmm_input.reshape(B, L, 15)
    seqs = tf.convert_to_tensor(seqs, dtype=tf.float32)
    dummy_hmm_input = tf.convert_to_tensor(dummy_hmm_input, dtype=tf.float32)
    return seqs, dummy_hmm_input


def run_hmm():
    model, pre_hmm_model = make_model(2, 4, out_dim=15)
    model.summary()
    hmm_layer = model.layers[-1]

    seqs, dummy_hmm_input = create_dummy_data()
    viterbi_seqs = hmm_layer.viterbi(dummy_hmm_input, seqs)
    print(f"viterbi_seqs shape: {viterbi_seqs.shape}\n{viterbi_seqs}")


if __name__ == '__main__':
    run_hmm()
