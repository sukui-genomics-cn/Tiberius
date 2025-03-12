import logging
import sys

sys.path.append("/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius_backups/Tiberius/learnMSA")
from models import add_hmm_layer
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from tensorflow.python.client import device_lib

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


def gen_seqs(batch_size, seq_len):
    # i.i.d. nucleotides witout N
    ind = np.random.randint(0, 4, size=(batch_size, seq_len))
    ind = np.eye(5)[ind].astype(np.float32)
    return ind


def run_experiments(batch_size):
    count_compile_time = False  # whether mode compile time should be counted
    num_reps = 10  # number of repetitions for each experiment
    sqrt_lengths = list(range(2, 10)) + list(range(10, 101, 10))
    print(f"sqrt_lengths: {sqrt_lengths}")

    df = pd.DataFrame(index=np.arange(0, len(sqrt_lengths)),
                      columns=["L", "time_non_parallel", "time_parallel"])

    for i, sqrt_L in enumerate(sqrt_lengths):
        times = []
        for hmm_factor in [2, sqrt_L]:
            model, pre_hmm_model = make_model(hmm_factor, batch_size)
            hmm_layer = model.layers[-1]
            print(f"hmm layer: {hmm_layer}")
            L = 16
            seqs = gen_seqs(batch_size, L)
            dummy_hmm_input = pre_hmm_model(seqs)
            if not count_compile_time:
                # make one call to compile; do not count compile time
                print(f"seqs: {seqs.shape}, dummy_hmm_input: {dummy_hmm_input.shape}")
                print(f"seqs: {seqs[0]}, dummy_hmm_input: {dummy_hmm_input[0]}")
                viterbi_seqs = hmm_layer.viterbi(dummy_hmm_input, seqs)
            start = time.time()
            for _ in range(num_reps):
                viterbi_seqs = hmm_layer.viterbi(dummy_hmm_input, seqs)
            end = time.time()
            times.append((end - start) / num_reps)
            tf.keras.backend.clear_session()
        df.loc[i] = [L] + times

    return df


def plot(df, batch_size):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(f"Batch size {batch_size}")

    # general plot
    ax1.plot(df["L"], df["time_non_parallel"], label="non-parallel")
    ax1.plot(df["L"], df["time_parallel"], label="parallel")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Time (s)")
    ax1.legend()

    # plot only the small lengths
    ax2.plot(df["L"][:8], df["time_non_parallel"][:8], label="non-parallel")
    ax2.plot(df["L"][:8], df["time_parallel"][:8], label="parallel")
    ax2.legend()


def main():
    # model_1, _ = make_model(1, 32)
    # model_1.summary()

    b = 2
    df = run_experiments(b)
    plot(df, b)


if __name__ == '__main__':
    main()
