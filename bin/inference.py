import sys, json, os, re, sys, csv, argparse, requests, time, logging, warnings
script_dir = os.path.dirname(os.path.realpath(__file__))
import subprocess as sp
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoModel
import tensorflow as tf
from tqdm import tqdm

from tiberius import parseCmd
from utils import cal_metric
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from cone_datasets import OneChrDataset, TrainDataset, TokenizerDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

import pickle
def configure_gpu(gpu_index=1):
    """
    Configures TensorFlow to use a specific GPU by its index.

    Args:
        gpu_index (int): The index of the GPU to use (0-based).
    """
    try:
        # List all physical GPUs
        physical_gpus = tf.config.list_physical_devices('GPU')
        if not physical_gpus:
            raise RuntimeError("No GPU devices found.")

        print(f"Available GPUs: {[gpu.name for gpu in physical_gpus]}")

        # Check if the specified GPU index exists
        if gpu_index >= len(physical_gpus) or gpu_index < 0:
            raise ValueError(f"GPU index {gpu_index} is out of range. Available GPUs: 0 to {len(physical_gpus)-1}")

        # Select the specified GPU
        selected_gpu = physical_gpus[gpu_index]
        tf.config.set_visible_devices(selected_gpu, 'GPU')
        print(f"Selected GPU: {selected_gpu.name}")

        # Enable memory growth for the selected GPU
        tf.config.experimental.set_memory_growth(selected_gpu, True)
        print(f"Memory growth enabled for GPU: {selected_gpu.name}")

        # Verify that TensorFlow sees only the selected GPU
        visible_gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow sees {len(visible_gpus)} GPU(s): {[gpu.name for gpu in visible_gpus]}")

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(f"RuntimeError: {e}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")


def verify_gpu_configuration():
    """
    Verifies the current GPU configuration in TensorFlow.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Currently visible GPUs: {[gpu.name for gpu in gpus]}")
    else:
        print("No GPUs are currently visible to TensorFlow.")



def add_metrics_to_model(model):
    current_optimizer = model.optimizer
    current_loss = model.loss
    current_metrics = model.metrics

    additional_metrics = ['accuracy', 'AUC']  # Add any other metrics you want

    # Recompile the model with the same optimizer and loss, and add new metrics
    model.compile(
        optimizer=current_optimizer,
        loss=current_loss,
        metrics=[
            'accuracy'
        ]
    )


def custom_evaluate(model, dataset, steps=None):
    """
    Custom evaluation loop that computes metrics and stores predictions and labels.

    Args:
        model (tf.keras.Model): The compiled TensorFlow model to evaluate.
        dataset (tf.data.Dataset): The dataset to evaluate on.
        steps (int, optional): Number of steps (batches) to evaluate. If None, evaluate on the entire dataset.

    Returns:
        dict: A dictionary of metric results.
        np.ndarray: Array of all predictions.
        np.ndarray: Array of all true labels.
    """
    # Initialize the progress bar
    if steps is not None:
        prog_bar = tqdm(total=steps, desc='Evaluating', unit='batch')
    else:
        prog_bar = tqdm(desc='Evaluating', unit='batch')

    # Reset the metrics at the start of the evaluation
    model.reset_metrics()

    all_predictions = []
    all_labels = []

    for step, (inputs, targets) in enumerate(dataset):
        if steps is not None and step >= steps:
            break

        # Perform a forward pass to get predictions
        predictions = model(inputs, training=False)

        # Update the model's metrics
        model.compiled_metrics.update_state(targets, predictions)

        # Store predictions and labels
        all_predictions.append(predictions.numpy())
        all_labels.append(targets.numpy())

        # Update the progress bar
        prog_bar.update(1)

    prog_bar.close()

    # Aggregate all predictions and labels
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_predictions = np.array([])
        all_labels = np.array([])

    # Collect the metrics
    results = {metric.name: metric.result().numpy() for metric in model.metrics}

    return results, all_predictions, all_labels



def main():

    # Configure TensorFlow to use GPU 2 and verify that
    configure_gpu(gpu_index=1)
    verify_gpu_configuration()

    start_time = time.time()
    args = parseCmd()  

    logging.info(f"Model path: {args.model}")

    print(f"learMSA path: {args.learnMSA}")
    sys.path.insert(0, args.learnMSA)

    from eval_model_class import PredictionGTF
    from models import make_weighted_cce_loss, custom_cce_f1_loss        
    from genome_anno import Anno

    model_path = os.path.abspath(args.model) if args.model else None

    assert model_path is not None, "You must provide model for inference"

    custom_objects = {}
    f1_factor = 2
    if f1_factor:
        cce_loss = custom_cce_f1_loss(2, batch_size=args.batch_size)
        custom_objects['custom_cce_f1_loss'] = cce_loss
        custom_objects['loss_'] = cce_loss
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()


    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )

    model.summary()

    #add_metrics_to_model(model)

    dataset = TokenizerDataset(pkl_paths="/home/share/huadjyin/home/nemanjaudovic/datasets/Mus_musculus", batch_size=64, split='val', repeat=False).create_dataset()


    #     # Assuming you have custom_evaluate that returns y_pred and y_true
    results, y_pred, y_true = custom_evaluate(model, dataset)
    # print(results)

    # # Get class indices (e.g., the class with highest probability)
    # y_pred_class = tf.argmax(y_pred, axis=-1)  # Shape: [batch_size,]
    # y_true_class = tf.argmax(y_true, axis=-1)  # Shape: [batch_size,]

    # # Flatten the arrays to shape: (8 * 9999,)
    # y_pred_class = y_pred_class.numpy().flatten()
    # y_true_class = y_true_class.numpy().flatten()

    # # Verify flattened shapes
    # print(f"y_pred_class shape: {y_pred_class.shape}")  # Expected: (79992,)
    # print(f"y_true_class shape: {y_true_class.shape}")  # Expected: (79992,)

    # # Calculate precision (average='macro' calculates precision for each class and averages)
    # precision = precision_score(y_true_class, y_pred_class, average='macro')  # or 'macro' or 'weighted'

    # # Calculate accuracy
    # accuracy = accuracy_score(y_true_class, y_pred_class)

    # # Calculate confusion matrix
    # conf_matrix = confusion_matrix(y_true_class, y_pred_class)

    # # Print results
    # print("Precision:", precision)
    # print("Accuracy:", accuracy)
    # print("Confusion Matrix:\n", conf_matrix)


    print(f"Utils metric: ")
    cal_metric(y_true, y_pred)






if __name__ == '__main__':
    main()
