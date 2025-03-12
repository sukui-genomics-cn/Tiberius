import sys, json, os, re, sys, csv, argparse, requests, time, logging, warnings, pickle
sys.path.insert(0, './learnMSA')
script_dir = os.path.dirname(os.path.realpath(__file__))
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

import wandb
gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tiberius import parseCmd
from utils import cal_metric
from models import lstm_model

from sklearn.metrics import precision_score, accuracy_score, confusion_matrix

from models import custom_cce_f1_loss

import tensorflow as tf

from train import train_hmm_model, train_lstm_model

def verify_gpu():
    """
    Verifies GPU availability, sets memory growth, and initializes MirroredStrategy for multi-GPU training.

    Returns:
        strategy (tf.distribute.MirroredStrategy): The initialized MirroredStrategy object.
    """
    # List all physical GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs detected. Using CPU.")
    else:
        print(f"GPUs Available ({len(gpus)}):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        # Prevent TensorFlow from allocating all GPU memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Set memory growth on GPUs.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Error setting memory growth:", e)
            print("Proceeding without setting memory growth.")

    # Initialize MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices:', strategy.num_replicas_in_sync)
    
    return strategy


def configure_gpu(gpu_index=2, use_mirrored_strategy=True):
    """
    Configures TensorFlow to use a specific GPU by its index and optionally
    sets up a MirroredStrategy for multi-GPU training.

    Args:
        gpu_index (int): The index of the GPU to use (0-based).
        use_mirrored_strategy (bool): Whether to use MirroredStrategy for multi-GPU.
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
        #tf.config.set_visible_devices(selected_gpu, 'GPU')
        print(f"Selected GPU: {selected_gpu.name}")

        # Enable memory growth for the selected GPU
        #tf.config.experimental.set_memory_growth(selected_gpu, True)
        print(f"Memory growth enabled for GPU: {selected_gpu.name}")

        # Verify that TensorFlow sees only the selected GPU
        visible_gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow sees {len(visible_gpus)} GPU(s): {[gpu.name for gpu in visible_gpus]}")

        # If MirroredStrategy is enabled
        if use_mirrored_strategy:
            # Create a MirroredStrategy
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices.")
            return strategy  # Return the strategy to use in model training
        else:
            print("MirroredStrategy is not enabled.")
            return None

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(f"RuntimeError: {e}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")


def setup_logger(name='tensorflow_training', log_file='training.log', level=logging.INFO):
    """
    Sets up a logger with console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): File path for the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers to the logger if it's already configured
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(log_file)  # File handler
        c_handler.setLevel(level)
        f_handler.setLevel(level)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger


def initialize_wandb(project_name="Tiberius_pretraining", config_dict=None, run_name=None):
    """
    Initializes a wandb run with the specified configuration.

    Args:
        project_name (str): The name of the wandb project.
        config_dict (dict): A dictionary containing hyperparameters and configurations.
        run_name (str, optional): A name for the wandb run.

    Returns:
        wandb.config: The configuration object for the run.
    """
    # Initialize wandb run
    wandb.init(
        project=project_name,
        config=config_dict,
        name=run_name,
        sync_tensorboard=True,  # Automatically syncs TensorBoard logs
        save_code=True
    )
    # Return the configuration object
    return wandb.config


def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(model, train_dataset, val_dataset, config, logger, checkpoint_path, epochs=None, steps_per_epoch=None, validation_steps=None):
    """
    Trains a TensorFlow model with integrated logging, wandb tracking, model checkpointing, and progress bars.

    Args:
        model (tf.keras.Model): The compiled TensorFlow model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        config (dict): Configuration dictionary containing hyperparameters (e.g., 'epochs', 'batch_size').
        logger (logging.Logger): Configured logger for logging training progress.
        checkpoint_path (str): File path pattern for saving model checkpoints (e.g., 'checkpoints/model_epoch_{epoch}.h5').
        epochs (int, optional): Number of epochs to train. If None, taken from config or defaults to 10.
        steps_per_epoch (int, optional): Number of steps (batches) per epoch. If None, inferred from dataset.
        validation_steps (int, optional): Number of validation steps. If None, inferred from dataset.

    Returns:
        history (tf.keras.callbacks.History): History object containing training details.
    """

    class TQDMProgressBar(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.total_epochs = self.params.get('epochs', 0)
            self.prog_bar = tqdm(total=self.total_epochs, desc='Training', unit='epoch')

        def on_epoch_begin(self, epoch, logs=None):
            steps = self.params.get('steps', None)
            if steps:
                desc = f'Epoch {epoch+1}/{self.total_epochs}'
                self.batch_bar = tqdm(total=steps, desc=desc, unit='batch', leave=False)
            else:
                self.batch_bar = tqdm(desc=f'Epoch {epoch+1}/{self.total_epochs}', unit='batch', leave=False)

        def on_batch_end(self, batch, logs=None):
            self.batch_bar.update(1)
            if logs:
                # Optionally, display batch metrics
                self.batch_bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()}, refresh=False)

        def on_epoch_end(self, epoch, logs=None):
            self.batch_bar.close()
            self.prog_bar.update(1)
            if logs:
                # Optionally, display epoch metrics
                self.prog_bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()}, refresh=False)

        def on_train_end(self, logs=None):
            self.prog_bar.close()

    class CustomLoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                # Log to Python logger
                log_message = f"Epoch {epoch + 1}: " + ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
                self.logger.info(log_message)

                # Log to wandb
                wandb.log(logs, step=epoch + 1)

    # Initialize callbacks
    tqdm_callback = TQDMProgressBar()
    custom_logging_callback = CustomLoggingCallback()
    custom_logging_callback.logger = logger  # Assign the logger to the callback

    # ModelCheckpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',           # Metric to monitor
        save_best_only=True,          # Save only when the monitored metric improves
        save_weights_only=False,      # Save the entire model
        verbose=1                     # Verbosity mode
    )

    wandb_callback = WandbCallback()

    callbacks = [tqdm_callback, custom_logging_callback, checkpoint_callback, wandb_callback]

    # Determine number of epochs
    if epochs is None:
        epochs = config.get('epochs', 10)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history




# class CustomLoggingCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if logs is not None:
#             # Log metrics to wandb
#             wandb.log(logs, step=epoch)

#             # If you have custom metrics to log
#             custom_metric = compute_custom_metric()
#             wandb.log({'custom_metric': custom_metric}, step=epoch)

#             # Also log using your logger if desired
#             log_message = f"Epoch {epoch + 1}: " + ", ".join([f"{key}={value:.4f}" for key, value in logs.items()])
#             logger.info(log_message)


class TrainDataset:
    def __init__(self, pkl_paths="/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic/homo_sapiens_tiberius_pkls_from_tfrecord_10K",
                 split='train', batch_size=8, buffer_size=1000):
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
                tf.TensorSpec(shape=(9999,15), dtype=tf.int32)   # For annotation 
            )
        )
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.pkl_paths)

def main():

    config_dict = {
            "num_epochs": 200,
            'use_hmm': False,
            "loss_weights": False,
            # [1,1,1e3,1e3,1e3],
            # [ 0.24064536,  1.23309401, 89.06682408, 89.68105166, 89.5963385 ],<- computed from class frequencies in train data
            # "loss_weights": [1.0, 1.0, 100.0, 100.0, 100.0],#[1., 1., 1., 1., 1.],
            # [1.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0],#[0.33, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0],#
            # binary weights: [0.5033910039153116, 74.22447990141231]
            "stride": 0,  # if > 0 reduces size of sequence CNN stride
            "units": 372,  # 192, #512, # output size of LSTMS
            "filter_size": 128,  # 192,#64, # filter size of CNNs
            "numb_lstm": 2,
            "numb_conv": 3,
            "dropout_rate": 0.0,
            "lstm_mask": False,
            # pool size is the reduction factor for the sequence before the LSTM,
            # number of adjacent nucleotides that are one position for the LSTM
            "pool_size": 9,
            "lr": 1e-4,
            "warmup": 1,  # currently not used
            "min_lr": 1e-4,  # currently not used
            "batch_size": 16,
            "w_size": 9999,  # sequence length
            "filter": False,  # if True, filters all training examples out that are IR-only
            "trainable_lstm": True,  # if False, LSTM is not trainable -> only HMM is trained
            # output_size determines the shape of all outputs and the labels
            # hmm code will try to adapt if output size of loaded lstm is different to this number
            'output_size': 15,  # default 15
            'multi_loss': False,  # if both this and use_hmm are True, uses a additional LSTM loss during training
            'l2_lambda': 0.,
            'temperature': 32 * 3,
            'initial_variance': 0.1,
            'hmm_factor': 99,
            # parallelization factor of HMM, use the factor of w_size that is closest to sqrt(w_size) (271 works well for w_size=99999, 99 for w_size=9999)
            'seq_weights': False,  # Adds 3d weights with higher weights around positions of exon borders
            'softmasking': True,  # Adds softmasking track to input
            'residual_conv': True,  # Adds result of CNNs to the input to the last dense layer of the LSTM model
            'hmm_loss_weight_mul': 0.1,
            'hmm_emit_embeddings': False,
            "hmm_dense": 32,  # size of embedding for HMM input
            'hmm_share_intron_parameters': False,
            'hmm_nucleotides_at_exons': False,
            'hmm_trainable_transitions': False,
            'hmm_trainable_starting_distribution': False,
            'hmm_trainable_emissions': False,  # does not affect embedding emissions, use hmm_emit_embeddings for that
            "neutral_hmm": False,  # initializes an HMM without human expert bias, currently not implemented
            'constant_hmm': False,  # maybe not working anymore
            'num_hmm_layers': 1,  # numb. of parallel HMMs, currently only 1 is used
            'clamsa': False,  # adds clamsa track to input
            'clamsa_kernel_size': 7,  # kernel size of CNN layer used after clamsa Input
            'clamsa_emb_size': 32,  # embedding size used in the clamsa model
            'clamsa_with_lstm': True,  # combines LSTM and clamsa model
            'loss_f1_factor': 2.0,
            'sgd': False,
            'oracle': False,  # if True, the correct labels will be used as input data. Can be used to debug the HMM.
            "lru_layer": False
        }
    
    set_seed()
    #configure_gpu()
    # logger = setup_logger()
    # logger.info("Logger initialized...")
    train_dataset_obj = TrainDataset(batch_size=64)
    train_dataset = train_dataset_obj.create_dataset()

    val_dataset_obj = TrainDataset(batch_size=64, split='val', repeat=False)
    val_dataset = val_dataset_obj.create_dataset()

    model_save_dir = '/home/share/huadjyin/home/nemanjaudovic/projects/Tiberius/model_pretrain'

    train_lstm_model(generator= train_dataset, model_save_dir=model_save_dir, config=config_dict, val_data = val_dataset)
    
def test_dataset():
    train_dataset_obj = TrainDataset()
    train_dataset = train_dataset_obj.create_dataset()
    print(train_dataset_obj.pkl_paths)


if __name__ == '__main__':
    main()

