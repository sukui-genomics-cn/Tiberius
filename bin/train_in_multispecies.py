#!/usr/bin/env python3
# ==============================================================
# Authors: Lars Gabriel, Felix Becker
#
# Train variety of LSTM or HMM models for gene prediction 
# using tfrecords.
# 
# Tensorflow 2.10.1, Transformers 4.31.0, 
# tensorflow_probability 0.18.0
# ==============================================================
import glob
import logging

import parse_args
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
# from bend_dataset import BendDataset, pytorch_to_tensorflow_dataset
from t2t_pkl_dataset import T2TTiberiusDataset, LABEL_NUM, pytorch_to_tensorflow_dataset

args = parse_args.parseCmd()
import sys, os, re, json, sys, csv

sys.path.insert(0, args.learnMSA)
if args.LRU:
    sys.path.insert(0, args.LRU)
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger
import models
from models import (weighted_categorical_crossentropy, custom_cce_f1_loss, BatchLearningRateScheduler,
                    add_hmm_only, add_hmm_layer, ValidationCallback,
                    BatchSave, EpochSave, lstm_model, add_constant_hmm,
                    make_weighted_cce_loss, )
logging.basicConfig(level=logging.INFO)
gpus = tf.config.list_physical_devices('GPU')

# for gpu in gpus:
# tf.config.experimental.set_memory_growth(gpu, True)
"""cluster_res = tf.distribute.cluster_resolver.SlurmClusterResolver(gpus_per_node=4,
    gpus_per_task=1)"""

strategy = tf.distribute.MirroredStrategy()
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

batch_save_numb = 1000


def train_hmm_model(generator, model_save_dir, config, val_data=None,
                    model_load=None, model_load_lstm=None, model_load_hmm=None, trainable=True, constant_hmm=False
                    ):
    """Trains a hybrid HMM-LSTM model with trainings example from a generator 
    and configuration, model and weights a re saved to model_save_dir.

    Parameters:
        - generator (DataGenerator): A data generator for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
        - model_load_lstm (optional): Path to a pre-trained LSTM model to be loaded, 
                                      it will be trained together with newly initialized HMM.
        - model_load_hmm (optional): Path to a pre-trained HMM model to be loaded,
        - trainable (bool): Flag indicating whether the LSTM model's layers are trainable. 
        - constant_hmm (bool): Flag to add a constant HMM layer to the model. 
    """

    model_save = model_save_dir + "/weights.{epoch:02d}"
    checkpoint = ModelCheckpoint(model_save, monitor='loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto')

    batch_callback = BatchSave(model_save_dir + "/weights_batch.{}", batch_save_numb)
    epoch_callback = EpochSave(model_save_dir)

    csv_logger = CSVLogger(f'{model_save_dir}/training.log',
                           append=True, separator=';')

    adam = Adam(learning_rate=config['lr'])
    with strategy.scope():
        if config['oracle']:
            inputs = tf.keras.layers.Input(shape=(None, 6 if config['softmasking'] else 5), name='main_input')
            oracle_inputs = tf.keras.layers.Input(shape=(None, config['output_size']), name='oracle_input')
            model = tf.keras.Model(inputs=[inputs, oracle_inputs], outputs=oracle_inputs)
        elif model_load_lstm:
            model = keras.models.load_model(model_load_lstm,
                                            custom_objects={
                                                'custom_cce_f1_loss': custom_cce_f1_loss(config['loss_f1_factor'],
                                                                                         config['batch_size']),
                                                'loss_': custom_cce_f1_loss(config['loss_f1_factor'],
                                                                            config['batch_size'])})
        else:
            relevant_keys = ['units', 'filter_size', 'kernel_size',
                             'numb_conv', 'numb_lstm', 'dropout_rate',
                             'pool_size', 'stride', 'lstm_mask', 'co',
                             'output_size', 'residual_conv', 'softmasking',
                             'clamsa_kernel', 'lru_layer', 'clamsa', 'clamsa_kernel']
            relevant_args = {key: config[key] for key in relevant_keys if key in config}
            model = lstm_model(**relevant_args)
        for layer in model.layers:
            layer.trainable = trainable
        if constant_hmm:
            model = add_constant_hmm(model, seq_len=config['sample_size'], batch_size=config['batch_size'],
                                     output_size=config['output_size'])
        else:
            if model_load_hmm:
                model_hmm = keras.models.load_model(model_load_hmm,
                                                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(
                                                        config['loss_f1_factor'], config['batch_size']),
                                                        'loss_': custom_cce_f1_loss(
                                                            config['loss_f1_factor'],
                                                            config['batch_size'])})
                gene_pred_layer = model_hmm.layers[-3]
            else:
                gene_pred_layer = None
            model = add_hmm_layer(
                model, gene_pred_layer,
                dense_size=config['hmm_dense'],
                pool_size=config['pool_size'],
                output_size=config['output_size'],
                num_hmm=config['num_hmm_layers'],
                l2_lambda=config['l2_lambda'],
                hmm_factor=config['hmm_factor'],
                batch_size=config['batch_size'],
                seq_len=config['w_size'],
                initial_variance=config['initial_variance'],
                temperature=config['temperature'],
                emit_embeddings=config['hmm_emit_embeddings'],
                share_intron_parameters=config['hmm_share_intron_parameters'],
                trainable_nucleotides_at_exons=config['hmm_nucleotides_at_exons'],
                trainable_emissions=config['hmm_trainable_emissions'],
                trainable_transitions=config['hmm_trainable_transitions'],
                trainable_starting_distribution=config['hmm_trainable_starting_distribution'],
                use_border_hints=False,
                include_lstm_in_output=config['multi_loss'],
                neutral_hmm=config['neutral_hmm'])
        if model_load:
            # load the weights onto the raw model instead of using model.load to allow hyperparameter changes
            # i.e. you can change hmm_factor and still use checkpoint saved with a different hmm_factor
            model.load_weights(model_load + "/variables/variables")
            if trainable:
                model.trainable = trainable
                for layer in model.layers:
                    layer.trainable = trainable
            print("Loaded model:", model_load)
        if config["loss_f1_factor"]:
            print("using f1 loss")
            loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        elif config['loss_weights']:
            loss = make_weighted_cce_loss(config['loss_weights'], config['batch_size'])
        else:
            loss = tf.keras.losses.CategoricalCrossentropy()
        if config['multi_loss']:
            # hmm_loss = loss
            # hmm_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            hmm_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"], from_logits=True)
            loss = [loss, hmm_loss]
            loss_weights = [1, config['hmm_loss_weight_mul']]
        else:
            loss_weights = None
            # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"], from_logits=True)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'], loss_weights=loss_weights)
        model.summary()
        b_lr_sched = BatchLearningRateScheduler(peak=config["lr"], warmup=config["warmup"],
                                                min_lr=config["min_lr"])
        model.save(model_save_dir + "/untrained.keras")
        model.fit(
            generator,
            epochs=config["num_epochs"],
            validation_data=val_data,
            steps_per_epoch=1000,
            validation_batch_size=config['batch_size'],
            callbacks=[epoch_callback, csv_logger]
        )
        return model


def read_species(file_name):
    """Reads a list of species from a given file, filtering out empty lines and comments.

    Parameters:
        - file_name (str): The path to the file containing species names.

        Returns:
        - list of str: A list of species names extracted from the file.
    """
    species = []
    with open(file_name, 'r') as f_h:
        species = f_h.read().strip().split('\n')
    return [s for s in species if s and s[0] != '#']


def train_clamsa(generator, model_save_dir, config, val_data=None, model_load=None, model_load_lstm=None):
    """Train simple CNN model that uses only CLAMSA as input.

    Parameters:
        - generator (DataGenerator): A data generator for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
        - model_load_lstm (optional): Path to a pre-trained LSTM model to be loaded.
    """

    model_save = model_save_dir + "/weights.{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(model_save, monitor='loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto')

    batch_callback = BatchSave(model_save_dir + "/weights_batch.{}.h5", batch_save_numb)
    epoch_callback = EpochSave(model_save_dir)

    adam = Adam(learning_rate=config['lr'])

    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    custom_objects = {}
    if config["loss_f1_factor"]:
        cce_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        custom_objects['custom_cce_f1_loss'] = cce_loss
        custom_objects['loss_'] = cce_loss
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
    if config["output_size"] == 1:
        cce_loss = tf.keras.losses.BinaryCrossentropy()

    csv_logger = CSVLogger(f'{model_save_dir}/training.log',
                           append=True, separator=';')
    with strategy.scope():
        if model_load:
            model = keras.models.load_model(model_load, custom_objects=custom_objects)
        else:
            if config['clamsa_with_lstm']:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                if model_load_lstm:
                    lstm_model = keras.models.load_model(model_load_lstm, custom_objects=custom_objects)
                    model = models.clamsa_lstm_model(lstm_model, **relevant_args)
                else:
                    relevant_keys = ['units', 'filter_size', 'kernel_size',
                                     'numb_conv', 'numb_lstm', 'dropout_rate',
                                     'pool_size', 'stride', 'lstm_mask', 'clamsa',
                                     'output_size', 'residual_conv', 'softmasking',
                                     'clamsa_kernel']
                    relevant_args = {key: config[key] for key in relevant_keys if key in config}
                    model = models.lstm_model(**relevant_args)
            elif config['use_hmm']:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size",
                                 "num_hmm", "hmm_factor", "share_intron_parameters"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                model = models.clamsa_hmm_model(only_hmm_output=not config["multi_loss"], **relevant_args)
            else:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                model = models.clamsa_only_model(**relevant_args)

        if config["loss_weights"]:
            model.compile(loss=cce_loss, optimizer=adam,
                          metrics=['accuracy'],  # sample_weight_mode='temporal',
                          loss_weights=config["loss_weights"]
                          )
        else:
            model.compile(loss=cce_loss, optimizer=adam,
                          metrics=['accuracy'])
        model.summary()

        model.fit(
            generator,
            epochs=500,
            steps_per_epoch=1000,
            callbacks=[epoch_callback, csv_logger])
        return model


def train_add_transformer2lstm(generator, model_save_dir, config, val_data=None, model_load=None):
    """Add transformer tokens as input and nuc transformer to the default LSTM model and train it.
    If model_load_lstm is included in config, it will be loaded and extend with the clamsa layers.

    Parameters:
        - generator (DataGenerator): A data generator for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
    """
    model_save = model_save_dir + "/weights.{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(model_save, monitor='loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto')

    batch_callback = BatchSave(model_save_dir + "/weights_batch.{}.h5", batch_save_numb)
    epoch_callback = EpochSave(model_save_dir)

    adam = Adam(learning_rate=config['lr'])

    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    custom_objects = {}
    if config["loss_f1_factor"]:
        cce_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        custom_objects['custom_cce_f1_loss'] = custom_cce_f1_loss(config["loss_f1_factor"])
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()

    if config["output_size"] == 1:
        cce_loss = tf.keras.losses.BinaryCrossentropy()

    csv_logger = CSVLogger(f'{model_save_dir}/training.log',
                           append=True, separator=';')
    with strategy.scope():
        if model_load:
            model = keras.models.load_model(model_load, custom_objects=custom_objects)
        else:
            model = add_transformer2lstm(config["model_load_lstm"],
                                         cnn_size=config["filter_size"], )

        if config["loss_weights"]:
            model.compile(loss=cce_loss, optimizer=adam,
                          metrics=['accuracy'],  # sample_weight_mode='temporal',
                          loss_weights=config["loss_weights"]
                          )
        else:
            model.compile(loss=cce_loss, optimizer=adam,
                          metrics=['accuracy'])
        model.summary()

        model.fit(generator, epochs=500,
                  steps_per_epoch=1000,
                  callbacks=[epoch_callback, csv_logger])
    return model


def train_lstm_model(generator, model_save_dir, config, val_data=None, model_load=None):
    """Trains the LSTM model using data provided by a generator, while saving the 
    training checkpoints and logging progress. The model can be trained from scratch or from a 
    pre-loaded state.

    Parameters:
        - generator (DataGenerator): A data generator for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
    """

    model_save = model_save_dir + "/weights.{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(model_save, monitor='loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto')

    batch_callback = BatchSave(model_save_dir + "/weights_batch.{}.h5", batch_save_numb)
    epoch_callback = EpochSave(model_save_dir)

    if config['sgd']:
        optimizer = SGD(learning_rate=config['lr'])
    else:
        optimizer = Adam(learning_rate=config['lr'])

    custom_objects = {}
    if config["loss_f1_factor"]:
        cce_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        custom_objects['loss_'] = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()

    if config["output_size"] == 1:
        cce_loss = tf.keras.losses.BinaryCrossentropy()

    csv_logger = CSVLogger(f'{model_save_dir}/training.log',
                           append=True, separator=';')
    with strategy.scope():

        relevant_keys = ['units', 'filter_size', 'kernel_size',
                         'numb_conv', 'numb_lstm', 'dropout_rate',
                         'pool_size', 'stride', 'lstm_mask', 'clamsa',
                         'output_size', 'residual_conv', 'softmasking',
                         'clamsa_kernel', 'lru_layer']
        relevant_args = {key: config[key] for key in relevant_keys if key in config}
        model = lstm_model(**relevant_args)
        if model_load:
            print(f"load model weight in: {model_load}")
            # ckpt = tf.train.Checkpoint(model=model)
            # model_load = "~"+model_load.split("~")[-1]
            # status = ckpt.restore(model_load + '/variables/variables').expect_partial()
            # print(f"status: {status}")
            model.load_weights(model_load + '/variables/variables')
            eval_val_data(model, val_data)
        if config["loss_weights"]:
            model.compile(
                loss=cce_loss,
                optimizer=optimizer,
                metrics=[
                    'accuracy',
                    # MatthewsCorrelationCoefficient(num_classes=9)
                ],
                sample_weight_mode='temporal',
                loss_weights=config["loss_weights"]
            )
        else:
            model.compile(
                loss=cce_loss, optimizer=optimizer,
                metrics=['accuracy'])
        model.summary()

        model.fit(
            generator,
            epochs=config["num_epochs"],
            validation_data=val_data,
            steps_per_epoch=1000,
            callbacks=[epoch_callback, csv_logger]
        )
    return model


def eval_val_data(model, val_data):
    labels = []
    y_predicts = model.predict(val_data)
    for val_i_data in val_data:
        feature, label = val_i_data
        labels.append(label)
    y_predicts = np.asarray(y_predicts)
    labels = np.concatenate(labels, axis=0)
    print(f"predict shape: {y_predicts.shape}; labels shape: {labels.shape}")
    cal_metric(labels, y_predicts)


def cal_metric(y_true, y_pred, ignore_index=9):
    """Calculates the Matthews correlation coefficient for binary classification.

    Parameters:
        - y_true (array): True binary labels.
        - y_pred (array): Predicted binary labels.

    Returns:
        - float: Matthews
    """
    binary_classification = y_pred.shape[-1] == 2
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    label = y_true.reshape(-1)
    predict = y_pred.reshape(-1)

    mask = label != ignore_index
    label = label[mask]
    predict = predict[mask]
    if binary_classification:
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict),
            'precision_score': precision_score(label, predict),
            # 'roc_auc_score': roc_auc_score(label, predict),
        }
    else:
        # fix sum up to 1.0 over classes
        # pred_prob = np.exp(pred_prob) / np.sum(np.exp(pred_prob), axis=1, keepdims=True)
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict, average='macro'),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict, average='macro'),
            'precision_score': precision_score(label, predict, average='macro'),
        }
    print(f"label: {label}, max label: {label.max()}")
    confu_matrix = confusion_matrix(label, predict)
    if confu_matrix.shape[-1] > 20:
        confu_matrix = confu_matrix[:20, :20]
        print("Confusion matrix is too large, only show the first 9x9 part")
    result["confu_matrix"] = str(confu_matrix)
    print(f"eval metric: \n{json.dumps(result)}")


def load_t2t_data(dest_path=None, batch_size=4, max_length=9999, dataset_name="gene_finding", split="train", repeat=True):
    # load bend data
    t2t_dataset = T2TTiberiusDataset(
        dest_path=dest_path,
        split=split,
        dataset_name=dataset_name,
        max_length=max_length,
        read_strand=False,
    )
    bend_dataset = pytorch_to_tensorflow_dataset(t2t_dataset)
    bend_dataset = bend_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    bend_dataset = bend_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if repeat:
        bend_dataset = bend_dataset.repeat()
    bend_dataset = bend_dataset.prefetch(tf.data.AUTOTUNE)
    return bend_dataset


def main():
    # currently only w_size=9999 is used
    w_size = 9999
    if w_size == 99999:
        batch_size = 96
        batch_save_numb = 100000
    elif w_size == 50004:
        batch_size = 28
        batch_save_numb = 100000
    elif w_size == 9999:
        batch_size = 16
        batch_save_numb = 1000
    elif w_size == 29997:
        batch_size = 120 * 4
        batch_save_numb = 1000
    if args.cfg:
        with open(args.cfg, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = {
            "num_epochs": 100,
            'use_hmm': args.hmm,
            "loss_weights": False,
            # [1,1,1e3,1e3,1e3],
            # [ 0.24064536,  1.23309401, 89.06682408, 89.68105166, 89.5963385 ],<- computed from class frequencies in train data
            # "loss_weights": [6.37, 1485.62, 1.52, 1485.62, 6.11, 1438.04, 1.52, 1438.04, 1.0, 1.0],
            # [1., 1., 1., 1., 1.],
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
            "batch_size": batch_size,
            "w_size": w_size,  # sequence length
            "filter": False,  # if True, filters all training examples out that are IR-only
            "trainable_lstm": True,  # if False, LSTM is not trainable -> only HMM is trained
            # output_size determines the shape of all outputs and the labels
            # hmm code will try to adapt if output size of loaded lstm is different to this number
            'output_size': LABEL_NUM,  # default 15
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
            'clamsa': args.clamsa,  # adds clamsa track to input
            'clamsa_kernel_size': 7,  # kernel size of CNN layer used after clamsa Input
            'clamsa_emb_size': 32,  # embedding size used in the clamsa model
            'clamsa_with_lstm': True,  # combines LSTM and clamsa model
            'loss_f1_factor': 2.0,
            'sgd': False,
            'oracle': False,  # if True, the correct labels will be used as input data. Can be used to debug the HMM.
            "lru_layer": False
        }

    config_dict['model_load'] = os.path.abspath(args.load) if args.load else None
    config_dict['model_save_dir'] = os.path.abspath(args.out)
    config_dict['model_load_lstm'] = os.path.abspath(args.load_lstm) if args.load_lstm else None
    config_dict['model_load_hmm'] = os.path.abspath(args.load_hmm) if args.load_hmm else None
    # config_dict['nuc_trans'] = args.nuc_trans

    # write config file
    with open(f'{config_dict["model_save_dir"]}/config.json', 'w+') as f:
        json.dump(config_dict, f)

    # init tfrecord generator
    generator = load_t2t_data(
        dest_path=args.data,
        dataset_name=args.dataset_name,
        split="train",
        batch_size=batch_size,
        max_length=9999
    )

    val_data = None
    if args.val_data and os.path.exists(args.val_data):
        val_data = load_t2t_data(
            dest_path=args.val_data,
            dataset_name=args.dataset_name,
            batch_size=batch_size,
            split="val",
            repeat=False
        )

    if args.hmm:
        logging.info("start train hmm model")
        model = train_hmm_model(
            generator=generator, val_data=val_data,
            model_save_dir=config_dict["model_save_dir"], config=config_dict,
            model_load_lstm=config_dict["model_load_lstm"],
            model_load_hmm=config_dict["model_load_hmm"],
            model_load=config_dict["model_load"],
            trainable=config_dict["trainable_lstm"],
            constant_hmm=config_dict["constant_hmm"]
        )
    elif args.clamsa:
        logging.info("start train_clamsa")
        model = train_clamsa(
            generator=generator,
            model_save_dir=config_dict["model_save_dir"],
            config=config_dict,
            model_load=config_dict["model_load"],
            model_load_lstm=config_dict["model_load_lstm"])
    elif args.nuc_trans:
        logging.info("start train_add_transformer2lstm")
        model = train_add_transformer2lstm(
            generator=generator,
            model_save_dir=config_dict["model_save_dir"],
            config=config_dict,
            model_load=config_dict["model_load"])
    else:
        logging.info("start train_lstm_model")
        model = train_lstm_model(
            generator=generator,
            val_data=val_data,
            model_save_dir=config_dict["model_save_dir"],
            config=config_dict,
            model_load=config_dict["model_load"]
        )

    if val_data is not None:
        eval_val_data(model, val_data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
