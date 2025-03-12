from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np
import json
import pandas as pd

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
    confu_matrix = confusion_matrix(label, predict, labels=list(range(15)))
    print(confu_matrix)
    cm_df = pd.DataFrame(confu_matrix, columns=[f'Pred_{i}' for i in range(confu_matrix.shape[1])],
                     index=[f'True_{i}' for i in range(confu_matrix.shape[0])])
    print(cm_df)
    cm_df.to_csv('confusion_matrix.csv', index=True)
    print("Saved confusion_matrix_107.csv")
    if confu_matrix.shape[-1] > 20:
        confu_matrix = confu_matrix[:20, :20]
        print("Confusion matrix is too large, only show the first 9x9 part")
    #result["confu_matrix"] = str(confu_matrix)
    print(f"eval metric: \n{json.dumps(result)}")



def tiberius_reduce_labels(y_batch: np.ndarray, output_size=7):
    """
    Set output size for the model with tiberius labels, only support y_batch dim are 7 and 15
    :param output_size:
    :return:
    """
    # reformat labels so that they fit the output size
    if y_batch.shape[-1] == 7:
        if output_size == 5:
            # reduce intron labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2:] = y_batch[..., 4:]
        elif output_size == 3:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 15:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., :y_batch.shape[-1]] = y_batch
        else:
            y_new = y_batch.astype(np.float32)
        y_batch = y_new
    elif y_batch.shape[-1] == 15:
        if output_size == 3:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 5:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 3] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 4] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        elif output_size == 7:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., :4] = y_batch[..., :4]
            y_new[..., 4] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 5] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 6] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        elif output_size == 15:
            y_new = y_batch.astype(np.float32)
        elif output_size == 2:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = np.sum(y_batch[..., :4], axis=-1)
            y_new[..., 1] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 4:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = np.sum(y_batch[..., :4], axis=-1)
            y_new[..., 1] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 3] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        else:
            raise ValueError(f"output size {output_size} not supported, only support 7, 15 classes")
        y_batch = y_new
    return y_batch
