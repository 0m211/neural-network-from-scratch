import numpy as np


def _validate_binary_inputs(y_true, y_pred):
    """
    Validate binary classification labels and predictions.

    y_true: Actual labels containing only 0 and 1.
    y_pred: Predicted labels containing only 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Inputs cannot be empty."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains non-finite values."
        )

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError(
            "y_pred must contain only 0 and 1."
        )

    return (
        y_true.astype(np.int64).ravel(),
        y_pred.astype(np.int64).ravel(),
    )


def confusion_matrix(y_true, y_pred):
    """
    Return a 2x2 binary confusion matrix.

    Rows represent actual classes.
    Columns represent predicted classes.

    [[TN, FP],
     [FN, TP]]
    """
    y_true, y_pred = _validate_binary_inputs(
        y_true,
        y_pred,
    )

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )


def accuracy_score(y_true, y_pred):
    """Calculate the proportion of correct predictions."""
    y_true, y_pred = _validate_binary_inputs(
        y_true,
        y_pred,
    )

    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred):
    """
    Calculate precision.

    Precision = TP / (TP + FP)

    Returns 0 when there are no positive predictions.
    """
    matrix = confusion_matrix(y_true, y_pred)

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return float(tp / denominator)


def recall_score(y_true, y_pred):
    """
    Calculate recall.

    Recall = TP / (TP + FN)

    Returns 0 when there are no actual positive examples.
    """
    matrix = confusion_matrix(y_true, y_pred)

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return float(tp / denominator)


def f1_score(y_true, y_pred):
    """
    Calculate the harmonic mean of precision and recall.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return float(
        2 * precision * recall / denominator
    )


def classification_report(y_true, y_pred):
    """Return common binary classification metrics."""
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": matrix,
    }