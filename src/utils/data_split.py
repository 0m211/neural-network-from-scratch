import numpy as np


def train_validation_test_split(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    stratify=True,
):
    """
    Split a dataset into training, validation, and test sets.

    Parameters
    ----------
    X : array-like
        Features with shape (samples, features).

    y : array-like
        Labels with one label per sample.

    train_ratio, validation_ratio, test_ratio : float
        Fractions that must sum to 1.

    random_state : int or None
        Seed for reproducible shuffling.

    stratify : bool
        Preserve class proportions for classification datasets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 0 or y.ndim == 0:
        raise ValueError("X and y must have a sample dimension.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if X.shape[0] < 3:
        raise ValueError(
            "At least three samples are required."
        )

    ratios = np.asarray(
        [train_ratio, validation_ratio, test_ratio],
        dtype=float,
    )

    if not np.all(np.isfinite(ratios)):
        raise ValueError("Split ratios must be finite.")

    if np.any(ratios <= 0):
        raise ValueError("All split ratios must be positive.")

    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    if not isinstance(stratify, (bool, np.bool_)):
        raise ValueError("stratify must be True or False.")

    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    if stratify:
        if y.ndim == 2 and y.shape[1]==1:
            y=y.ravel()
        elif y.ndim !=1:    
            raise ValueError(
                "Stratification currently requires 1D labels."
                "column vector with shape (n_samples, 1)."
            )

        classes, inverse = np.unique(
            y,
            return_inverse=True,
        )

        class_indices = [
            np.flatnonzero(inverse == class_index)
            for class_index in range(len(classes))
        ]

        if any(len(indices) < 3 for indices in class_indices):
            raise ValueError(
                "Each class needs at least three samples "
                "for stratified splitting."
            )

        train_indices = []
        val_indices = []
        test_indices = []

        for indices in class_indices:
            indices = rng.permutation(indices)
            count = len(indices)

            n_train = int(round(count * train_ratio))
            n_val = int(round(count * validation_ratio))

            # Ensure each class appears in all three splits.
            n_train = max(1, n_train)
            n_val = max(1, n_val)

            if n_train + n_val >= count:
                n_train = count - 2
                n_val = 1

            n_test = count - n_train - n_val

            train_indices.extend(indices[:n_train])
            val_indices.extend(
                indices[n_train:n_train + n_val]
            )
            test_indices.extend(
                indices[n_train + n_val:]
            )

        train_indices = rng.permutation(train_indices)
        val_indices = rng.permutation(val_indices)
        test_indices = rng.permutation(test_indices)

    else:
        indices = rng.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * validation_ratio)

        if n_train < 1 or n_val < 1:
            raise ValueError(
                "The dataset is too small for these split ratios."
            )

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        if len(test_indices) == 0:
            raise ValueError(
                "The test split must contain at least one sample."
            )

    return (
        X[train_indices],
        X[val_indices],
        X[test_indices],
        y[train_indices],
        y[val_indices],
        y[test_indices],
    )