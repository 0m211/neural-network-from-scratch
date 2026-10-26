import numpy as np

from src.nn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def main():
    print("=" * 60)
    print("BINARY CLASSIFICATION METRICS TEST")
    print("=" * 60)

    # Example phishing detection results.
    y_true = np.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    )

    y_pred = np.array(
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    )

    # Expected confusion matrix:
    #
    # [[TN, FP],
    #  [FN, TP]]
    #
    # [[3, 2],
    #  [2, 3]]

    matrix = confusion_matrix(y_true, y_pred)

    expected_matrix = np.array(
        [
            [3, 2],
            [2, 3],
        ]
    )

    assert np.array_equal(
        matrix,
        expected_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(accuracy, 0.6)
    assert np.isclose(precision, 0.6)
    assert np.isclose(recall, 0.6)
    assert np.isclose(f1, 0.6)

    print("\nConfusion matrix:")
    print(matrix)

    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
    )

    assert np.isclose(
        report["accuracy"],
        accuracy,
    )

    # Test invalid labels.
    try:
        accuracy_score(
            np.array([0, 2]),
            np.array([0, 1]),
        )
    except ValueError:
        print("\nPASS: Invalid labels are rejected.")
    else:
        raise AssertionError(
            "Invalid labels were not rejected."
        )

    # Test mismatched shapes.
    try:
        accuracy_score(
            np.array([0, 1]),
            np.array([0]),
        )
    except ValueError:
        print("PASS: Mismatched shapes are rejected.")
    else:
        raise AssertionError(
            "Mismatched shapes were not rejected."
        )

    print("\nALL METRICS TESTS PASSED!")


if __name__ == "__main__":
    main()