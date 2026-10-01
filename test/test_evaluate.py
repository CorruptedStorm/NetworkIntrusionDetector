from src.evaluate import evaluate


def test_evaluate_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_evaluate_catches_missed_attack():
    y_true = [0, 1, 1, 1]
    y_pred = [0, 0, 1, 1]  # missed one attack (a false negative)

    metrics = evaluate(y_true, y_pred)

    assert metrics["recall"] < 1.0
    assert metrics["precision"] == 1.0  # everything we flagged was correct
