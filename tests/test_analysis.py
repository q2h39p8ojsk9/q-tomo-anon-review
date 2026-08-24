from q_tomo.analysis import (
    logistic_train_test_auc,
    logistic_train_test_scores,
    roc_auc,
    separability_auc,
)


def test_auc():
    labels = [0, 0, 1, 1]
    assert roc_auc(labels, [0.0, 0.1, 0.9, 1.0]) == 1.0
    assert separability_auc(labels, [1.0, 0.9, 0.1, 0.0]) == 1.0


def test_logistic_train_test_auc_transfers_a_simple_boundary():
    train_features = [[-2.0], [-1.0], [1.0], [2.0]]
    train_labels = [0, 0, 1, 1]
    test_features = [[-3.0], [-0.5], [0.5], [3.0]]
    test_labels = [0, 0, 1, 1]
    assert logistic_train_test_auc(train_features, train_labels, test_features, test_labels) == 1.0


def test_logistic_train_test_scores_preserve_test_order():
    scores = logistic_train_test_scores(
        [[-2.0], [-1.0], [1.0], [2.0]],
        [0, 0, 1, 1],
        [[-3.0], [-0.5], [0.5], [3.0]],
    )
    assert len(scores) == 4
    assert scores == sorted(scores)
