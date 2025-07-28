import numpy as np
from utils import load_train_csv, load_valid_csv, load_public_test_csv, load_train_sparse
from knn import knn_impute_by_user
from item_response import irt, evaluate as irt_evaluate, sigmoid
from matrix_factorization import svd_reconstruct

def bootstrap_data(data):
    n = len(data["user_id"])
    # generate an array of length n, where each entry is 0 - n with replacement
    idx = np.random.choice(n, n, replace=True)
    return {
        "user_id": [data["user_id"][i] for i in idx],
        "question_id": [data["question_id"][i] for i in idx],
        "is_correct": [data["is_correct"][i] for i in idx],
    }

def main():
    # Load data
    train_data = load_train_csv("./data")
    base_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Create 3 bootstrap training sets
    bs1 = bootstrap_data(train_data)
    bs2 = bootstrap_data(train_data)
    bs3 = bootstrap_data(train_data)

    # since not all examples are guaranteed to be in our bootsrapped training sets, create new sparse matrices
    def build_matrix(sample):
        mat = np.full(base_matrix.shape, np.nan)
        for u, q, c in zip(sample["user_id"], sample["question_id"], sample["is_correct"]):
            mat[u, q] = c
        return mat

    mat1 = build_matrix(bs1)
    mat3 = build_matrix(bs3)

    # bagged model 1: KNN
    preds1_train = ...
    preds1_valid = ...
    # bagged model 2: IRT
    theta2, beta2, _, _, _ = irt(bs2, val_data, lr=0.01, iterations=50)
    preds2_train = np.array([sigmoid(theta2[u] - beta2[q]) for u, q in zip(test_data["user_id"], test_data["question_id"])])
    preds2_valid = np.array([sigmoid(theta2[u] - beta2[q]) for u, q in zip(val_data["user_id"], val_data["question_id"])])
    # bagged model 3: SVD
    preds3_train = ...
    preds3_valid = ...

    #ensemble by averaging probabilities
    avg_probs_train = (preds1_train + preds2_train + preds3_train) / 3
    avg_probs_valid = (preds1_valid + preds2_valid + preds3_valid) / 3

    final_preds_train = []
    final_preds_valid = []
    for prob_train, prob_valid in zip(avg_probs_train, avg_probs_valid):
        final_preds_train.append(prob_train >= 0.5)
        final_preds_valid.append(prob_valid >= 0.5)
        

    train_acc = np.mean(final_preds_train == np.array(test_data["is_correct"]))
    valid_acc = np.mean(final_preds_valid == np.array(val_data["is_correct"]))
    print(f"Ensemble Test Accuracy: {train_acc:.4f}")
    print(f"Ensemble Val Accuracy: {valid_acc:.4f}")


if __name__ == "__main__":
    main()
