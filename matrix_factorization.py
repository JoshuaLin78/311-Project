import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Compute prediction and error
    pred = np.dot(u[n], z[q])
    err = pred - c

    # Update user and item vectors via SGD
    u[n] -= lr * err * z[q]
    z[q] -= lr * err * u[n]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Perform stochastic updates
    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def print_accuracy_table(title, k_list, acc_list):
    """
    Print accuracy results in table format.
    """
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'k':<10} | {'Validation Accuracy':>20}")
    print("-" * 60)
    for k, acc in zip(k_list, acc_list):
        print(f"{str(k):<10} | {acc:>20.4f}")
    print("\n")


def als_with_best_k(train_data, val_data, k, lr, num_iterations, loss_interval):
    """Performs ALS algorithm with the optimal k found in Q3 part (d)
        """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    train_losses = []
    val_losses = []

    for i in range(1, num_iterations + 1):
        u, z = update_u_z(train_data, lr, u, z)

        if i % loss_interval == 0:
            train_loss = squared_error_loss(train_data, u, z)
            val_loss = squared_error_loss(val_data, u, z)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    final_matrix = np.dot(u, z.T)
    return final_matrix, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    # Dimension of Σ is at most 542, choose k << 542
    svd_k_vals = [10, 20, 50, 100, 120]
    svd_val_accuracies = []
    best_val_acc = 0
    best_k_svd = None
    best_svd_matrix = None

    for k in svd_k_vals:
        svd_matrix = svd_reconstruct(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, svd_matrix)
        svd_val_accuracies.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k_svd = k
            best_svd_matrix = svd_matrix

    test_acc = sparse_matrix_evaluate(test_data, best_svd_matrix)
    print_accuracy_table("SVD Reconstruction", svd_k_vals, svd_val_accuracies)
    print(f"[SVD] Best k={best_k_svd}, Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    # Dimension of latent factors
    als_k_vals = [10, 20, 50, 80, 100]
    best_val_acc = 0
    best_k_als = None

    lr = 0.1
    iterations = 350000
    loss_interval = 10000
    als_val_accuracies = []

    for k in als_k_vals:
        als_matrix = als(train_data, k=k, lr=lr, num_iteration=iterations)
        val_acc = sparse_matrix_evaluate(val_data, als_matrix)
        als_val_accuracies.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k_als = k

    print_accuracy_table("ALS with SGD", als_k_vals, als_val_accuracies)

    # Part (e) final validation accuracy and test accuracy & Plotting

    als_matrix, train_losses, val_losses = als_with_best_k(
        train_data, val_data, k=best_k_als, lr=lr, num_iterations=iterations, loss_interval=loss_interval
    )

    val_acc = sparse_matrix_evaluate(val_data, als_matrix)
    test_acc = sparse_matrix_evaluate(test_data, als_matrix)
    print(f"\n[ALS] k={best_k_als}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")

    x = np.arange(1, len(train_losses) + 1) * loss_interval
    plt.plot(x, train_losses, label="Train Loss")
    plt.plot(x, val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Squared Error Loss")
    plt.title(f"ALS with SGD (k={best_k_als},lr={lr})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
