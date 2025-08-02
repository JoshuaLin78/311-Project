import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc



def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Transpose so rows = questions, columns = students
    matrix_T = matrix.T

    nbrs = KNNImputer(n_neighbors=k)
    mat_T = nbrs.fit_transform(matrix_T)

    # Transpose back so rows = students, columns = questions
    mat = mat_T.T

    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Different k values
    k_list = [1, 6, 11, 16, 21, 26]
    val_acc_user = []

    # === User-based KNN ===
    print("Running User-based KNN...")
    for k in k_list:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        val_acc_user.append(acc)
        print("Validation Accuracy: {}".format(acc))

    # Find best k for user-based
    best_k_user = k_list[np.argmax(val_acc_user)]
    print(f"Best k for user-based = {best_k_user}")

    # Test accuracy with best k
    final_matrix_user = KNNImputer(n_neighbors=best_k_user).fit_transform(sparse_matrix)
    test_acc_user = sparse_matrix_evaluate(test_data, final_matrix_user)
    print(f"Test Accuracy (User-based) = {test_acc_user}")

    # part c: item-based KNN
    val_acc_item = []
    print("\nRunning Item-based KNN...")
    for k in k_list:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc_item.append(acc)
        print(f"Validation Accuracy (Item-based, k={k}): {acc}")

    best_k_item = k_list[np.argmax(val_acc_item)]
    print(f"Best k for item-based = {best_k_item}")
    final_matrix_item = KNNImputer(n_neighbors=best_k_item).fit_transform(sparse_matrix.T).T
    test_acc_item = sparse_matrix_evaluate(test_data, final_matrix_item)
    print(f"Test Accuracy (Item-based) = {test_acc_item}")

    # Plot both user- and item-based on same graph
    plt.plot(k_list, val_acc_user, marker='o', label="User-based KNN")
    plt.plot(k_list, val_acc_item, marker='s', label="Item-based KNN")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs k")
    plt.grid(True)
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
