from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-x))


def neg_log_likelihood(data, theta, beta, alpha):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """

    log_lklihood = 0.0
    epsilon = 1e-8
    reg_strength = 10

    # Iterate through each data point
    for k in range(len(data["user_id"])):
        i = data["user_id"][k]  # student index
        j = data["question_id"][k]  # question index
        c_ij = data["is_correct"][k]  # correctness (0 or 1)

        # calculate p_ij
        p_ij = sigmoid(alpha[j] * (theta[i] - beta[j]))
        p_ij = np.clip(p_ij, epsilon, 1 - epsilon)

        # add to log-likelihood
        log_lklihood += c_ij * np.log(p_ij) + (1 - c_ij) * np.log(1 - p_ij)

    # L2 regularization on alpha
    log_lklihood -= reg_strength * np.sum(alpha ** 2)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
    """Update theta, beta, alpha using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """
    reg_strength = 10

    # initialize gradients
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)
    alpha_grad = np.zeros_like(alpha)

    # compute gradients
    for k in range(len(data["user_id"])):
        i = data["user_id"][k]  # student index
        j = data["question_id"][k]  # question index
        c_ij = data["is_correct"][k]  # correctness (0 or 1)

        # calculate p_ij
        p_ij = sigmoid(alpha[j] * (theta[i] - beta[j]))

        diff = c_ij - p_ij

        # gradient of neg log-likelihood w.r.t. theta_i
        theta_grad[i] += alpha[j] * diff

        # gradient of neg log-likelihood w.r.t. beta_j
        beta_grad[j] += -alpha[j] * diff

        # gradient of neg log-likelihood w.r.t. alpha_j
        alpha_grad[j] += (theta[i] - beta[j]) * diff

    # update parameters
    theta = theta - lr * theta_grad
    beta = beta - lr * beta_grad

    # Add regularization gradient for alpha
    alpha_grad += 2 * reg_strength * alpha

    # Prevent alpha_j growing too large
    alpha -= lr * alpha_grad
    alpha = np.clip(alpha, 0.01, 2.0)

    return theta, beta, alpha


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, alpha, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst)
    """

    num_students = len(set(data["user_id"]))
    num_questions = len(set(data["question_id"]))

    # assume student abilities (theta) and question difficulty (beta) is normally distributed around 0
    theta = np.random.normal(0, 1, num_students)
    beta = np.random.normal(0, 1, num_questions)
    alpha = np.ones(num_questions)  # initialize informativeness

    val_acc_lst = []
    train_neg_lld_lst = []
    val_neg_lld_lst = []

    for i in range(iterations):
        # Calculate negative log-likelihoods BEFORE updating parameters
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)

        # Store values
        val_acc_lst.append(score)
        train_neg_lld_lst.append(train_neg_lld)
        val_neg_lld_lst.append(val_neg_lld)

        # print("NLLK: {} \t Score: {}".format(train_neg_lld, score))

        # Update parameters
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst


def evaluate(data, theta, beta, alpha):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = alpha[q]*(theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # hyperparameters
    lr = 0.00005
    iterations = 50

    print(f"Training IRT model with lr={lr}, iterations={iterations}")

    # track training and validation log-likelihoods for plotting
    train_neg_lld_list = []
    val_neg_lld_list = []

    np.random.seed(42)

    theta, beta, alpha, val_acc_lst, train_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iterations)

    final_val_acc = evaluate(val_data, theta, beta, alpha)
    final_test_acc = evaluate(test_data, theta, beta, alpha)

    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    print(f"Selected Hyperparameters: lr={lr}, iterations={iterations}")

    # plot Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_acc_lst) + 1), val_acc_lst, 'g-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
