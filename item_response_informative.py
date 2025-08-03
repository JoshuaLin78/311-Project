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
    reg_strength = 0.5

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

        # L2 regularization on theta, beta, and alpha
    log_lklihood -= reg_strength * (np.sum(theta ** 2) + np.sum(beta ** 2) + np.sum(alpha ** 2))
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
    reg_strength = 0.5

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
        theta_grad[i] += -alpha[j] * diff

        # gradient of neg log-likelihood w.r.t. beta_j
        beta_grad[j] += alpha[j] * diff

        # gradient of neg log-likelihood w.r.t. alpha_j
        alpha_grad[j] += -(theta[i] - beta[j]) * diff

    # L2 regularization gradient
    theta_grad += 2 * reg_strength * theta
    beta_grad += 2 * reg_strength * beta
    alpha_grad += 2 * reg_strength * alpha

    # update parameters
    theta = theta - lr * theta_grad
    beta = beta - lr * beta_grad

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

    # assume question informativeness is uniformly distributed
    alpha = np.ones(num_questions)

    val_acc_lst = []
    train_neg_lld_lst = []
    val_neg_lld_lst = []
    train_acc_lst = []

    for i in range(iterations):
        # Calculate negative log-likelihoods BEFORE updating parameters
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha)
        train_score = evaluate(data=data, theta=theta, beta=beta, alpha=alpha)
        val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)


        # Store values
        train_acc_lst.append(train_score)
        val_acc_lst.append(val_score)
        train_neg_lld_lst.append(train_neg_lld)
        val_neg_lld_lst.append(val_neg_lld)

        # Update parameters
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    return theta, beta, alpha, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst


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
    lr, iterations = 0.04, 60

    np.random.seed(42)

    theta, beta, alpha, train_acc_lst, val_acc_lst, train_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iterations)

    final_train_acc = evaluate(data=train_data, theta=theta, beta=beta, alpha=alpha)
    final_val_acc = evaluate(val_data, theta, beta, alpha)
    final_test_acc = evaluate(test_data, theta, beta, alpha)

    print(f"\nFinal Results:")
    print(f"Train Accuracy: {final_train_acc:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    print(f"Selected Hyperparameters: lr={lr}, iterations={iterations}")

    # plot Training and Validation Accuracy

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_acc_lst) + 1), val_acc_lst, 'r-', label=f'Val Acc', linewidth=2)
    plt.plot(range(1, len(train_acc_lst) + 1), train_acc_lst, 'b--', label=f'Train Acc', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
