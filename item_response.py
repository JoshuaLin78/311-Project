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
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    
    # Iterate through each data point
    for k in range(len(data["user_id"])):
        i = data["user_id"][k]  # student index
        j = data["question_id"][k]  # question index
        c_ij = data["is_correct"][k]  # correctness (0 or 1)
        
        # calculate p_ij
        p_ij = sigmoid(theta[i] - beta[j])
        
        # add to log-likelihood
        log_lklihood += c_ij * np.log(p_ij) + (1 - c_ij) * np.log(1 - p_ij)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

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
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # initialize gradients
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)
    
    # compute gradients
    for k in range(len(data["user_id"])):
        i = data["user_id"][k]  # student index
        j = data["question_id"][k]  # question index
        c_ij = data["is_correct"][k]  # correctness (0 or 1)
        
        # calculate p_ij
        p_ij = sigmoid(theta[i] - beta[j])
        
        # gradient of neg log-likelihood w.r.t. theta_i
        theta_grad[i] += -(c_ij - p_ij)
        
        # graidnet of neg log-likelihood w.r.t. beta_j
        beta_grad[j] += (c_ij - p_ij)
    
    # update parameters
    theta = theta - lr * theta_grad
    beta = beta - lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst)
    """
    # TODO: Initialize theta and beta.
    num_students = len(set(data["user_id"]))
    num_questions = len(set(data["question_id"]))
    
    # assume student abilities (theta) and question difficulty (beta) is normally distributed around 0
    theta = np.random.normal(0, 1, num_students)
    beta = np.random.normal(0, 1, num_questions)

    val_acc_lst = []
    train_neg_lld_lst = []
    val_neg_lld_lst = []

    for i in range(iterations):
        # Calculate negative log-likelihoods BEFORE updating parameters
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        
        # Store values
        val_acc_lst.append(score)
        train_neg_lld_lst.append(train_neg_lld)
        val_neg_lld_lst.append(val_neg_lld)
        
        # print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        
        # Update parameters
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # hyperparameters
    lr = 0.01
    iterations = 50
    
    print(f"Training IRT model with lr={lr}, iterations={iterations}")
    
    # track training and validation log-likelihoods for plotting
    train_neg_lld_list = []
    val_neg_lld_list = []
    
    np.random.seed(42)
    
    theta, beta, val_acc_lst, train_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iterations)
    
    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc = evaluate(test_data, theta, beta)
    
    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    print(f"Selected Hyperparameters: lr={lr}, iterations={iterations}")
    
    # print training curve data
    print(f"\nTraining Curve Data:")
    print("Iter\tTrain NLLK\tVal NLLK\tVal Acc")
    print("-" * 40)
    for i in range(len(train_neg_lld_list)):
        print(f"{i+1:4d}\t{train_neg_lld_list[i]:.4f}\t{val_neg_lld_list[i]:.4f}\t{val_acc_lst[i]:.4f}")
            
    # plot 1: Negative Log-Likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_neg_lld_list) + 1), train_neg_lld_list, 'b-', label='Training NLLK', linewidth=2)
    plt.plot(range(1, len(val_neg_lld_list) + 1), val_neg_lld_list, 'r-', label='Validation NLLK', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training and Validation Negative Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
        
    # plot 2: Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_acc_lst) + 1), val_acc_lst, 'g-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    
    # sort questions by beta (difficulty)
    question_difficulties = [(j, beta[j]) for j in range(len(beta))]
    question_difficulties.sort(key=lambda x: x[1])
    
    # select questions of different difficulties
    easy_idx = len(question_difficulties) // 4         # first quarter
    medium_idx = len(question_difficulties) // 2       # near the middle
    hard_idx = 3 * len(question_difficulties) // 4     # third quarter
    
    j1 = question_difficulties[easy_idx][0]    # easy question
    j2 = question_difficulties[medium_idx][0]  # medium question
    j3 = question_difficulties[hard_idx][0]    # hard question
    
    selected_questions = [j1, j2, j3]
    question_labels = ['Easy', 'Medium', 'Hard']
    
    print(f"\nSelected questions with different difficulty levels:")
    print(f"j1 (Easy):   Question {j1:3d}, β = {beta[j1]:7.4f}")
    print(f"j2 (Medium): Question {j2:3d}, β = {beta[j2]:7.4f}")
    print(f"j3 (Hard):   Question {j3:3d}, β = {beta[j3]:7.4f}")
    
    # create evenly spaced range of student abilities (theta) for plotting
    theta_range = np.linspace(-3, 3, 100)
    
    # plot the three probability curves
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'red']
    
    for i, (j, label, color) in enumerate(zip(selected_questions, question_labels, colors)):
        probabilities = sigmoid(theta_range - beta[j])
        plt.plot(theta_range, probabilities, color=color, linewidth=3, 
                label=f'{label}: j{i+1} (β = {beta[j]:.3f})')
    
    # add reference lines for better interpretation
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1, 
                label='50% probability')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.xlabel('Student Ability (θ)', fontsize=14)
    plt.ylabel('Probability of Correct Response', fontsize=14)
    plt.title('P(correct) vs Student Ability', fontsize=16)
    plt.legend(fontsize=12, loc='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(-3, 3)
    
    plt.tight_layout()
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
