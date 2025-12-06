import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==============================================================================
# WEEK 8 HOMEWORK SOLUTIONS (Sigmoid, Logistic Regression, Metrics)
# ==============================================================================

print("\n" + "="*40)
print("WEEK 8 SOLUTIONS")
print("="*40)

# --- Exercise 1: Sigmoid Function ---
print("\n--- Exercise 1: Sigmoid Function ---")

def sigmoid(x):
    """Compute the sigmoid function for input x."""
    return 1 / (1 + np.exp(-x))

# Generate x values and compute sigmoid
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

# Plot Sigmoid
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Sigmoid Function')
plt.title('Week 8: Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
plt.legend()
plt.show()

# Test values
test_values = [-5, -2, 0, 2, 5]
print("Sigmoid function test:")
for val in test_values:
    print(f"Sigmoid({val}) = {sigmoid(val):.4f}")


# --- Exercise 2: Logistic Regression Probability ---
print("\n--- Exercise 2: Logistic Regression Probability ---")

def predict_probability(features, coefficients, bias, threshold=0.5):
    """Calculate probability and prediction using sigmoid function."""
    z = bias + np.dot(features, coefficients)
    prob = sigmoid(z)
    pred = 1 if prob >= threshold else 0
    return prob, pred

# Sample data
feature1, feature2 = 1.5, -0.8
bias = 0.5
coef1, coef2 = 0.8, -0.3

# Manual calculation
z = bias + (coef1 * feature1) + (coef2 * feature2)
probability = sigmoid(z)
prediction = 1 if probability >= 0.5 else 0

print(f"Linear combination z: {z:.4f}")
print(f"Probability of class 1: {probability:.4f}")
print(f"Predicted class: {prediction}")

# Test function
prob, pred = predict_probability([1.5, -0.8], [0.8, -0.3], 0.5)
print(f"Test Function -> Probability: {prob:.4f}, Prediction: {pred}")


# --- Exercise 3: Confusion Matrix ---
print("\n--- Exercise 3: Confusion Matrix ---")

y_true = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]

def calculate_confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1: TP += 1
        elif true == 0 and pred == 0: TN += 1
        elif true == 0 and pred == 1: FP += 1
        elif true == 1 and pred == 0: FN += 1
    return TP, TN, FP, FN

TP, TN, FP, FN = calculate_confusion_matrix(y_true, y_pred)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Visualization
conf_matrix = np.array([[TN, FP], [FN, TP]])
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Week 8: Confusion Matrix')
plt.show()


# --- Exercise 4: Metrics ---
print("\n--- Exercise 4: Classification Metrics ---")

def calculate_metrics(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = calculate_metrics(TP, TN, FP, FN)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# ==============================================================================
# WEEK 7 HOMEWORK SOLUTIONS (Bayes, Linear Regression, Gradient Descent)
# ==============================================================================

print("\n" + "="*40)
print("WEEK 7 SOLUTIONS")
print("="*40)

# --- Problem A: Bayesian Dice Game ---
print("\n--- Problem A: Bayesian Dice Game ---")

dice_probs = {'A': 0.1, 'B': 0.3, 'C': 0.6}
dice_names = ['A', 'B', 'C']
prior = np.array([1/3, 1/3, 1/3])
n_rolls = 10

def binomial_prob(n, k, p):
    return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

def simulate_round():
    die_idx = np.random.choice([0, 1, 2])
    die_name = dice_names[die_idx]
    k = np.random.binomial(n_rolls, dice_probs[die_name])
    return die_name, k

def posterior_given_k(k):
    likelihoods = np.array([binomial_prob(n_rolls, k, dice_probs[d]) for d in dice_names])
    numerator = likelihoods * prior
    evidence = np.sum(numerator)
    return numerator / evidence

# Simulation
true_die, k = simulate_round()
posterior = posterior_given_k(k)
print(f"Observed {k} sixes. True die: {true_die}")
for die, p in zip(dice_names, posterior):
    print(f"P({die}|k): {p:.3f}")

# Accuracy check
correct = 0
for _ in range(100):
    t_die, k_obs = simulate_round()
    pred_die = dice_names[np.argmax(posterior_given_k(k_obs))]
    if pred_die == t_die: correct += 1
print(f"Inference Accuracy (100 runs): {correct/100:.2f}")

# Plot Likelihoods
ks = np.arange(0, 11)
plt.figure(figsize=(8, 5))
for die, p in dice_probs.items():
    plt.plot(ks, [binomial_prob(n_rolls, k, p) for k in ks], marker='o', label=f"Die {die} (p={p})")
plt.title('Week 7: Likelihoods')
plt.legend()
plt.show()


# --- Problem B: Linear Regression ---
print("\n--- Problem B: Linear Regression ---")

x_lin = np.array([-2, -1, 0, 1, 2])
y_lin = np.array([7, 4, 3, 4, 7])
X_lin = np.c_[np.ones(len(x_lin)), x_lin]

# Normal Equation
theta = np.linalg.inv(X_lin.T @ X_lin) @ X_lin.T @ y_lin
y_pred_lin = X_lin @ theta
mse = np.mean((y_lin - y_pred_lin)**2)

print(f"Theta: {theta}")
print(f"MSE: {mse}")

plt.scatter(x_lin, y_lin, color='red')
plt.plot(x_lin, y_pred_lin, label='Fit')
plt.title('Week 7: Linear Regression Fit')
plt.legend()
plt.show()


# --- Problem C: Gradient Descent ---
print("\n--- Problem C: Gradient Descent ---")

def grad_descent_vals(w0, alpha, steps):
    w = w0
    history = [w]
    for i in range(steps):
        grad = 20 * (w - 11)**3  # Derivative of 5(w-11)^4
        w = w - alpha * grad
        history.append(w)
    return np.array(history)

# Run descent
hist_140 = grad_descent_vals(13, 1/400, 200)
hist_180 = grad_descent_vals(13, 1/4000000, 200)

print("First 5 steps (alpha=1/400):", hist_140[:5])

# Plot Descent
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_140)
plt.title('Alpha = 1/400')
plt.axhline(11, color='r', linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(hist_180)
plt.title('Alpha = 1/4,000,000')
plt.axhline(11, color='r', linestyle='--')
plt.tight_layout()
plt.show()
