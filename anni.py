import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# The dataset is provided as 'as1-bank.csv'
try:
    df = pd.read_csv('as1-bank.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'as1-bank.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Identify features (X) and target (y)
# The target variable is 'y' (binary classification: 'yes' or 'no')
X = df.drop('y', axis=1)
y = df['y']

# Convert 'y' target variable to numerical (0 and 1)
# 'no' -> 0, 'yes' -> 1
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Target variable 'y' encoded: {le.classes_[0]} -> 0, {le.classes_[1]} -> 1")

# Identify categorical and numerical features
# Based on the assignment description, many categorical variables are already numeric.
# However, 'default', 'housing', 'loan', 'poutcome' are still 'yes'/'no' or categorical strings.
# Let's re-evaluate based on the snippet provided:
# 'age', 'balance', 'duration', 'campaign', 'pdays', 'previous' are numerical.
# 'marital', 'education', 'contact' are already numeric (0, 1, 2, etc.).
# 'default', 'housing', 'loan', 'poutcome' are 'yes'/'no' or other categories.

# For simplicity and based on the assignment stating "Largely the data has already been transformed
# ready for the task; however, you should consider how many inputs you wish to provide to the network",
# we will treat 'default', 'housing', 'loan', 'poutcome' as categorical for one-hot encoding,
# even if some are binary 'yes'/'no' which could be label encoded.
# The remaining 'marital', 'education', 'contact' are already numeric and can be treated as such
# or one-hot encoded if their numeric values imply order when none exists.
# For this example, we will one-hot encode all non-numeric columns that are not 'y'.

categorical_cols = X.select_dtypes(include='object').columns
print(f"Categorical columns identified for one-hot encoding: {list(categorical_cols)}")

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) # drop_first to avoid multicollinearity

# Split data into training and testing sets
# Using a common split ratio, e.g., 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# Scale numerical features
# It's crucial to scale features for ANNs to ensure optimal performance.
# We fit the scaler only on the training data to prevent data leakage.
numerical_cols_after_ohe = X_train.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[numerical_cols_after_ohe] = scaler.fit_transform(X_train[numerical_cols_after_ohe])
X_test[numerical_cols_after_ohe] = scaler.transform(X_test[numerical_cols_after_ohe])
print("Numerical features scaled using StandardScaler.")

# --- ANN Model Definition and Training Function ---
print("\n--- ANN Model Definition and Training ---")

def create_and_train_ann(input_dim, hidden_layers_config, learning_rate=0.01, epochs=50, batch_size=32):
    """
    Creates, compiles, and trains a Feedforward Artificial Neural Network.

    Args:
        input_dim (int): Number of input features.
        hidden_layers_config (list of int): A list where each element represents the
                                            number of nodes in a hidden layer.
                                            E.g., [64, 32] means two hidden layers
                                            with 64 and 32 nodes respectively.
        learning_rate (float): Learning rate for the SGD optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: (model, history) - the trained Keras model and its training history.
    """
    model = Sequential()
    # Input layer
    model.add(Dense(hidden_layers_config[0] if hidden_layers_config else 10,
                    input_dim=input_dim, activation='relu')) # Default to 10 nodes if no hidden layers specified

    # Hidden layers
    for nodes in hidden_layers_config[1:]: # Iterate from the second hidden layer if any
        model.add(Dense(nodes, activation='relu'))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with SGD optimizer (as required by assignment)
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(f"\n--- Training ANN with architecture: Input ({input_dim}) -> {hidden_layers_config} -> Output (1) ---")
    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1, # Use a small validation split from training data
                        verbose=0) # Set verbose to 0 to suppress training output per epoch

    return model, history

# --- Experimentation with Different Architectures ---
print("\n--- Starting ANN Architecture Experiments ---")

# Define different ANN architectures to experiment with
# Each tuple contains (description, list of hidden layer nodes)
architectures = [
    ("Single Hidden Layer (32 nodes)", [32]),
    ("Single Hidden Layer (64 nodes)", [64]),
    ("Two Hidden Layers (64, 32 nodes)", [64, 32]),
    ("Two Hidden Layers (128, 64 nodes)", [128, 64]),
    ("Three Hidden Layers (128, 64, 32 nodes)", [128, 64, 32]),
]

results = {}

for desc, config in architectures:
    print(f"\nRunning experiment for: {desc}")
    model, history = create_and_train_ann(X_train.shape[1], config)

    # Evaluate the model on the test data
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[desc] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "history": history
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# --- Presenting Results and Visualizations ---
print("\n--- Experiment Results Summary ---")

# Create a DataFrame for easy comparison
metrics_df = pd.DataFrame({
    "Architecture": [desc for desc, _ in architectures],
    "Accuracy": [results[desc]["accuracy"] for desc, _ in architectures],
    "Precision": [results[desc]["precision"] for desc, _ in architectures],
    "Recall": [results[desc]["recall"] for desc, _ in architectures],
    "F1-Score": [results[desc]["f1_score"] for desc, _ in architectures],
    "ROC AUC": [results[desc]["roc_auc"] for desc, _ in architectures],
})

print(metrics_df.round(4))

# Plotting training history (Accuracy and Loss) for each architecture
print("\n--- Plotting Training History (Loss and Accuracy) ---")

for desc, res in results.items():
    history = res["history"]
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{desc} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{desc} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plotting Confusion Matrices for each architecture
print("\n--- Plotting Confusion Matrices ---")
for desc, res in results.items():
    cm = res["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title(f'{desc} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

print("\n--- Analysis and Conclusion Guidance ---")
print("Based on the metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and the plots:")
print("1. Compare the performance across different architectures. Which one consistently performs better?")
print("2. Look at the training and validation loss/accuracy curves. Are there signs of overfitting (train loss decreasing, val loss increasing) or underfitting (both train and val loss high)?")
print("3. Analyze the confusion matrices. Which architecture minimizes false positives and false negatives, considering the problem context (e.g., is it more critical to identify all 'yes' cases, or to avoid misclassifying 'no' as 'yes'?)")
print("4. Consider the trade-offs between model complexity (number of layers/nodes) and performance. A simpler model might be preferred if its performance is comparable to a more complex one.")
print("5. The `pdays` column has a value of -1 for customers not previously contacted. This might need special handling (e.g., converting -1 to a specific category or imputing). The current preprocessing treats it as a numerical value, which might not be optimal.")
print("6. The `duration` feature is highly predictive but is known to cause data leakage in real-world scenarios as it's only known after the call. For this assignment, it's used as-is, but in a real system, it would be excluded or handled differently.")
print("\nThis script provides the practical implementation and evaluation framework. You can use these results to write your report, justifying your choice of optimal architecture and discussing the experimental findings.")
