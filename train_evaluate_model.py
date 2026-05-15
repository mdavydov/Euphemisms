import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load training and test embeddings
train_df = pd.read_csv('embeddings_train.csv')
test_df = pd.read_csv('embeddings_test.csv')

embedding_cols = [col for col in train_df.columns if col.startswith('emb_')]
X_train_full = train_df[embedding_cols].values
y_train_full = train_df['label'].values
X_test = test_df[embedding_cols].values
y_test = test_df['label'].values

print(f"Training data: {len(X_train_full)} examples")
print(f"Test data: {len(X_test)} examples")

# Training percentages to test (percentage of the training set to use)
train_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Store results
results = {
    'train_percentage': [],
    'train_sentences': [],
    'test_sentences': [],
    'true_positives': [],
    'true_negatives': [],
    'false_positives': [],
    'false_negatives': [],
    'precision': [],
    'recall': [],
    'f1': []
}

# Train and evaluate for each percentage
for train_pct in train_percentages:
    if train_pct == 100:
        X_train = X_train_full
        y_train = y_train_full
    else:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=train_pct/100, random_state=42
        )
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the fixed test set
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_val = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store results
    total_test = len(y_test)
    results['train_percentage'].append(train_pct)
    results['train_sentences'].append(len(X_train))
    results['test_sentences'].append(total_test)
    results['true_positives'].append(round(100 * tp / total_test, 2))
    results['true_negatives'].append(round(100 * tn / total_test, 2))
    results['false_positives'].append(round(100 * fp / total_test, 2))
    results['false_negatives'].append(round(100 * fn / total_test, 2))
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1_val)
    
    print(f"Train: {train_pct}% ({len(X_train)} samples) | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_val:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(results['train_percentage'], results['precision'], marker='o', label='Precision', linewidth=2)
plt.plot(results['train_percentage'], results['recall'], marker='s', label='Recall', linewidth=2)
plt.plot(results['train_percentage'], results['f1'], marker='^', label='F1 Score', linewidth=2)

plt.xlabel('Training Data Percentage (%)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance vs Training Data Size', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(train_percentages)
plt.ylim([0, 1.05])

# Save the plot
plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'model_performance.png'")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('training_results.csv', index=False)
print("Results saved to 'training_results.csv'")

plt.show()
