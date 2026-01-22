import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('embeddings_with_labels.csv')

# Extract embedding columns (emb_0 to emb_1535)
embedding_cols = [f'emb_{i}' for i in range(1536)]
X = df[embedding_cols].values
y = df['label'].values

# Training percentages to test
train_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Store results
results = {
    'train_percentage': [],
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
    # Split data: train_pct% for training, rest for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_pct/100, random_state=42, stratify=y
    )
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store results
    total_test = len(y_test)
    results['train_percentage'].append(train_pct)
    results['test_sentences'].append(total_test)
    results['true_positives'].append(round(100 * tp / total_test, 2))
    results['true_negatives'].append(round(100 * tn / total_test, 2))
    results['false_positives'].append(round(100 * fp / total_test, 2))
    results['false_negatives'].append(round(100 * fn / total_test, 2))
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    
    print(f"Train: {train_pct}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

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
