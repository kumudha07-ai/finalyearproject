# Import necessary libraries
import pickle  # For serializing and de-serializing Python objects
from sklearn.ensemble import RandomForestClassifier  # For creating a Random Forest Classifier model
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For evaluation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For better visualizations

print("=" * 60)
print("Sign Language Recognition - Model Training")
print("=" * 60)

# Load the data from 'data.pickle'
print("\n[1/5] Loading dataset...")
data_dict = pickle.load(open('data.pickle', 'rb'))

# Convert data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"      Total samples: {len(data)}")
print(f"      Features per sample: {len(data[0]) if len(data) > 0 else 0}")
print(f"      Classes: {np.unique(labels)}")

# Show class distribution
print("\n[2/5] Class distribution:")
unique_labels = np.unique(labels)
class_counts = []
for label in unique_labels:
    count = np.sum(labels == label)
    class_counts.append(count)
    print(f"      Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")

# Split the data into training and testing sets
print("\n[3/5] Splitting data (60% train, 20% validation, 20% test)...")
x_temp, x_test, y_temp, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.25, shuffle=True, stratify=y_temp, random_state=42
)
print(f"      Training set: {len(x_train)} samples")
print(f"      Validation set: {len(x_val)} samples")
print(f"      Test set: {len(x_test)} samples")

# Initialize an optimized Random Forest Classifier model
print("\n[4/5] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=20,           # Maximum depth of trees
    min_samples_split=5,    # Minimum samples to split
    min_samples_leaf=2,     # Minimum samples in leaf
    random_state=42,        # For reproducibility
    n_jobs=-1               # Use all CPU cores
)

# Train the model using the training data
model.fit(x_train, y_train)
print("      Training complete!")

# Validate the model
print("\n[5/5] Evaluating model performance...")

# Validation accuracy
val_predict = model.predict(x_val)
val_score = accuracy_score(y_val, val_predict)
print(f"\n      Validation Accuracy: {val_score * 100:.2f}%")

# Test accuracy
y_predict = model.predict(x_test)
test_score = accuracy_score(y_test, y_predict)
print(f"      Test Accuracy: {test_score * 100:.2f}%")

# Cross-validation score
cv_scores = cross_val_score(model, data, labels, cv=5)
print(f"      Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# Classification Report
print("\n" + "-" * 60)
print("Classification Report (Test Set):")
print("-" * 60)
print(classification_report(y_test, y_predict))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print("-" * 60)
print("Confusion Matrix:")
print("-" * 60)
print(cm)

# Save the trained model to a pickle file
print("\n" + "=" * 60)
model_file = 'model.p'
with open(model_file, 'wb') as f:
    pickle.dump({'model': model}, f)
print(f"Model saved to: {model_file}")

# ============================================================
# PLOTTING SECTION
# ============================================================
print("\nGenerating plots...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Class Distribution
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(unique_labels)))
bars = plt.bar(unique_labels, class_counts, color=colors, edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution')
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(count), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')

# Plot 2: Train/Val/Test Split
plt.figure(figsize=(8, 6))
split_labels = ['Train', 'Validation', 'Test']
split_sizes = [len(x_train), len(x_val), len(x_test)]
pie_colors = ['#2ecc71', '#3498db', '#e74c3c']
wedges, texts, autotexts = plt.pie(split_sizes, labels=split_labels, autopct='%1.1f%%', 
                                    colors=pie_colors, explode=(0.05, 0.05, 0.05),
                                    shadow=True, startangle=90)
plt.title('Dataset Split')
plt.tight_layout()
plt.savefig('dataset_split.png', dpi=150, bbox_inches='tight')

# Plot 3: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels,
            cbar=True, annot_kws={'size': 14, 'weight': 'bold'})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')

# Plot 4: Accuracy Comparison
plt.figure(figsize=(8, 6))
accuracy_labels = ['Validation', 'Test', 'Cross-Val']
accuracy_values = [val_score * 100, test_score * 100, cv_scores.mean() * 100]
bar_colors = ['#9b59b6', '#e74c3c', '#f39c12']
bars = plt.bar(accuracy_labels, accuracy_values, color=bar_colors, edgecolor='black')
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy')
plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
for bar, val in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('model_accuracy.png', dpi=150, bbox_inches='tight')

# Plot 5: Feature Importance (Top 20)
plt.figure(figsize=(8, 6))
feature_importance = model.feature_importances_
top_n = min(20, len(feature_importance))
indices = np.argsort(feature_importance)[-top_n:]
plt.barh(range(top_n), feature_importance[indices], color='teal', edgecolor='black')
plt.yticks(range(top_n), [f'Feature {i}' for i in indices])
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')

# Plot 6: Cross-Validation Scores
plt.figure(figsize=(8, 6))
fold_labels = [f'Fold {i+1}' for i in range(5)]
plt.bar(fold_labels, cv_scores * 100, color='#1abc9c', edgecolor='black')
plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean()*100:.1f}%')
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)')
plt.title('Cross-Validation Scores')
plt.legend()
for i, v in enumerate(cv_scores * 100):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('cross_validation_scores.png', dpi=150, bbox_inches='tight')

print("Plots saved separately as PNG files.")
plt.show()

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"  Total Samples:        {len(data)}")
print(f"  Training Samples:     {len(x_train)}")
print(f"  Validation Accuracy:  {val_score * 100:.2f}%")
print(f"  Test Accuracy:        {test_score * 100:.2f}%")
print(f"  Cross-Val Accuracy:   {cv_scores.mean() * 100:.2f}%")
print(f"  Model File:           {model_file}")
print(f"  Plot Files:           Separate .png files generated")
print("=" * 60)
print("\nReady to use! Run Sign_classifier.py to test real-time recognition.")
