import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (using raw string for file path)
df = pd.read_parquet(r'C:\Users\navya\OneDrive\Desktop\FakeNews Project\GossipCop\gossipcop_combined.parquet')

# Print column names to see what's available
print("Available columns:", df.columns.tolist())

# Combine title and text
df['combined_text'] = df['title'] + ' ' + df['text']

# Prepare features and target
X = df['combined_text']
y = df['veracity']

# First split: 80% train, 20% remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: split the remaining 20% into validation and test (50% each, resulting in 10-10 split)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, 
                        stop_words='english',
                        lowercase=True)

# Transform the text data
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Make predictions on validation set
val_predictions = lr_model.predict(X_val_tfidf)

# Evaluate on validation set
print("\nValidation Set Performance:")
print(classification_report(y_val, val_predictions))
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")

# Make predictions on test set
test_predictions = lr_model.predict(X_test_tfidf)

# Evaluate on test set
print("\nTest Set Performance:")
print(classification_report(y_test, test_predictions))
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions):.4f}")

# Define the output directory
output_dir = r'C:\Users\navya\OneDrive\Desktop\FakeNews Project\GossipCop'

# Save classification reports
with open(f'{output_dir}/classification_report_validation.txt', 'w') as f:
    f.write("Validation Set Performance:\n")
    f.write(classification_report(y_val, val_predictions))
    f.write(f"\nValidation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")

with open(f'{output_dir}/classification_report_test.txt', 'w') as f:
    f.write("Test Set Performance:\n")
    f.write(classification_report(y_test, test_predictions))
    f.write(f"\nTest Accuracy: {accuracy_score(y_test, test_predictions):.4f}")

# Plot and save confusion matrix for test set
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{output_dir}/confusion_matrix_test.png', bbox_inches='tight', dpi=300)
plt.show()

# Plot and save confusion matrix for validation set
plt.figure(figsize=(8, 6))
cm_val = confusion_matrix(y_val, val_predictions)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{output_dir}/confusion_matrix_validation.png', bbox_inches='tight', dpi=300)
plt.show()