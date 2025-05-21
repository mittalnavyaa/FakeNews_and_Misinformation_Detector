import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from pathlib import Path

# 1. Download stopwords
nltk.download('stopwords')

# 2. Load and combine all Parquet files
base_path = Path("C:/Users/navya/OneDrive/Desktop/FakeNewsDetector/GossipCop/")
file_paths = {
    'HF': base_path / "HF-00000-of-00001-b7ad0013efd98ff4.parquet",
    'HR': base_path / "HR-00000-of-00001-043a35ac2a425b62.parquet",
    'MF': base_path / "MF-00000-of-00001-2d256f82f8c8e2dd.parquet",
    'MR': base_path / "MR-00000-of-00001-c9324d9fd00efb16.parquet"
}

dfs = []
for split_name, file_path in file_paths.items():
    df = pd.read_parquet(file_path)
    df['split'] = split_name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# 3. Create veracity column based on splits
combined_df['veracity'] = combined_df['split'].map({
    'HF': 'Fake',
    'HR': 'Real',
    'MF': 'Fake',
    'MR': 'Real'
})

# 4. Print available columns for verification
print("Available columns:", combined_df.columns.tolist())

# ====================
# EDA VISUALIZATIONS
# ====================

plt.figure(figsize=(15, 5))

# Plot 1: Overall distribution
plt.subplot(1, 3, 1)
sns.countplot(data=combined_df, x='veracity', palette={'Real': 'green', 'Fake': 'red'})
plt.title('Overall Veracity Distribution')
plt.xlabel('')
plt.ylabel('Count')

# Plot 2: Split distribution
plt.subplot(1, 3, 2)
sns.countplot(data=combined_df, x='split', order=['HF', 'MF', 'MR', 'HR'],
             palette={'HF': 'red', 'MF': 'orange', 'MR': 'lightgreen', 'HR': 'green'})
plt.title('Distribution by Split')
plt.xlabel('Dataset Split')
plt.ylabel('')

# Plot 3: Text length
plt.subplot(1, 3, 3)
combined_df['text_length'] = combined_df['text'].str.len()
sns.boxplot(data=combined_df, x='veracity', y='text_length',
           palette={'Real': 'green', 'Fake': 'red'})
plt.title('Text Length by Veracity')
plt.xlabel('')
plt.ylabel('Characters (log scale)')
plt.yscale('log')

plt.tight_layout()
plt.show()

# ====================
# WORD CLOUD GENERATION
# ====================

# Prepare text data for word clouds
fake_text = ' '.join(combined_df[combined_df['veracity'] == 'Fake']['text'].astype(str))
real_text = ' '.join(combined_df[combined_df['veracity'] == 'Real']['text'].astype(str))

# Set up stopwords and word cloud parameters
stop_words = set(stopwords.words('english'))
extra_stopwords = {'said', 'will', 'also', 'one', 'like', 'new', 'us'}
stop_words.update(extra_stopwords)

wc_params = {
    'width': 1000,
    'height': 600,
    'background_color': 'white',
    'stopwords': stop_words,
    'max_words': 150,
    'colormap': 'viridis'
}

# Generate and display word clouds
plt.figure(figsize=(15, 8))

# Fake news word cloud
plt.subplot(1, 2, 1)
fake_wc = WordCloud(**wc_params).generate(fake_text)
plt.imshow(fake_wc, interpolation='bilinear')
plt.title('Fake News Word Cloud', fontsize=16)
plt.axis('off')

# Real news word cloud
plt.subplot(1, 2, 2)
real_wc = WordCloud(**wc_params).generate(real_text)
plt.imshow(real_wc, interpolation='bilinear')
plt.title('Real News Word Cloud', fontsize=16)
plt.axis('off')

plt.tight_layout()
plt.show()

# Save outputs
output_dir = Path("C:/Users/navya/OneDrive/Desktop/FakeNewsDetector/GossipCop")

# Create directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save word clouds and combined parquet file
fake_wc.to_file(output_dir / 'fake_news_wordcloud.png')
real_wc.to_file(output_dir / 'real_news_wordcloud.png')
combined_df.to_parquet(output_dir / 'gossipcop_combined.parquet', index=False)

print("\nOutputs saved in:", output_dir)
print("- fake_news_wordcloud.png")
print("- real_news_wordcloud.png")
print("- gossipcop_combined.parquet")