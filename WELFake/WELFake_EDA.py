import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from pathlib import Path

# 1. Download stopwords
nltk.download('stopwords')

# 2. Load WELFake dataset
base_path = Path("C:/Users/navya/OneDrive/Desktop/FakeNewsDetector/WELFake")
df = pd.read_csv(base_path / "WELFake.csv")

# 3. Create veracity column (assuming 0=Real, 1=Fake)
df['veracity'] = df['label'].map({0: 'Real', 1: 'Fake'})

# 4. Print basic information
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass Distribution:")
print(df['veracity'].value_counts())

# ====================
# EDA VISUALIZATIONS
# ====================

plt.figure(figsize=(15, 5))

# Plot 1: Overall distribution
plt.subplot(1, 3, 1)
sns.countplot(data=df, x='veracity', palette={'Real': 'green', 'Fake': 'red'})
plt.title('News Veracity Distribution')
plt.xlabel('')
plt.ylabel('Count')

# Plot 2: Text length distribution
plt.subplot(1, 3, 2)
df['text_length'] = df['text'].str.len()
sns.boxplot(data=df, x='veracity', y='text_length',
           palette={'Real': 'green', 'Fake': 'red'})
plt.title('Text Length by Veracity')
plt.xlabel('')
plt.ylabel('Characters (log scale)')
plt.yscale('log')

# Plot 3: Title length distribution
plt.subplot(1, 3, 3)
df['title_length'] = df['title'].str.len()
sns.boxplot(data=df, x='veracity', y='title_length',
           palette={'Real': 'green', 'Fake': 'red'})
plt.title('Title Length by Veracity')
plt.xlabel('')
plt.ylabel('Characters')

plt.tight_layout()
plt.show()

# ====================
# WORD CLOUD GENERATION
# ====================

# Prepare text data for word clouds (combining title and text)
fake_text = ' '.join(df[df['veracity'] == 'Fake'][['title', 'text']].astype(str).agg(' '.join, axis=1))
real_text = ' '.join(df[df['veracity'] == 'Real'][['title', 'text']].astype(str).agg(' '.join, axis=1))

# Set up stopwords
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
output_dir = Path("C:/Users/navya/OneDrive/Desktop/FakeNewsDetector/WELFake")
output_dir.mkdir(parents=True, exist_ok=True)

# Save word clouds
fake_wc.to_file(output_dir / 'fake_news_wordcloud.png')
real_wc.to_file(output_dir / 'real_news_wordcloud.png')

print("\nOutputs saved in:", output_dir)
print("- fake_news_wordcloud.png")
print("- real_news_wordcloud.png")