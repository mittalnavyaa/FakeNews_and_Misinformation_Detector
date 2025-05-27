# Fake News and Misinformation Detector

This project aims to detect fake news and misinformation using multiple datasets and machine learning models.

---

## Dataset Sources

- **GossipCop**:  
  You need to download the GossipCop dataset from [Hugging Face - GossipCop](https://huggingface.co/datasets/Jinyan1/GossipCop)

- **WELFake**:  
  You need to download the WELFake dataset from [Hugging Face - WELFake](https://huggingface.co/datasets/davanstrien/WELFake)

- **PolitiFact**:  
  You need to download the PolitiFact dataset from [Hugging Face - PolitiFact](https://huggingface.co/datasets/Jinyan1/PolitiFact)

---

## Project Structure and File Descriptions

### GossipCop Folder
- **GossipCop_EDA.py**  
  Performs Exploratory Data Analysis (EDA) on the GossipCop dataset, generating insights, visualizations, and word clouds.

---

### WELFake Folder
- **WELFake_preprocess.py**  
  Preprocessing script for cleaning and preparing the WELFake dataset before training models.

- **WELFake_EDA.py**  
  Script to perform exploratory data analysis and visualize features in the WELFake dataset.

---

### PolitiFact Folder
- **PolitiFact_EDA.ipynb**  
  Jupyter notebook for exploratory data analysis on the PolitiFact dataset.

- **Model_PolitiFact.ipynb**  
  Jupyter notebook for training and evaluating models on the PolitiFact dataset using TF-IDF and Logistic Regression.

---

## Notes
- Large dataset files like `.csv` and `.parquet` are **not pushed to GitHub** due to size and GitHub limits.
- Please download the datasets manually from the provided Hugging Face links.
- Make sure to place dataset files in the respective folders before running the scripts.

---

Feel free to contribute, report issues, or ask questions!

---

*Project maintained by Navya Mittal and Lohitaksha Guha*
