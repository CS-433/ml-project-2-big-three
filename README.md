# Twitter Sentiment Classification

<br>

## Authors (big-three)

* Ilias Merigh
* Nicolas Filimonov
* Thai-Nam Hoang

## Table of Contents

<p>
  <a href="#introduction-">Introduction</a> •
  <a href="#setup-">Setup</a> •
  <a href="#project-structure-">Project structures</a> •
  <a href="#gathering-data-">Gathering data</a> •
  <a href="#timeline-">Timeline</a> •
  <a href="#team-organization-">Team Organization</a>
</p>

## Introduction

This project aims to classify tweets as negative (-1) or non-negative (1) using a blend of machine learning and deep
learning techniques. We implement various methods to represent tweets (TF-IDF, Glove embeddings). We employed
traditional machine learning models like Stochastic Gradient Descent (SGD) and Logistic
Regression for their efficiency in text classification. To capture the nuanced context of language, we integrated the
Gated Recurrent Unit (GRU), a neural network adept at processing sequential data. Additionally, we utilized BERT (
Bidirectional Encoder Representations from Transformers), a cutting-edge deep learning model, to enhance our
understanding of complex language patterns in tweets. This combination of diverse approaches ensures a comprehensive and
accurate sentiment analysis of Twitter data.

## Setup

Make sure to install conda on your machine first. Then, run the following commands to create the environment and install
packages:

```bash
conda list --export > requirements.txt
conda env create -f my_environment.yml
conda activate tweet-sentiment
```

Alternatively, you can install the packages manually:

```bash
pip install -r requirements.txt
```

## Project structures

This structure was used during the project's development, and we recommend sticking to it because the locations of all
files are organized according to this framework.

`data`: contains the raw data and the preprocessed data.

`models`: contains the GRU and BERT models, inherited from the `Model` abstract class.

`notebooks`: contains the notebooks used for data exploration and model training.

`submissions`: contains the submissions to AIcrowd.

`utility`: contains decorators, file path and resources for preprocessing the tweets.

`weights`: contains the saved weights.

`preprocess.py`: contains the preprocessing pipeline.

`run.py`: run the model that yields the best submission on AICrowd.

## Gathering data

### Step 1. Download raw data

Download raw data from the [AICrowd site](https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files),
extract and put into `data` folder. The structure should be as followed:

```
├── data
│   ├── train_pos.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   └── test_data.txt
```

### Step 2. Download the GloVe embedding

We used GloVe embedding from Stanford NLP. You can download it from
[their website](https://nlp.stanford.edu/projects/glove/), extract and use the `glove.twitter.27B.100d.txt` or using
[this link](https://drive.google.com/file/d/1jUFh6uWs5rpPRj0ngi-vOsc1QLVJ-U6z/view?usp=drive_link) to download directly
without extracting. Afterward, put it into the data folder.

**Required space**: 974Mb for `glove.twitter.27B.100d.txt`

```
├── data
│   ├── glove.twitter.27B.100d.txt
│   ├── test_data.txt
│   ├── train_neg.txt
│   ├── train_neg_full.txt
│   ├── train_pos.txt
│   └── train_pos_full.txt
```

### Step 3. Preprocessed data

Download the preprocessed data and put it into the data folder. You can download it
from [this link](https://drive.google.com/drive/folders/1b9YH1vRdGKUFq0TQNtcKmMRG-7D8EfIV?usp=drive_link).

```
├── data
│   ├── preprocessed
│   │   ├── bert
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   ├── gru
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   ├── ml
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   ├── glove.twitter.27B.100d.txt
│   ...
│   └── train_pos_full.txt
```

### Step 4. Weights

Weights are essential to give predictions from model without retraining. You can download the weights
from [this link](https://drive.google.com/drive/folders/1lRFsM6QaWmykkHzVE6jAQDe34fqU-XAK?usp=drive_link)