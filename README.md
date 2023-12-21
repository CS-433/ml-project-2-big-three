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
  <a href="#notebooks-">Notebooks</a> •
  <a href="#best-submissions-">Best submissions</a> •
  <a href="#contact-information-">Contact information</a> •
</p>

## Introduction

This project aims to classify tweets as negative (-1) or non-negative (1) using a blend of machine learning and deep
learning techniques. We implement various methods to represent
tweets ([TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[Glove embeddings](https://nlp.stanford.edu/projects/glove/)). We employed
traditional machine learning models like Stochastic Gradient
Descent ([SGD](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)) and [Logistic
Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
) for their efficiency in text classification. To capture the nuanced context of language, we integrated the
Gated Recurrent Unit ([GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)), a neural network adept at
processing sequential data. Additionally, we utilized a
transformer-based model [BERT](https://arxiv.org/abs/1810.04805) and its
variation [RoBERTa](https://arxiv.org/abs/1907.11692), to enhance our understanding of complex language patterns in
tweets. This combination of diverse approaches ensures a comprehensive and
accurate sentiment analysis of Twitter data.

For more details, read the [report.pdf](./report.pdf)

## Setup

Make sure to install conda on your machine first. Then, run the following commands to create the environment and install
packages:

```bash
conda env create -f environment.yml
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

`preprocessing.py`: contains the preprocessing pipeline.

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

### Step 3. Download the Word2Vec embedding

We used Word2Vec embedding from Google. You can download it
from [Kaggle](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors/), extract and put into `data` folder.

```
├── data
│   ├── preprocessed
│   ...
│   ├── glove.twitter.27B.100d.txt
│   ├── GoogleNews-vectors-negative300.bin
│   └── train_pos_full.txt
```

**Required space**: 3.64Gb

### Step 4. Preprocessed data

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

### Step 5. Weights

Weights are essential to give predictions from model without retraining. You can download the weights
from [this link](https://drive.google.com/drive/folders/1lRFsM6QaWmykkHzVE6jAQDe34fqU-XAK?usp=drive_link)

```
├── weights
│   ├── bert
│   │   ├── config.json
│   │   └── tf_model.h5
│   ├── bert-large
│   │   ├── config.json
│   │   └── tf_model.h5
│   ├── gru
│   │   ├── config.json
│   │   └── model.keras
│   └── README.md
```

**Required space**: 2.14Gb for total weights

### Step 6. Run the model

[`run.py`](run.py) is the main script to load weights and run the model. You can run it with the following command:

```bash
python3 run.py -w
```

This will run the pretrained model and load the best weight for the model. You can also run the model without loading
the weights by running:

```bash
python3 run.py
```

A more detailed help can be found by running:

```bash
python3 run.py -h
```

Submissions will be saved in the [`submissions/bert`](submissions/bert) folder, under the name `
submission_YYYY-MM-DD_HH:MM:SS.csv`.

## Notebooks

The notebooks are used for data exploration and model training. They are located in the [`notebooks`](notebooks) folder.
Those notebooks are well-documented and will give relevant information in the processing of finishing this project.
They are structured as followed:

`model_BERT.ipynb`: contains the BERT model training.

`model_GRU.ipynb`: contains the GRU model training.

`model_logistic_regression.ipynb`:  contains the logistic regression model training.

`model_RoBERTa.ipynb`: contains the RoBERTa model training.

`model_SGD.ipynb`: contains the SGD model training.

`preprocessing.ipynb`: contains the preprocessing pipeline.

`preprocessing_exploration.ipynb`: contains the data exploration and preprocessing pipeline.

## Best submissions

Our best submission on AICrowd was a BERT-based model with `bert-large-uncased` variation. After downloading the weights
and loading the files for generating predictions, it will take roughly an hour to run on a normal laptop.

The best submission can be reproduced by running the command in [Step 6](#step-6-run-the-model). If any problems
occured, please go
to [/submissions/bert/test_prediction_BERT_large.csv](./submissions/bert/test_predictions_BERT_large.csv) to download
the
submission.

## Contact information

For help or issues using this repo, please submit a GitHub issue.
For personal communication related to this project, please contact Ilias Merigh <ilias.merigh@epfl.ch>, Nicolas
Filimonov <nicolas.filimonov@epfl.ch> or Thai-Nam Hoang <thai.hoang@eplf.ch>.