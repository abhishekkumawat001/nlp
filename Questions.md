# Classify Product Type

### Environment
Ubuntu 22.04 LTS which includes **Python 3.9.12** and utilities *curl*, *git*, *vim*, *unzip*, *wget*, and *zip*. There is no *GPU* support.

The IPython Kernel allows you to execute Python code in the Notebook cell and Python console.

### Installing packages
- Run `!mamba list "package_name"` command to check the package installation status. For example,

```python
!mamba list numpy
"""
# packages in environment at /opt/conda:
#
# Name                    Version                   Build  Channel
numpy                     1.21.6           py39h18676bf_0    conda-forge
"""
```

    You can also try importing the package.

- Run the `!mamba install "package_name"` to install a package

### Excluding large files
HackerRank rejects any submission larger than **20MB**. Therefore, you must exclude any large files by adding these to the *.gitignore* file.
You can **Submit** code to validate the status of your submission.

## Data Description

| Column    | Description                                         |
|:----------|:----------------------------------------------------|
| `desc`  | description of product                                |
| `label`   | Product Type (0-Movie DVD, 1-Electronics, 2-Kitchen Appliances) |


```python
!mamba install pandas 
!mamba install numpy Tokenizer pad_sequence
```


```python
# Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text
import Tokenzier 
import pad_sequences
pd.set_option("display.max_columns", 101)
pd.set_option("display.max_colwidth", None)
```


```python
# The training dataset is already loaded below.
data = pd.read_csv("train.csv")
data.head()
```


```python
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words, oov_token ="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(test_df['desc'])

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
```


```python

```

## Deep Learning

Build a neural network that can classify the product types.
- **The model's performance will be evaluated on the basis of accuracy score.**


```python

```


```python

```


```python

```

> #### Task:
- **Submit the predictions on the test dataset using your optimized model** <br/>
    For each record in the test set (test.csv), predict the value of the `label` variable.  You should submit a CSV file with a header row and one row per test entry.

The file (`submissions.csv`) should have exactly 2 columns:


| Column    | Description                                         |
|:----------|:----------------------------------------------------|
| `desc`  | description of product                                |
| `label`   | Product Type (0-Movie DVD, 1-Electronics, 2-Kitchen Appliances) |


```python
# The testing dataset is already loaded below.
test = pd.read_csv("test.csv")
test.head()
```


```python

```


```python

```


```python
# Submission
submission_df.to_csv("submissions.csv", index=False)
```
