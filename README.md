# Topic-Seg-Label

Codes for the paper "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning", and you can find our paper at [here](https://doi.org/10.24963/ijcai.2018/612)

Cite this paper:
```
@inproceedings{takanobu2018weakly,
  title={A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning},
  author={Takanobu, Ryuichi and Huang, Minlie and Zhao, Zhongzhou and Li, Fenglin and Chen, Haiqing and Nie, Liqiang and Zhu, Xiaoyan},
  booktitle={IJCAI-ECAI},
  pages={4403--4410},
  year={2018}
}
```

## Data Format

**Note:** Since the dialogue datasets used in the paper derive from an E-commerce services platform, to protect users' privacy concerns, we are sorry that the datasets will not be published at present. So, we provide the data format instead here for others to use our codes with their own corpus.

As can be seen in the directory `./code/example`, the data required for the model are separated into 7 files:

### my_vector

This file contains the word embedding vector for the datasets. Each row consists of a floating point array of length `embed_units`, and there are `symbols` rows in all .

### XXX.csv

The main data. The file contains multiple dialogue sessions in the following format:

```
N,
label_1, data_1
label_2, data_2
...
label_N, data_N
```

where `label_i` is the topic category for the ith sentence `data_i` (The labels of training/validation data are obtained according to `3.1 Noisy Labeling with Prior Knowledge` in the paper, which are noisy)

The words of ith sentence (i.e. `data_i`) is in the form of `string`, which are concatenated with `/` signs, and each word is replaced by its index in `my_vector` file.

### keywords_XXX

The file contains the related keywords information for each dialogue sessions:

```
key_1, value_1
key_2, value_2
...
key_N, value_N
```

where `key_i` is the topic category assigned by `keyword matching` but without `nearest neighboring` (see `3.1 Noisy Labeling with Prior Knowledge` for details), and `value_i` for the corresponding frequency value (i.e. The keywords of topic `key_i`have appeared `value_i` times in the sentence `data_i`.)

If a sentence contains no keyword, or there are two topic categories that both's keywords appear most times in a sentence, `key_i`will be set to `-1`. In this case, the `value_i` is also set to `-1`.

## Run

Command 
```
cd code
python model.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Change the corresponding options to set hyperparameters:
```
tf.app.flags.DEFINE_integer("symbols", 12210, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 7, "Number of topic labels.")
tf.app.flags.DEFINE_integer("epoch_pre", 2, "Number of epoch on pretrain.")
tf.app.flags.DEFINE_integer("epoch_max", 15, "Maximum of epoch in iterative training.")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("hidden_units", 100, "Size of hidden layer.")
tf.app.flags.DEFINE_integer("sample_round", 4, "Sample round in RL.")
tf.app.flags.DEFINE_float("keyword", 3.0, "Coefficient of keyword reward.")
tf.app.flags.DEFINE_float("continuity", 1.0, "Coefficient of continuity reward.")
tf.app.flags.DEFINE_float("learning_rate_srn", 0.0001, "SRN Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_pn", 0.00001, "PN Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Drop out rate.")
tf.app.flags.DEFINE_float("softmax_smooth", 0.5, "Discount rate in softmax.")
tf.app.flags.DEFINE_float("threshold", 0.005, "Threshold to judge the convergence.")

tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters.")
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory.")
tf.app.flags.DEFINE_string("train_filename", "train", "Filename for training set.")
tf.app.flags.DEFINE_string("valid_filename", "valid", "Filename for validation set.")
tf.app.flags.DEFINE_string("test_filename", "test", "Filename for test set.")
tf.app.flags.DEFINE_string("word_vector_filename", "my_vector", "Filename for word embedding vector.")
tf.app.flags.DEFINE_string("train_dir_srn", "./train_srn", "SRN Training directory.")
tf.app.flags.DEFINE_string("train_dir_pn", "./train_pn", "PN Training directory.")
```

The ``states`` of SRN are saved at ``FLAGS.train_dir_srn + '/states_' + FLAGS.XXX_filename`` 

The ``labels`` of PN are saved at ``FLAGS.train_dir_pn + '/labels_' + FLAGS.XXX_filename`` 

### Requirements

tensorflow >= 1.0
