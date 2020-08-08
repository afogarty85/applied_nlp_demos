# Transformers
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import re
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import string

SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)


# tell pytorch to use cuda
device = torch.device("cuda")

# prepare and load data
def prepare_df(pkl_location):
    # read pkl as pandas
    df = pd.read_pickle(pkl_location)
    # just keep us/kabul labels
    df = df.loc[(df['target'] == 'US') | (df['target'] == 'Kabul')]
    # mask DV to recode
    us = df['target'] == 'US'
    kabul = df['target'] == 'Kabul'
    # apply mask
    df.loc[us, 'target'] = 1
    df.loc[kabul, 'target'] = 0
    # reset index
    df = df.reset_index(drop=True)
    return df


df = prepare_df('C:\\Users\\Andrew\\Desktop\\df.pkl')

# experiment 50%
df = df.sample(frac = 0.6)


# prepare data
def clean_df(df):
    # strip dash but keep a space
    df['body'] = df['body'].str.replace('-', ' ')
    # prepare keys for punctuation removal
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    # lower case the data
    df['body'] = df['body'].apply(lambda x: x.lower())
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    # remove punctuation  -- f1 improves by .05 by disabling this
    #df['body'] = df['body'].apply(lambda x: x.translate(translator))
    # generate a word count
    df['word_count'] = df['body'].apply(lambda x: len(x.split()))
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))

    return df


df = clean_df(df)


# lets remove rare words
def remove_rare_words(df):
    # get counts of each word -- necessary for vocab
    counts = Counter(" ".join(df['body'].values.tolist()).split(" "))
    # remove low counts -- keep those above 2
    counts = {key: value for key, value in counts.items() if value > 2}

    # remove rare words from corpus
    def remove_rare(x):
        return ' '.join(list(filter(lambda x: x in counts.keys(), x.split())))

    # apply funx
    df['body'] = df['body'].apply(remove_rare)
    return df


df = remove_rare_words(df)


# remove transliterated words that GloVe can't find
no_matches = np.load('translit_no_match.npy')
no_matches = dict(zip(set(no_matches), range(len(set(no_matches)))))

# remove transliterated words from corpus
df['body'] = df['body'].apply(lambda x: ' '.join(list(filter(lambda x: x not in no_matches.keys(), x.split()))))

# instantiate BERT tokenizer with upper + lower case
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# a look at some of the DistilBERT vocab
word_map = dict(zip(tokenizer.vocab.keys(), range(len(tokenizer))))
word_map.get('the')  # find index value
list(tokenizer.vocab.keys())[2000:2010]
len(tokenizer)


# how long are tokenized docs
#ls = []
#for i in range(df.shape[0]):
#    ls.append(len(tokenizer.tokenize(df.iloc[i]['body'])))

#temp_df = pd.DataFrame({'len_tokens': ls})
#temp_df['len_tokens'].mean()  # 273
#temp_df['len_tokens'].median()  # 103
#temp_df['len_tokens'].max()  # 5843

# tokenize corpus using BERT
def tokenize_corpus(df, tokenizer, max_len):
    # token ID storage
    input_ids = []
    # attension mask storage
    attention_masks = []
    # max len -- 512 is max
    max_len = max_len
    # for every document:
    for doc in df:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            doc,  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=max_len,  # set max length
                            truncation=True,  # truncate longer messages
                            pad_to_max_length=True,  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )

        # add the tokenized sentence to the list
        input_ids.append(encoded_dict['input_ids'])

        # and its attention mask (differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


input_ids, attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)

# convert the labels into tensors.
labels = torch.tensor(df['target'].values.astype(np.float32))

# show document 7422, now as a list of token ids
#print('Original: ', df.iloc[7422]['body'])
# to do - work on punctuation white space fix
#print('Token IDs:', input_ids[7422])

# token 100: unknown [UNK]
# token 101: CLS for classification tasks [CLS]
# token 102: SEP at end of each sentence; each document in our case [SEP]
# token 0: padding [PAD]


def train(model, dataloader, optimizer):

    # capture time
    total_t0 = time.time()

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1 = 0

    # put model into traning mode
    model.train()

    # for each batch of training data...
    for step, batch in enumerate(dataloader):

        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # Unpack this training batch from our dataloader:
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda().long()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            loss, logits = model(input_ids=b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            train_total_loss += loss.item()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # update the learning rate
        scheduler.step()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate preds
        rounded_preds = np.argmax(logits, axis=1).flatten()

        # calculate f1
        total_train_f1 += f1_score(rounded_preds, y_true,
                                   average='weighted',
                                   labels=np.unique(rounded_preds))

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # calculate the average f1 over all of the batches
    avg_train_f1 = total_train_f1 / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1': avg_train_f1
        }
    )

    # training time end
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {training_time:}")

    return None


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1 = 0
    total_valid_recall = 0
    total_valid_precision = 0

    # evaluate data for one epoch
    for batch in dataloader:

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda().long()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            loss, logits = model(input_ids=b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

        # accumulate validation loss
        total_valid_loss += loss.item()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate preds
        rounded_preds = np.argmax(logits, axis=1).flatten()

        # calculate f1
        total_valid_f1 += f1_score(rounded_preds, y_true,
                                   average='weighted',
                                   labels=np.unique(rounded_preds))

        # calculate accuracy
        total_valid_accuracy += accuracy_score(rounded_preds, y_true)

        # calculate precision
        total_valid_precision += precision_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

        # calculate recall
        total_valid_recall += recall_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

    # report final accuracy of validation run
    avg_accuracy = total_valid_accuracy / len(dataloader)

    # report final f1 of validation run
    global avg_val_f1
    avg_val_f1 = total_valid_f1 / len(dataloader)

    # report final f1 of validation run
    avg_precision = total_valid_precision / len(dataloader)

    # report final f1 of validation run
    avg_recall = total_valid_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val Accur.': avg_accuracy,
            'Val precision': avg_precision,
            'Val recall': avg_recall,
            'Val F1': avg_val_f1
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {training_time:}")

    return None


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_test_accuracy = 0
    total_test_loss = 0
    total_test_f1 = 0
    total_test_recall = 0
    total_test_precision = 0

    # evaluate data for one epoch
    for batch in dataloader:

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda().long()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            loss, logits = model(input_ids=b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

        # accumulate validation loss
        total_test_loss += loss.item()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate preds
        rounded_preds = np.argmax(logits, axis=1).flatten()

        # calculate f1
        total_test_f1 += f1_score(rounded_preds, y_true,
                                   average='weighted',
                                   labels=np.unique(rounded_preds))

        # calculate accuracy
        total_test_accuracy += accuracy_score(rounded_preds, y_true)

        # calculate precision
        total_test_precision += precision_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

        # calculate recall
        total_test_recall += recall_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

    # report final accuracy of validation run
    avg_accuracy = total_test_accuracy / len(dataloader)

    # report final f1 of validation run
    avg_test_f1 = total_test_f1 / len(dataloader)

    # report final f1 of validation run
    avg_precision = total_test_precision / len(dataloader)

    # report final f1 of validation run
    avg_recall = total_test_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test Accur.': avg_accuracy,
            'Test precision': avg_precision,
            'Test recall': avg_recall,
            'Test F1': avg_test_f1
        }
    )
    return None


# prepare tensor data sets
def prepare_dataset(padded_tokens, attention_masks, target):
    # prepare target into np array
    target = np.array(target.values, dtype=np.int64).reshape(-1, 1)
    # create tensor data sets
    tensor_df = TensorDataset(padded_tokens, attention_masks, torch.from_numpy(target))
    # 80% of df
    train_size = int(0.8 * len(df))
    # 20% of df
    val_size = len(df) - train_size
    # 50% of validation
    test_size = int(val_size - 0.5*val_size)
    # divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(tensor_df, [train_size, val_size])
    # divide validation by randomly selecting samples
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size])

    return train_dataset, val_dataset, test_dataset


# create tensor data sets
train_dataset, val_dataset, test_dataset = prepare_dataset(input_ids,
                                                           attention_masks,
                                                           df['target'])


# helper function to count target distribution inside tensor data sets
def target_count(tensor_dataset):
    # set empty count containers
    count0 = 0
    count1 = 0
    # set total container to turn into torch tensor
    total = []
    # for every item in the tensor data set
    for i in tensor_dataset:
        # if the target is equal to 0
        if i[2].item() == 0:
            count0 += 1
        # if the target is equal to 1
        elif i[2].item() == 1:
            count1 += 1
    total.append(count0)
    total.append(count1)
    return torch.tensor(total)


# prepare weighted sampling for imbalanced classification
def create_sampler(target_tensor, tensor_dataset):
    # generate class distributions [x, y]
    class_sample_count = target_count(tensor_dataset)
    # weight
    weight = 1. / class_sample_count.float()
    # produce weights for each observation in the data set
    samples_weight = torch.tensor([weight[t[2]] for t in tensor_dataset])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler


# create samplers for just the training set
train_sampler = create_sampler(target_count(train_dataset), train_dataset)


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load DistilBERT with a single a single linear classification layer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.config.__dict__['hidden_dropout_prob'] = 0.3

# instantiate model - attach to GPU
model.cuda();

# optimizer
optimizer = AdamW(model.parameters(),
                  lr=5.67886390082615e-06,
                  weight_decay=0.9
                )

# set number of epochs
epochs = 4

# create DataLoaders with samplers
train_dataloader = DataLoader(train_dataset,
                              batch_size=8,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=8,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                              batch_size=8,
                              shuffle=True)


# set LR scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# lets check class balance for each batch to see how the sampler is working
for i, (x, y, z) in enumerate(train_dataloader):
    if i in range(0, 5):
        print("batch index {}, 0/1: {}/{}".format(
            i, (z == 0).sum(), (z == 1).sum()))

# create gradient scaler for mixed precision
scaler = GradScaler()

# create training result storage
training_stats = []
valid_stats = []
best_valid_loss = float('inf')

# for each epoch
for epoch in range(epochs):
    # train
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'bert-model1.pt')  # torch save
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('./model_save/bert/')  # transformers save
        tokenizer.save_pretrained('./model_save/bert/')  # transformers save


# organize results
pd.set_option('precision', 3)
df_train_stats = pd.DataFrame(data=training_stats)
df_valid_stats = pd.DataFrame(data=valid_stats)
df_stats = pd.concat([df_train_stats, df_valid_stats], axis=1)
df_stats.insert(0, 'Epoch', range(1, len(df_stats)+1))
df_stats = df_stats.set_index('Epoch')
df_stats


# plot results
def plot_results(df):
    # styling from seaborn.
    sns.set(style='darkgrid')
    # uncrease the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # plot the learning curve.
    plt.plot(df_stats['Train Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Val Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(list(range(1, epochs+1)))
    return plt.show()


plot_results(df_stats)


# test the model
test_stats = []
model.load_state_dict(torch.load('bert-model1.pt'))
testing(model, test_dataloader)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats  # .867


#

# fine tune BERT
training_stats = []
valid_stats = []
epochs = 4

# create gradient scaler for mixed precision
scaler = GradScaler()
def objective(trial):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                num_labels=2)

    # instantiate model - attach to GPU
    model.cuda()

    hidden_dropout_prob = trial.suggest_float('hidden_dropout_prob', low=0.1, high=0.4, step=0.05)
    learning_rate = trial.suggest_loguniform('lr', 1e-7, 1e-5)
    weight_decay = trial.suggest_float('weight_decay', low=0.5, high=1, step=0.05)
    model.config.__dict__['hidden_dropout_prob'] = hidden_dropout_prob

    # Generate the model.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8,
                                  sampler=train_sampler,
                                  shuffle=False)

    valid_dataloader = DataLoader(val_dataset,
                                  batch_size=8,
                                  shuffle=True)

    # optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)

    # set LR scheduler
    total_steps = len(train_dataloader) * epochs
    global scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    global epoch
    for epoch in range(epochs):
        # set containers
        train_total_loss = 0
        total_train_f1 = 0

        # put model into traning mode
        model.train()

        # for each batch of training data...
        for batch in train_dataloader:
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda().long()

            optimizer.zero_grad()

            with autocast():
                loss, logits = model(input_ids=b_input_ids,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

                train_total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # validation
        model.eval()

        total_valid_loss = 0
        total_valid_f1 = 0

        # evaluate data for one epoch
        for batch in valid_dataloader:

            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda().long()

            with torch.no_grad():
                loss, logits = model(input_ids=b_input_ids,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            total_valid_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            y_true = b_labels.detach().cpu().numpy()

            rounded_preds = np.argmax(logits, axis=1).flatten()

            total_valid_f1 += f1_score(rounded_preds, y_true,
                                       average='weighted',
                                       labels=np.unique(rounded_preds))

        global avg_val_f1
        avg_val_f1 = total_valid_f1 / len(valid_dataloader)

        # calculate the average loss over all of the batches.
        global avg_val_loss
        avg_val_loss = total_valid_loss / len(valid_dataloader)

    trial.report(avg_val_loss, epoch)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_val_loss


study = optuna.create_study(direction="minimize",
                            pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                  max_resource=4,
                                                                  reduction_factor=3,
                                                                  ))
study.optimize(objective, n_trials=25)


pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=0.9,  # at most 90% of docs
                        max_features=10000,
                        use_idf=True,
                        ngram_range=(1, 3))


X_tfidf = tfidf.fit_transform(df['body'])


# split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X_tfidf, df['target'].values.astype(np.int64), random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, random_state=42)

# C set ex-post from optuna trials
model_LR = LogisticRegression(C=1, penalty='l2', class_weight='balanced')
model_LR.fit(X_train_full, y_train_full);
X_test.shape
test1 = np.zeros(2512,)


# generate results
predicted_y = model_LR.predict(test1)

# f1 score
logit_f1 = f1_score(y_test, test1, average='weighted')
print(logit_f1)
