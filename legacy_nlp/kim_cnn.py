import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time, datetime, re, random, string
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
from transformers import get_linear_schedule_with_warmup
from itertools import repeat
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, AdamW

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

# find min/max word count
max(df['word_count'])
min(df['word_count'])

# trim the corpus of really small messages
df = df.loc[df['word_count'] > 20]

# what is 95th percentile of word count?
percentile_95 = int(df['word_count'].quantile(0.95))
print(percentile_95)

# whats the length of the vocab?
counts = Counter(" ".join(df['body'].values.tolist()).split(" "))
vocab = sorted(counts, key=counts.get, reverse=True)
print(len(vocab))



# load GloVe embeddings
def load_GloVe(file_path):
    embeddings_dictionary = dict()
    glove_file = open(file_path, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary


# load GloVe
file_path = 'C:\\Users\\Andrew\\Desktop\\glove.6B.200d.txt'
embeddings_dictionary = load_GloVe(file_path)

# useful computing to check vector values and indices
embeddings_dictionary.get('.')  # get key value
list(embeddings_dictionary.keys()).index('.')  # get key index


# create vectors for "unknown" and "padding" and add to GloVe
def modify_GloVe(embeddings_dictionary):
    # create key values for unknown
    unknown_vector = np.random.uniform(-0.14, 0.14, 201)  # var of GloVe 200d
    # create key values for padding
    pad_vector = np.repeat(0, 200)
    pad_vector = np.append(pad_vector, 1)
    # turn dict into list to append easily
    embeddings_tensor = list(embeddings_dictionary.values())
    # extend GloVe dimension by 2
    embeddings_tensor = [np.append(i, 0) for i in embeddings_tensor]
    # add unknown and pad vectors via vstack
    embeddings_tensor = np.vstack([embeddings_tensor, unknown_vector])
    embeddings_tensor = np.vstack([embeddings_tensor, pad_vector])
    # finalize transform into tensor
    embeddings_tensor = torch.Tensor(embeddings_tensor)
    return embeddings_tensor

# modify GloVe and turn into torch tensor
embeddings_tensor = modify_GloVe(embeddings_dictionary)
# check shape
print(embeddings_tensor.shape)


# convert strings to GloVe familiar tokens
def text_to_GloVe_tokens(df, embeddings_dictionary):
    # create container for words that do not match
    no_matches = []
    # create container for tokenized strings
    glove_tokenized_data = []
    # create lookup for token ids
    word_map = dict(zip(embeddings_dictionary.keys(), range(len(embeddings_dictionary))))
    # for each document
    for doc in df['body']:
        # split each string
        doc = doc.split()
        # create token container
        tokens = []
        # for each word in the document
        for word in doc:
            # if word is a GloVE word
            if word in word_map:
                # get its GloVe index
                idx = word_map.get(word)
                # save its token
                tokens.append(idx)
            # otherwise
            else:
                # it must be an unknown word to GloVe
                idx = 400000  # unknown word
                # so append that word to no matches
                no_matches.append(word)
                # but also give it a vector lookup
                tokens.append(idx)
        # combine the tokens
        glove_tokenized_data.append(tokens)
    return no_matches, glove_tokenized_data


# get a list of no matches and our GloVe tokens
no_matches, glove_tokenized_data = text_to_GloVe_tokens(df, embeddings_dictionary)

# save words not found in GloVe; useful for other models later
np.save('translit_no_match.npy', no_matches)

# after removing rare words, how many words are we not accounting for now?
print(len(set(no_matches)))

# check original vs tokenized data
print('Original: ', df.iloc[0]['body'])
print('GloVe: ', glove_tokenized_data[0])


# post pad GloVe
def pad_GloVe(tokenized_data, max_len):
    padded_tokens = []
    max_len = max_len
    # for each tokenized document
    for tokenized_sent in tokenized_data:
        # if current doc length is greater than max length
        if len(tokenized_sent) > max_len:
            # trim it to max length
            current_sent = tokenized_sent[:max_len]
            # append
            padded_tokens.append(current_sent)

        # if current doc length is less than max length
        if len(tokenized_sent) < max_len:
            # find the difference in length
            extension = max_len - len(tokenized_sent)
            # pad sentences to max_len
            tokenized_sent.extend(repeat(400001, extension))
            # append new padded token
            padded_tokens.append(tokenized_sent)

        elif len(tokenized_sent) == max_len:
            padded_tokens.append(tokenized_sent)

    return np.array(padded_tokens, dtype=np.int64)


# get new padded tokens
padded_GloVe = pad_GloVe(glove_tokenized_data, percentile_95)
# check shape; 9994 documents, 974 length
print(padded_GloVe.shape)

# check to make sure padding done right
print([i for i, x in enumerate(padded_GloVe) if len(x) != percentile_95])


# prepare tensor data sets
def prepare_dataset(padded_tokens, target):
    # prepare target into np array
    target = np.array(target.values, dtype=np.int64).reshape(-1, 1)
    # create tensor data sets
    tensor_df = TensorDataset(torch.from_numpy(padded_tokens), torch.from_numpy(target))
    # 80% of df
    train_size = int(0.8 * len(df))
    # 20% of df
    val_size = len(df) - train_size
    # 50% of validation
    test_size = int(val_size - 0.5*val_size)
    # divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(tensor_df, [train_size, val_size])
    # divide validation by randomly selecting samples
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1])

    return train_dataset, val_dataset, test_dataset

# create tensor data sets
train_dataset, val_dataset, test_dataset = prepare_dataset(padded_GloVe, df['target'])


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
        if i[1].item() == 0:
            count0 += 1
        # if the target is equal to 1
        elif i[1].item() == 1:
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
    samples_weight = torch.tensor([weight[t[1]] for t in tensor_dataset])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler


# create samplers for each data set
train_sampler = create_sampler(target_count(train_dataset), train_dataset)


# create DataLoaders with samplers
train_dataloader = DataLoader(train_dataset,
                              batch_size=80,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=80,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                              batch_size=80,
                              shuffle=True)

# lets check class balance for each batch to see how the sampler is working
for i, (x, y) in enumerate(train_dataloader):
    if i in range(0, 10):
        print("batch index {}, 0/1: {}/{}".format(
            i, (y == 0).sum(), (y == 1).sum()))


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Build Kim Yoon CNN
class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel  # number of kernels
        num_classes = config.num_classes  # number of targets to predict
        vocab_size = config.vocab_size  # vocab size of corpus ** RESEARCH
        embedding_dim = config.embedding_dim  # GloVe embed dim size
        pre_embed = config.pre_embed  # GloVe coefs
        self.mode = config.mode  # static, or not
        ks = 3  # three conv nets here
        dropout = config.dropout  # dropout value
        padding = config.padding_idx  # padding indx value

        # for single embedding, input_channel = 1
        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(vocab_size, embedding_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)

        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(pre_embed,
                                                             freeze=True,
                                                             padding_idx=padding)

        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(pre_embed,
                                                                 freeze=False,
                                                                 padding_idx=padding)

        # input channel increases with trainable and untrainable embeddings
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(pre_embed,
                                                             freeze=True,
                                                             padding_idx=padding)
            self.non_static_embed = nn.Embedding.from_pretrained(pre_embed,
                                                                 freeze=False,
                                                                 padding_idx=padding)
            input_channel = 2

        else:
            print("Unsupported Mode")
            raise Exception

        # input_channel = word embeddings at a value of 1; 3 for RGB images
        # output_channel = number of kernels
        # [3, 4, 5] = window height
        # embedding_dim = length of embedding dim; my GloVe is 202
        # padding = padding to account for height of search window
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, embedding_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, embedding_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, embedding_dim), padding=(4, 0))
        # apply dropout
        self.dropout = nn.Dropout(dropout)
        # fully connected layer for classification
        # 3x conv nets * output channel
        self.fc1 = nn.Linear(ks * output_channel, num_classes)

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            word_input = self.embed(x)  # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)

        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)

        elif self.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)

        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)

        else:
            print("Unsupported Mode")
            raise Exception

        # squeeze to get size; (batch, channel_output, ~=sent_len) * ks
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # max-over-time pooling; # (batch, channel_output) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # concat results; (batch, channel_output * ks)
        x = torch.cat(x, 1)
        # add dropout
        x = self.dropout(x)
        # generate logits (batch, target_size)
        logit = self.fc1(x)
        return logit


def train(model, dataloader, optimizer, criterion):

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
        # As we unpack the batch, we'll also copy each tensor to the GPU
        #
        # `batch` contains two pytorch tensors:
        #   [0]: input ids
        #   [1]: labels
        b_input_ids = batch[0].cuda()
        b_labels = batch[1].cuda().long()

        # clear previously calculated gradients
        optimizer.zero_grad()

        with autocast():
            # forward propagation (evaluate model on training batch)
            logits = model(b_input_ids)

        # calculate cross entropy loss
        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

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

        # get preds
        _, predicted = torch.max(logits, 1)

        # move logits and labels to CPU
        predicted = predicted.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate f1
        total_train_f1 += f1_score(predicted, y_true,
                                   average='weighted',
                                   labels=np.unique(predicted))

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

    torch.cuda.empty_cache()

    return None


def validating(model, dataloader, criterion):

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

        # unpack batch from dataloader
        b_input_ids = batch[0].cuda()
        b_labels = batch[1].cuda().long()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            logits = model(b_input_ids)

            # calculate BCEWithLogitsLoss
            loss = criterion(logits.view(-1, 2), b_labels.view(-1))

            # calculate preds
            _, predicted = torch.max(logits, 1)

        # accumulate validation loss
        total_valid_loss += loss.item()

        # move logits and labels to CPU
        predicted = predicted.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate f1
        total_valid_f1 += f1_score(predicted, y_true,
                                   average='weighted',
                                   labels=np.unique(predicted))

        # calculate accuracy
        total_valid_accuracy += accuracy_score(predicted, y_true)

        # calculate precision
        total_valid_precision += precision_score(predicted, y_true,
                                                 average='weighted',
                                                 labels=np.unique(predicted))

        # calculate recall
        total_valid_recall += recall_score(predicted, y_true,
                                                 average='weighted',
                                                 labels=np.unique(predicted))

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


def testing(model, dataloader, criterion):

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
    for step, batch in enumerate(dataloader):
        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # unpack batch from dataloader
        b_input_ids = batch[0].cuda()
        b_labels = batch[1].cuda().long()

        # tell pytorch not to bother calculating gradients
        # only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            logits = model(b_input_ids)

            # calculate cross entropy loss
            loss = criterion(logits.view(-1, 2), b_labels.view(-1))

            # calculate preds
            _, predicted = torch.max(logits, 1)

            # accumulate validation loss
            total_test_loss += loss.item()

        # move logits and labels to CPU
        predicted = predicted.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate f1
        total_test_f1 += f1_score(predicted, y_true,
                                   average='weighted',
                                   labels=np.unique(predicted))

        # calculate accuracy
        total_test_accuracy += accuracy_score(predicted, y_true)

        # calculate precision
        total_test_precision += precision_score(predicted, y_true,
                                                 average='weighted',
                                                 labels=np.unique(predicted))

        # calculate recall
        total_test_recall += recall_score(predicted, y_true,
                                                 average='weighted',
                                                 labels=np.unique(predicted))

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


# instantiate model config -- set ex-post from optuna search
class config:
    def __init__(self):
        config.pre_embed = embeddings_tensor  # GloVe vectors
        config.mode = 'static'  # dont train embedding
        config.num_classes = 2  # num classes
        config.output_channel = 300  # number of kernels
        config.embedding_dim = 201  # GloVe embed dimension (202)
        config.vocab_size = len(vocab)+2  # vocab size of corpus
        config.dropout = 0.4  # dropout value
        config.padding_idx = 400001  # padding token index
        return None


# create config
config1 = config()

# instantiate model - attach to GPU
model = KimCNN(config1).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 5

# set optimizer
optimizer = AdamW(model.parameters(),
                  lr=0.0009978734977728082,
                  weight_decay=0.5
                )

# set LR scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# create gradient scaler for mixed precision
scaler = GradScaler()

# create training result storage
training_stats = []
valid_stats = []
best_valid_loss = float('inf')

# for each epoch
for epoch in range(epochs):
    # train
    train(model, train_dataloader, optimizer, criterion)
    # validate
    validating(model, valid_dataloader, criterion)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'cnn-model1.pt')

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
model.load_state_dict(torch.load('cnn-model1.pt'))
testing(model, test_dataloader, criterion)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats  #.867


# inference
def infer_class(model, string_df, min_len=percentile_95):
    # tokenize the string into GloVe
    no_matches, glove_tokenized_data = text_to_GloVe_tokens(string_df, embeddings_dictionary)
    # check length
    if len(glove_tokenized_data[0]) < min_len:
        glove_tokenized_data = pad_GloVe(glove_tokenized_data, percentile_95)
    tensor = torch.LongTensor(glove_tokenized_data[0]).cuda().unsqueeze(0)
    logits = model(tensor)
    _, predicted = torch.max(logits, 1)
    return print(predicted.item())


str_US = ['You will no longer be sent to an eternal war, President Donald Trump told a gathering marking the graduation of the US military academy. You will no longer fight in countries whose names are not known to most Americans. You will no longer be involved in the wars of the ancient nations. Clearly, Trump is referring to the long-running war in Afghanistan. The United States has been fighting the for the past two decades; it suffered a high number of casualties along with enormous financial losses. Many times, senior US officials have acknowledged that they cannot win the war in Afghanistan. Trump has told the truth, that his troops will no longer be involved in the wars of the ancient nations, because the never-ending war in Afghanistan has severely damaged America is reputation on the international level and caused the country extreme economic hardships. Recent surveys show that the support in the United States for this lost war has plummeted, and the people have now realized that the American leaders made false promises about this war. Nearly 20 years ago, the founder of the Islamic Emirate, the late Amir al- Mu aminin Mullah Mohammad Omar, warned the Americans to give up the intention of occupying Afghanistan. The Afghan soil has never been digested by any invader. The Americans should not roll up their sleeves to invade Afghanistan, otherwise, they will get involved in a war from which there is no way out. The prediction of the late Mullah Omar came true and America is engaged in an endless war. Trump must fulfill his promises and withdraw all his troops from Afghanistan as soon as possible and let the Afghans decide their own future. The agreement reached in Doha between the Islamic Emirate and the United States in February is the best and shortest way out of the current complex situation in Afghanistan. The full implementation of this agreement is beneficial for both the Afghan and American nations. The Afghan nation will regain its freedom and independence and build a powerful Islamic system on its own. The American nation will be saved from further self-harm and financial loss, and the deterioration of its reputation on the international level.']

str_AF = ['On 15 June, the soldiers of the puppet regime came to carry out operations in Sheikhano area , Tagab District, Kapisa Province. The mujahideen retaliated severely: 17 offensive soldiers of the puppet were killed in the operation; many of their corpses are lying on the battlefield; and many others were wounded. Meanwhile, the mujahideen have confiscated 16 6 -mm guns, one M 49 submachine gun, one rocket, two grenade launchers, a large amount of ammunition, and military equipment. They should know that the mujahideen would not give the enemy an opportunity to carry out operations, and they would be faced with severe counterattacks.']


temp_df_US = pd.DataFrame({'body': str_US})
temp_df_US = clean_df(temp_df_US)
infer_class(model, temp_df_US)  # 1 = US = Correct

temp_df_AF = pd.DataFrame({'body': str_AF})
temp_df_AF = clean_df(temp_df_AF)
infer_class(model, temp_df_AF)  # 0 = Kabul = Correct


# optuna -- tune hyperparameters
# create gradient scaler for mixed precision
scaler = GradScaler()

training_stats = []
valid_stats = []
epochs = 5
def objective(trial):

    # alter hyperparameters
    kernel_num = trial.suggest_int('output_channel', low=300, high=1500, step=50)
    dropout_num = trial.suggest_float('dropout', low=0.1, high=0.5, step=0.05)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_float('weight_decay', low=0.5, high=1, step=0.05)
    config1 = config()
    config1.output_channel = kernel_num
    config1.dropout = dropout_num

    # data loaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=80,
                                  sampler=train_sampler,
                                  shuffle=False)

    valid_dataloader = DataLoader(val_dataset,
                                  batch_size=80,
                                  shuffle=True)

    # instantiate model
    model = KimCNN(config1).cuda()
    # set optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay
                    )

    criterion = nn.BCEWithLogitsLoss()

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
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].cuda()
            b_labels = batch[1].cuda().type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            with autocast():
                logits = model(b_input_ids)
                loss = criterion(logits, b_labels)

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
            b_labels = batch[1].cuda().type(torch.cuda.FloatTensor)

            with torch.no_grad():
                logits = model(b_input_ids)
                loss = criterion(logits, b_labels)

            total_valid_loss += loss.item()

        # generate predictions
        rounded_preds = torch.round(torch.sigmoid(logits))

        # move logits and labels to CPU
        rounded_preds = rounded_preds.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate f1
        total_valid_f1 += f1_score(rounded_preds, y_true,
                                   average='weighted',
                                   labels=np.unique(rounded_preds))

        avg_val_f1 = total_valid_f1 / len(valid_dataloader)

        avg_val_loss = total_valid_loss / len(valid_dataloader)

    trial.report(avg_val_loss, epoch)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_val_loss



study = optuna.create_study(direction="minimize",
                            pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                  max_resource=5,
                                                                  reduction_factor=3,
                                                                  ))
study.optimize(objective, n_trials=35)


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
# put the model in evaluation mode
model.eval()

# track variables
total_test_accuracy = 0
total_test_loss = 0
total_test_f1 = 0
total_test_recall = 0
total_test_precision = 0
preds = []
true = []
criterion2 = nn.CrossEntropyLoss()
# evaluate data for one epoch
for step, batch in enumerate(test_dataloader):
    # progress update every 40 batches.
    if step % 40 == 0 and not step == 0:

        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # unpack batch from dataloader
    b_input_ids = batch[0].cuda()
    b_labels = batch[1].cuda().long()

    # tell pytorch not to bother calculating gradients
    # only necessary for training
    with torch.no_grad():

        # forward propagation (evaluate model on training batch)
        logits = model(b_input_ids)

        _, predicted = torch.max(logits, 1)

        # calculate cross entropy loss
        loss = criterion2(logits.view(-1, 2), b_labels.view(-1))
        # accumulate validation loss
        total_test_loss += loss.item()

    # move logits and labels to CPU
    preds.append(predicted.cpu().numpy())
    true.append(b_labels.cpu().numpy())

preds   = np.concatenate(preds)
true = np.concatenate(true)

f1_score(preds, true, average='weighted', labels=np.unique(preds))
predicted

preds   = np.concatenate(preds)

#
