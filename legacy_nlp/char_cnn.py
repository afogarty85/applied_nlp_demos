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
    df['body'] = df['body'].apply(lambda x: x.translate(translator))
    # generate a word count
    df['word_count'] = df['body'].apply(lambda x: len(x.split()))
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))

    return df


df = clean_df(df)


# get counts of each char -- necessary for vocab
counts = Counter(" ".join(df['body'].values.tolist()))

# build corpus vocab
vocab = sorted(counts, key=counts.get, reverse=True)

# provide the vocab indicies
vocab_to_int = {word: i for i, word in enumerate(counts, 1)}

# add padding
vocab_to_int['PAD'] = 0

# vocab size
vocab_size = len(vocab_to_int.keys())

# encode text
max_char_length = 1014

def encode(text):
    encoded = np.zeros([vocab_size, max_char_length], dtype='float32')
    review = text.lower()[:max_char_length-1:-1]
    i = 0
    for letter in text:
        if i >= max_char_length:
            break
        if letter in vocab_to_int:
            encoded[vocab_to_int[letter]][i] = 1
        i += 1
    return encoded


encoded_text = []
for doc in df['body'].values:
    encoded_text.append(encode(doc))

encoded_text = np.asarray(encoded_text, dtype=np.float32)
encoded_text.shape

encoded_text = encoded_text.reshape(len(df), max_char_length, vocab_size)
encoded_text.shape

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
train_dataset, val_dataset, test_dataset = prepare_dataset(encoded_text, df['target'])



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


class CharCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_conv_filters = config.num_conv_filters
        output_channel = config.output_channel
        num_affine_neurons = config.num_affine_neurons
        target_class = config.target_class
        input_channel = 38  # vocab size

        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, num_affine_neurons)
        self.fc2 = nn.Linear(num_affine_neurons, num_affine_neurons)
        self.fc3 = nn.Linear(num_affine_neurons, target_class)

    def forward(self, x, **kwargs):
        x = x.transpose(1, 2).type(torch.cuda.FloatTensor)

        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# instantiate model config -- set ex-post from optuna search
class config:
    def __init__(self):
        config.num_conv_filters = 256
        config.output_channel = 256
        config.num_affine_neurons = 1024
        config.target_class = 2
        config.dropout = 0.4
        return None



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


config1 = config()

model = CharCNN(config1).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 7

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
        torch.save(model.state_dict(), 'char-cnn-model1.pt')

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
model.load_state_dict(torch.load('char-cnn-model1.pt'))
testing(model, test_dataloader, criterion)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats  #.869



#
