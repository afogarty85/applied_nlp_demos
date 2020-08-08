# Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time, datetime, random, optuna, re, string
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer


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

# remove excess white spaces
df['body'] = df['body'].apply(lambda x: " ".join(x.split()))

# remove excess spaces near punctuation
df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))

# lower case the data
df['body'] = df['body'].apply(lambda x: x.lower())

# instantiate BERT tokenizer with upper + lower case
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# a look at some of the DistilBERT vocab
word_map = dict(zip(tokenizer.vocab.keys(), range(len(tokenizer))))
word_map.get('the')  # find index value
list(tokenizer.vocab.keys())[2000:2010]
len(tokenizer)


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


# create tokenized data
input_ids, attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)

# convert the labels into tensors.
labels = torch.tensor(df['target'].values.astype(np.float32))


# token 100: unknown [UNK]
# token 101: CLS for classification tasks [CLS]
# token 102: SEP at end of each sentence; each document in our case [SEP]
# token 0: padding [PAD]


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
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1])

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


# Build Kim Yoon CNN
class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel  # number of kernels
        num_classes = config.num_classes  # number of targets to predict
        dropout = config.dropout  # dropout value
        embedding_dim = config.embedding_dim  # length of embedding dim

        ks = 3  # three conv nets here

        # input_channel = word embeddings at a value of 1; 3 for RGB images
        input_channel = 4  # for single embedding, input_channel = 1

        # [3, 4, 5] = window height
        # padding = padding to account for height of search window

        # 3 convolutional nets
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, embedding_dim), padding=(2, 0), groups=4)
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, embedding_dim), padding=(3, 0), groups=4)
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, embedding_dim), padding=(4, 0), groups=4)

        # apply dropout
        self.dropout = nn.Dropout(dropout)

        # fully connected layer for classification
        # 3x conv nets * output channel
        self.fc1 = nn.Linear(ks * output_channel, num_classes)

    def forward(self, x, **kwargs):
        #x = x.unsqueeze(1)  # get another dimension at first index pos
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

    # put both models into traning mode
    model.train()
    kim_model.train()

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
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # Update the scheduler
        scheduler.step()

        # calculate preds
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

    # training time end
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1': avg_train_f1,
            'Train Time': training_time
        }
    )

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {training_time:}")

    #torch.cuda.empty_cache()

    return None


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1 = 0
    total_valid_recall = 0
    total_valid_precision = 0
    total_bert_valid_loss = 0

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_valid_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val Accur.': avg_accuracy,
            'Val precision': avg_precision,
            'Val recall': avg_recall,
            'Val F1': avg_val_f1,
            'Val Time': training_time
        }
    )


    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {training_time:}")

    return None


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # capture test time
    total_t0 = time.time()

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_test_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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

    # report final accuracy of test run
    avg_accuracy = total_test_accuracy / len(dataloader)

    # report final f1 of test run
    avg_test_f1 = total_test_f1 / len(dataloader)

    # report final f1 of test run
    avg_precision = total_test_precision / len(dataloader)

    # report final f1 of test run
    avg_recall = total_test_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    # capture end testing time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test Accur.': avg_accuracy,
            'Test precision': avg_precision,
            'Test recall': avg_recall,
            'Test F1': avg_test_f1,
            'Test Time': training_time
        }
    )
    # print result summaries
    print("")
    print("summary results")
    print("epoch | test loss | test f1 | test time")
    print(f"{epoch+1:5d} | {avg_test_loss:.5f} | {avg_test_f1:.5f} | {training_time:}")

    return None


class config:
    def __init__(self):
        config.num_classes = 2  # binary
        config.output_channel = 16  # number of kernels
        config.embedding_dim = 768  # embed dimension
        config.dropout = 0.4  # dropout value
        return None


# create config
config1 = config()

# instantiate BERT
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()

# instantiate CNN
kim_model = KimCNN(config1).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 4

# only train the last 4 layers; saves ~600mb of GPU mem and 30s of compute
BERT_parameters = []
allowed_layers = [11, 10, 9, 8]

for name, param in model.named_parameters():
    for layer_num in allowed_layers:
        layer_num = str(layer_num)
        if ".{}.".format(layer_num) in name:
            BERT_parameters.append(param)

# set optimizer
optimizer = AdamW([{'params': BERT_parameters, 'lr': 5.67886390082615e-06}], weight_decay=1.0)


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
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'bert-cnn-model1.pt')


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
model.load_state_dict(torch.load('bert-cnn-model1.pt'))
testing(model, test_dataloader)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats


###############



























##########
# Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time, datetime, random, optuna, re, string
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer


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

# remove excess white spaces
df['body'] = df['body'].apply(lambda x: " ".join(x.split()))

# remove excess spaces near punctuation
df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))

# lower case the data
df['body'] = df['body'].apply(lambda x: x.lower())

# instantiate BERT tokenizer with upper + lower case
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# a look at some of the DistilBERT vocab
word_map = dict(zip(tokenizer.vocab.keys(), range(len(tokenizer))))
word_map.get('the')  # find index value
list(tokenizer.vocab.keys())[2000:2010]
len(tokenizer)


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


# create tokenized data
input_ids, attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)

# convert the labels into tensors.
labels = torch.tensor(df['target'].values.astype(np.float32))


# token 100: unknown [UNK]
# token 101: CLS for classification tasks [CLS]
# token 102: SEP at end of each sentence; each document in our case [SEP]
# token 0: padding [PAD]


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
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1])

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


# Build Kim Yoon CNN
class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel  # number of kernels
        num_classes = config.num_classes  # number of targets to predict
        dropout = config.dropout  # dropout value
        embedding_dim = config.embedding_dim  # length of embedding dim

        ks = 3  # three conv nets here

        # input_channel = word embeddings at a value of 1; 3 for RGB images
        input_channel = 1  # for single embedding, input_channel = 1

        # [3, 4, 5] = window height
        # padding = padding to account for height of search window

        # 3 convolutional nets
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, embedding_dim), padding=(2, 0), groups=1)
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, embedding_dim), padding=(3, 0), groups=1)
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, embedding_dim), padding=(4, 0), groups=1)

        # apply dropout
        self.dropout = nn.Dropout(dropout)

        # fully connected layer for classification
        # 3x conv nets * output channel
        self.fc1 = nn.Linear(ks * output_channel, num_classes)

    def forward(self, x, **kwargs):
        #x = x.unsqueeze(1)  # get another dimension at first index pos
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

    # put both models into traning mode
    model.train()
    kim_model.train()

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
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

            hidden_layers = torch.sum(hidden_layers, dim=1)  # aggregate layers

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # Update the scheduler
        scheduler.step()

        # calculate preds
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

    # training time end
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1': avg_train_f1,
            'Train Time': training_time
        }
    )

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {training_time:}")

    #torch.cuda.empty_cache()

    return None


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1 = 0
    total_valid_recall = 0
    total_valid_precision = 0
    total_bert_valid_loss = 0

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

            hidden_layers = torch.sum(hidden_layers, dim=1)

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_valid_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val Accur.': avg_accuracy,
            'Val precision': avg_precision,
            'Val recall': avg_recall,
            'Val F1': avg_val_f1,
            'Val Time': training_time
        }
    )


    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {training_time:}")

    return None


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # capture test time
    total_t0 = time.time()

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

            hidden_layers = torch.sum(hidden_layers, dim=1)

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_test_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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

    # report final accuracy of test run
    avg_accuracy = total_test_accuracy / len(dataloader)

    # report final f1 of test run
    avg_test_f1 = total_test_f1 / len(dataloader)

    # report final f1 of test run
    avg_precision = total_test_precision / len(dataloader)

    # report final f1 of test run
    avg_recall = total_test_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    # capture end testing time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test Accur.': avg_accuracy,
            'Test precision': avg_precision,
            'Test recall': avg_recall,
            'Test F1': avg_test_f1,
            'Test Time': training_time
        }
    )
    # print result summaries
    print("")
    print("summary results")
    print("epoch | test loss | test f1 | test time")
    print(f"{epoch+1:5d} | {avg_test_loss:.5f} | {avg_test_f1:.5f} | {training_time:}")

    return None


class config:
    def __init__(self):
        config.num_classes = 2  # binary
        config.output_channel = 16  # number of kernels
        config.embedding_dim = 768  # embed dimension
        config.dropout = 0.4  # dropout value
        return None


# create config
config1 = config()

# instantiate BERT
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()

# instantiate CNN
kim_model = KimCNN(config1).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 4

# only train the last 4 layers; saves ~600mb of GPU mem and 30s of compute
BERT_parameters = []
allowed_layers = [11, 10, 9, 8]

for name, param in model.named_parameters():
    for layer_num in allowed_layers:
        layer_num = str(layer_num)
        if ".{}.".format(layer_num) in name:
            BERT_parameters.append(param)

# set optimizer
optimizer = AdamW([{'params': BERT_parameters, 'lr': 2e-5}], weight_decay=1.0)


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
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'bert-cnn-model1.pt')


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
model.load_state_dict(torch.load('bert-cnn-model1.pt'))
testing(model, test_dataloader)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats











##########





























##############


##########
# Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time, datetime, random, optuna, re, string
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer


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

# remove excess white spaces
df['body'] = df['body'].apply(lambda x: " ".join(x.split()))

# remove excess spaces near punctuation
df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))

# lower case the data
df['body'] = df['body'].apply(lambda x: x.lower())

# instantiate BERT tokenizer with upper + lower case
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# a look at some of the DistilBERT vocab
word_map = dict(zip(tokenizer.vocab.keys(), range(len(tokenizer))))
word_map.get('the')  # find index value
list(tokenizer.vocab.keys())[2000:2010]
len(tokenizer)


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


# create tokenized data
input_ids, attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)

# convert the labels into tensors.
labels = torch.tensor(df['target'].values.astype(np.float32))


# token 100: unknown [UNK]
# token 101: CLS for classification tasks [CLS]
# token 102: SEP at end of each sentence; each document in our case [SEP]
# token 0: padding [PAD]


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
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1])

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


# Build Kim Yoon CNN
class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel  # number of kernels
        num_classes = config.num_classes  # number of targets to predict
        dropout = config.dropout  # dropout value
        embedding_dim = config.embedding_dim  # length of embedding dim

        ks = 3  # three conv nets here

        # input_channel = word embeddings at a value of 1; 3 for RGB images
        input_channel = 1  # for single embedding, input_channel = 1

        # [3, 4, 5] = window height
        # padding = padding to account for height of search window

        # 3 convolutional nets
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, embedding_dim), padding=(2, 0), groups=1)
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, embedding_dim), padding=(3, 0), groups=1)
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, embedding_dim), padding=(4, 0), groups=1)

        # apply dropout
        self.dropout = nn.Dropout(dropout)

        # fully connected layer for classification
        # 3x conv nets * output channel
        self.fc1 = nn.Linear(ks * output_channel, num_classes)

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)  # get another dimension at first index pos
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

    # put both models into traning mode
    model.train()
    kim_model.train()

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
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # (batch_size, channel, max_sent_length, hidden_size)

            hidden_layers = hidden_layers[:, -4:]   # (batch_size, channel, max_sent_length, hidden_size)

            hidden_layers = torch.sum(hidden_layers, dim=1)   # (batch_size, max_sent_length, hidden_size)

            # just cls token
            hidden_layers = hidden_layers[:, 0, :]   # (batch_size, max_sent_length, hidden_size)

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # Update the scheduler
        scheduler.step()

        # calculate preds
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

    # training time end
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1': avg_train_f1,
            'Train Time': training_time
        }
    )

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {training_time:}")

    #torch.cuda.empty_cache()

    return None


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1 = 0
    total_valid_recall = 0
    total_valid_precision = 0
    total_bert_valid_loss = 0

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

            hidden_layers = torch.sum(hidden_layers, dim=1)

            # just cls token
            hidden_layers = hidden_layers[:, 0, :]   # (batch_size, max_sent_length, hidden_size)

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_valid_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val Accur.': avg_accuracy,
            'Val precision': avg_precision,
            'Val recall': avg_recall,
            'Val F1': avg_val_f1,
            'Val Time': training_time
        }
    )


    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {training_time:}")

    return None


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # capture test time
    total_t0 = time.time()

    # put both models in evaluation mode
    model.eval()
    kim_model.eval()

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
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            hidden_layers = outputs[2]  # get hidden layers

            hidden_layers = torch.stack(hidden_layers, dim=1)  # stack the layers

            hidden_layers = hidden_layers[:, -4:]  # get the last 4 layers

            hidden_layers = torch.sum(hidden_layers, dim=1)

            # just cls token
            hidden_layers = hidden_layers[:, 0, :]   # (batch_size, max_sent_length, hidden_size)

            hidden_layers = hidden_layers.unsqueeze(1)

        logits = kim_model(hidden_layers)

        loss = criterion(logits.view(-1, 2), b_labels.view(-1))

        # accumulate validation loss
        total_test_loss += loss.item()

        # calculate preds
        _, predicted = torch.max(logits, 1)

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

    # report final accuracy of test run
    avg_accuracy = total_test_accuracy / len(dataloader)

    # report final f1 of test run
    avg_test_f1 = total_test_f1 / len(dataloader)

    # report final f1 of test run
    avg_precision = total_test_precision / len(dataloader)

    # report final f1 of test run
    avg_recall = total_test_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    # capture end testing time
    training_time = format_time(time.time() - total_t0)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test Accur.': avg_accuracy,
            'Test precision': avg_precision,
            'Test recall': avg_recall,
            'Test F1': avg_test_f1,
            'Test Time': training_time
        }
    )
    # print result summaries
    print("")
    print("summary results")
    print("epoch | test loss | test f1 | test time")
    print(f"{epoch+1:5d} | {avg_test_loss:.5f} | {avg_test_f1:.5f} | {training_time:}")

    return None


class config:
    def __init__(self):
        config.num_classes = 2  # binary
        config.output_channel = 16  # number of kernels
        config.embedding_dim = 768  # embed dimension
        config.dropout = 0.4  # dropout value
        return None


# create config
config1 = config()

# instantiate BERT
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()

# instantiate CNN
kim_model = KimCNN(config1).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 4

# only train the last 4 layers; saves ~600mb of GPU mem and 30s of compute
BERT_parameters = []
allowed_layers = [11, 10, 9, 8]

for name, param in model.named_parameters():
    for layer_num in allowed_layers:
        layer_num = str(layer_num)
        if ".{}.".format(layer_num) in name:
            BERT_parameters.append(param)

# set optimizer
optimizer = AdamW([{'params': BERT_parameters, 'lr': 2e-5}], weight_decay=1.0)


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
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'bert-cnn-model1.pt')


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
model.load_state_dict(torch.load('bert-cnn-model1.pt'))
testing(model, test_dataloader)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats



#########
