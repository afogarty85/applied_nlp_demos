# HAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import time, datetime, random, re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from collections import Counter
import nltk
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from torch.cuda.amp import autocast, GradScaler


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
    # lower case the data
    df['body'] = df['body'].apply(lambda x: x.lower())
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
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



# prepare data for 3 dimensional transformation
# create containers for storage
text_to_sentences = []
full_text = []

# for each document
for idx in range(df.shape[0]):
    # retrieve the plain text
    text = df['body'].iloc[idx]
    # append it to the full_text container
    full_text.append(text)
    # use the nltk tokenizer to split document into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    # append them into the container
    text_to_sentences.append(sentences)

# calculate the average number of sentences and words per sentence
n_sent = 0
n_words = 0
for i in range(df.shape[0]):
    sentence = nltk.tokenize.sent_tokenize(df.loc[i, 'body'])
    for word in sentence:
        n_words += len(nltk.tokenize.word_tokenize(word))
    n_sent += len(sentence)

print("Average number of words in each sentence: ", round(n_words / n_sent))
print("Average number of sentences in each document: ", round(n_sent / df.shape[0]))

# since its the average, extend them slightly
MAX_SENT_LENGTH = 35
MAX_SENTS = 15

# instantiate a GloVe word map so we can tokenize based on GloVe IDs
word_map = dict(zip(embeddings_dictionary.keys(), range(len(embeddings_dictionary))))

# instantiate empty 3d data set
data_set_3d = np.zeros((len(full_text), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
data_set_3d.shape

# for every document
for i, sentences in enumerate(text_to_sentences):
    # for every split document into sentences
    for j, sent in enumerate(sentences):
        # if the step is less than max sentences
        if j < MAX_SENTS:
            # get the number of words in the sentence
            wordTokens = text_to_word_sequence(sent)
            # set k = 0
            k = 0
            # set container
            words_in_sent = []
            # for each word in the sentence
            for _, word in enumerate(wordTokens):
                # if k is less than max sentences
                if k < MAX_SENT_LENGTH:
                    # if we can find the word in GloVe
                    if word in word_map.keys():
                        # append word token to position document_i, sentence_j, position_k in sentence
                        data_set_3d[i, j, k] = word_map.get(word)
                        # add the word to list of words found in the sentence
                        words_in_sent.append(word)
                    else:
                        # other wise, add unknown token
                        data_set_3d[i, j, k] = 400000
                        # add 1 to k
                        # keep looping until max sentence length is achieved
                    k = k + 1

# change np.zeros to the unknown token, 400001
# otherwise it will calculate a bunch of 'the' vectors
data_set_3d[data_set_3d == 0] = 400001

# check data
data_set_3d[0][0]


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
train_dataset, val_dataset, test_dataset = prepare_dataset(data_set_3d, df['target'])


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

# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# create HAN
class HAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)


# create Sentence Level RNN
class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden, target_class)
        self.soft_sent = nn.Softmax(dim=1)

    def forward(self,x):
        sentence_h,_ = self.sentence_gru(x)
        x = torch.relu(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1,0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x


# create Word Level RNN
class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        pre_embed = config.pre_embed  # embeddings
        word_num_hidden = config.word_num_hidden
        words_num = config.vocab_size
        words_dim = config.words_dim
        self.mode = config.mode
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(vocab_size, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(pre_embed, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(pre_embed, freeze=False)
        else:
            print("Unsupported order")
            raise Exception
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax(dim=1)

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            raise Exception
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x


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
        b_input_ids = batch[0].cuda().long()
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
        b_input_ids = batch[0].cuda().long()
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
        b_input_ids = batch[0].cuda().long()
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


# HAN config class
class HANconfig():
    def __init__(self):
        HANconfig.mode = 'static'
        HANconfig.pre_embed = embeddings_tensor
        HANconfig.word_num_hidden = 120
        HANconfig.words_dim = 201
        HANconfig.sentence_num_hidden = 115
        HANconfig.vocab_size = len(vocab)
        HANconfig.target_class = 2
        return None


# instantiate HAN config
configHAN = HANconfig()

# instantiate model - attach to GPU
model = HAN(configHAN).cuda()

# set loss
criterion = nn.CrossEntropyLoss()

# set number of epochs
epochs = 5

# set optimizer
optimizer = AdamW(model.parameters(),
                  lr=0.0009515568738386746,
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
        torch.save(model.state_dict(), 'han-model1.pt')


# organize results
# Display floats with two decimal places.
pd.set_option('precision', 3)
df_train_stats = pd.DataFrame(data=training_stats)
df_valid_stats = pd.DataFrame(data=valid_stats)
df_stats = pd.concat([df_train_stats, df_valid_stats], axis=1)
df_stats.insert(0, 'Epoch', range(1, len(df_stats)+1))
df_stats = df_stats.set_index('Epoch')
df_stats


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
model.load_state_dict(torch.load('han-model1.pt'))
testing(model, test_dataloader, criterion)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats  #.813



# optuna -- tune hyperparameters
# create gradient scaler for mixed precision
scaler = GradScaler()

training_stats = []
valid_stats = []
epochs = 5
def objective(trial):

    # alter hyperparameters
    sent_num_hidden = trial.suggest_int('sentence_num_hidden', low=25, high=175, step=5)
    word_num_hidden = trial.suggest_int('word_num_hidden', low=25, high=175, step=5)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_float('weight_decay', low=0.5, high=1, step=0.05)
    configHAN = HANconfig()
    configHAN.sent_num_hidden = sent_num_hidden
    configHAN.word_num_hidden = word_num_hidden

    # data loaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=80,
                                  sampler=train_sampler,
                                  shuffle=False)

    valid_dataloader = DataLoader(val_dataset,
                                  batch_size=80,
                                  shuffle=True)

    # instantiate model
    model = HAN(configHAN).cuda()

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
            b_input_ids = batch[0].cuda().long()
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

            b_input_ids = batch[0].cuda().long()
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
                                                                  max_resource=7,
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



############################
