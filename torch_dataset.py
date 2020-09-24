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
import re, os
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, DistilBertModel
import string
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)


# tell pytorch to use cuda
device = torch.device("cuda")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
df = pd.read_csv('C:\\Users\\Andrew\\Desktop\\test_export.csv')


# Create Dataset
class CSVDataset(Dataset):
    """Propganda dataset."""

    def __init__(self, csv_file, text_col, cat_cols, target, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            text_col (string): column containing the text for anaylsis.
            cat_cols (string): column(s) containing string categorical data.
            target (string): column containing the dependent variable.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # initialize
        self.data_frame = pd.read_csv(csv_file)
        self.categorical_features = cat_cols
        self.text_features = text_col
        self.target = target
        self.transform = transform

        # encode categorical variables
        label_encoders = {}
        for cat_col in self.categorical_features:
            label_encoders[cat_col] = LabelEncoder()
            self.data_frame[cat_col] = label_encoders[cat_col].fit_transform(self.data_frame[cat_col])

        # embedding info
        self.cat_dims = [int(self.data_frame[col].nunique()) for col in self.categorical_features]
        self.emb_dims = [(x, min(50, (x + 1) // 2)) for x in self.cat_dims]
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in self.emb_dims])

        # encode outcome
        self.data_frame[target] = LabelEncoder().fit_transform(self.data_frame[target])

        # get length of df
    def __len__(self):
        return len(self.data_frame)

        # get target
    def __get_target__(self):
        return self.data_frame.target

        # get df filtered by indices
    def __get_values__(self, indices):
        return self.data_frame.iloc[indices]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pull a sample of data
        text = self.data_frame.iloc[idx][self.text_features]
        cats = self.data_frame.iloc[idx][self.categorical_features]
        cats = torch.tensor(cats).long()
        target = self.data_frame.iloc[idx][self.target]

        # create embeddings
        self.embeddings = []
        for i, emb in enumerate(self.all_embeddings):
            self.embeddings.append(emb(cats[i]))
        self.embeddings = torch.cat(self.embeddings, 0)

        # hold sample in a dict
        sample = {'text': text,
                  'cats': self.embeddings,
                  'target': target,
                  'idx': torch.tensor(idx)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize_Transform():

    # retrieve sample and unpack it
    def __call__(self, sample):
        text, cats, target, idx = (sample['text']['body'],
                              sample['cats'],
                              sample['target'].values.astype(np.int64),
                              sample['idx'])

        # transform text to input ids and attn masks
        tokenizer_output = tokenizer.encode_plus(
                            text,  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=512,  # set max length
                            truncation=True,  # truncate longer messages
                            pad_to_max_length=True,  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )
        input_ids, attn_mask = tokenizer_output['input_ids'], tokenizer_output['attention_mask']

        # yield another dict
        return {'input_ids': input_ids,
                'attn_mask': attn_mask,
                'cats': cats,
                'target': torch.from_numpy(target),
                'idx': idx}

# instantiate the lazy data set
csv_dataset = CSVDataset(csv_file='C:\\Users\\Andrew\\Desktop\\test_export.csv',
                         text_col=['body'],
                         cat_cols=["sas_active", "peace_talks_active", "isisk_active", "administration"],
                         target=['target'],
                         transform=Tokenize_Transform())

# set train, valid, and test size
train_size = int(0.8 * len(csv_dataset))
valid_size = int(0.1 * len(csv_dataset))

# use random split to create three data sets; +1 for odd number of data
train_ds, valid_ds, test_ds = torch.utils.data.random_split(csv_dataset, [train_size, valid_size, valid_size+1])


# create custom transformer that concats the text and categorical embeddings
class DistillBERT_FE(torch.nn.Module):
    def __init__(self):
        super(DistillBERT_FE, self).__init__()
        # load model
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # pre-classifier layer
        self.pre_classifier = torch.nn.Linear(772, 772)  # 4 embed dim + 768
        # drop out
        self.dropout = torch.nn.Dropout(0.3)
        # final classification layer
        self.classifier = torch.nn.Linear(772, 2)  # 4 embed dim + 768

    def forward(self, input_ids, attention_mask):
        # generate outputs from BERT
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # last hidden layer
        pooled_output = hidden_state[:, 0]  # just the cls token

        # cat transformer embeddings with entity embeddings
        pooled_output = torch.cat([pooled_output, b_cats], dim=1)

        # send through pre-classifying linear layer
        pooled_output = self.pre_classifier(pooled_output)
        # relu
        pooled_output = torch.nn.ReLU()(pooled_output)
        # add dropout
        pooled_output = self.dropout(pooled_output)
        # final classifying layer to yield logits
        logits = self.classifier(pooled_output)

        return logits


# prepare weighted sampling for imbalanced classification
def create_sampler(train_ds, csv_dataset):
    # get indicies from train split
    train_indices = train_ds.indices
    # generate class distributions [y1, y2, etc...]
    bin_count = np.bincount(csv_dataset.__get_target__()[train_indices])
    # weight gen
    weight = 1. / bin_count.astype(np.float32)
    # produce weights for each observation in the data set
    samples_weight = torch.tensor([weight[t] for t in csv_dataset.__get_target__()[train_indices]])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler

# create sampler for the training ds
train_sampler = create_sampler(train_ds, csv_dataset)


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


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
        b_input_ids = batch['input_ids'].squeeze(1).cuda()
        b_input_mask = batch['attn_mask'].squeeze(1).cuda()
        global b_cats
        b_cats = batch['cats'].cuda()
        b_labels = batch['target'].cuda().long()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # loss
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
        b_input_ids = batch['input_ids'].squeeze(1).cuda()
        b_input_mask = batch['attn_mask'].squeeze(1).cuda()
        global b_cats
        b_cats = batch['cats'].cuda()
        b_labels = batch['target'].cuda().long()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # loss
            loss = criterion(logits.view(-1, 2), b_labels.view(-1))

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


# Load DistilBERT_FE
model = DistillBERT_FE().cuda()

# optimizer
optimizer = AdamW(model.parameters(),
                  lr=3.2696465645595003e-06,
                  weight_decay=1.0
                )

# set loss
criterion = nn.CrossEntropyLoss()


# set number of epochs
epochs = 5

# create DataLoaders with samplers
train_dataloader = DataLoader(train_ds,
                              batch_size=16,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(valid_ds,
                              batch_size=16,
                              shuffle=True)

test_dataloader = DataLoader(test_ds,
                              batch_size=16,
                              shuffle=True)


# set LR scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1*total_steps,
                                            num_training_steps=total_steps)


# lets check class balance for each batch to see how the sampler is working
for i, batch in enumerate(train_dataloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (batch['target'] == 0).sum(), (batch['target'] == 1).sum()))
    if i == 14:
        break

# lets have a look at a single batch of categorical embeddings
batch['cats']

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




# for error analysis
batch_idx = np.array(batch['idx'])
csv_dataset.__get_values__(batch_idx)

# with categorical embeddings - 5 epochs ~ 2:45 an epoch
# distilbert train: 0.265 loss, 0.897 f1
# distilbert valid: 0.280, 0.881 f1

# with warmup 0.1%
# distilbert train: 0.261 loss, 0.898 f1
# distilbert valid: 0.282, 0.880 f1









######## Try to tokenize the cats with the NLP embeddings

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
import re, os
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, DistilBertModel
import string
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)


# tell pytorch to use cuda
device = torch.device("cuda")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
df = pd.read_csv('C:\\Users\\Andrew\\Desktop\\test_export.csv')


# Create Dataset
class CSVDataset(Dataset):
    """Propganda dataset."""

    def __init__(self, csv_file, text_col, cat_cols, target, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.categorical_features = cat_cols
        self.text_features = text_col
        self.target = target
        self.transform = transform

        # encode outcome
        self.data_frame[target] = LabelEncoder().fit_transform(self.data_frame[target])

    def __len__(self):
        return len(self.data_frame)

    def __get_target__(self):
        return self.data_frame.target

    def __get_values__(self, indices):
        return self.data_frame.iloc[indices]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data_frame.iloc[idx][self.text_features]
        cats = self.data_frame.iloc[idx][self.categorical_features]
        target = self.data_frame.iloc[idx][self.target]


        sample = {'text': text, 'cats': cats.values, 'target': target, 'idx': torch.tensor(idx)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize_Transform():

    def __call__(self, sample):
        text, cats, target, idx = (sample['text']['body'],
                              sample['cats'],
                              sample['target'].values.astype(np.int64),
                              sample['idx'])

        # transform text to input ids and attn masks
        tokenizer_output = tokenizer.encode_plus(
                            text,  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=512,  # set max length
                            truncation=True,  # truncate longer messages
                            pad_to_max_length=True,  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )
        input_ids, attn_mask = tokenizer_output['input_ids'], tokenizer_output['attention_mask']

        return {'input_ids': input_ids,
                'attn_mask': attn_mask,
                'cats': cats,
                'target': target,
                'idx': idx}


class Tokenize_Cats():

    def __call__(self, sample):
        text, cats, target, idx = ((sample['input_ids'], sample['attn_mask']),
                              sample['cats'],
                              sample['target'],
                              sample['idx'])

        # transform text to input ids and attn masks
        cat_input_ids = []
        cat_attn_mask = []
        encoded_dict = tokenizer.encode_plus(
                                ' '.join(cats),                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 10,           # Pad & truncate all sentences.
                                truncation = True,
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
        cat_input_ids.append(encoded_dict['input_ids'])
        cat_attn_mask.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        cat_input_ids = torch.cat(cat_input_ids, dim=1)
        cat_attn_mask = torch.cat(cat_attn_mask, dim=1)

        return {'input_ids': text[0],
                'attn_mask': text[1],
                'cats_ids': cat_input_ids,
                'cats_mask': cat_attn_mask,
                'target': torch.from_numpy(target),
                'idx': idx}


csv_dataset = CSVDataset(csv_file='C:\\Users\\Andrew\\Desktop\\test_export.csv',
                         text_col=['body'],
                         cat_cols=["sas_active", "peace_talks_active", "isisk_active", "administration"],
                         target=['target'],
                         transform=transforms.Compose([Tokenize_Transform(), Tokenize_Cats()]))


train_size = int(0.8 * len(csv_dataset))
valid_size = int(0.1 * len(csv_dataset))

train_ds, valid_ds, test_ds = torch.utils.data.random_split(csv_dataset, [train_size, valid_size, valid_size+1])

# need to cat them
for index, batch in enumerate(train_ds):
    batch
    if index == 1:
        break

tokenizer.decode(batch['cats_ids'][0])

class DistillBERT_FE(torch.nn.Module):
    def __init__(self):
        super(DistillBERT_FE, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(1536, 1536)  # 4 embed dim + 768
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(1536, 2)  # 4 embed dim + 768

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        output_2 = self.l1(input_ids=cats_ids, attention_mask=cats_mask)
        hidden_state = output_1[0]  # last hidden layer
        pooled_output = hidden_state[:, 0]  # just the cls token

        hidden_state2 = output_2[0]  # last hidden layer
        pooled_output2 = hidden_state2[:, 0]  # just the cls token

        # cat transformer embeddings with entity embeddings
        pooled_output = torch.cat([pooled_output, pooled_output2], dim=1)

        # send through pre-classifying linear layer
        pooled_output = self.pre_classifier(pooled_output)
        # relu
        pooled_output = torch.nn.ReLU()(pooled_output)
        # add dropout
        pooled_output = self.dropout(pooled_output)
        # final classifying layer to yield logits
        logits = self.classifier(pooled_output)
        return logits


# prepare weighted sampling for imbalanced classification
def create_sampler(train_ds):
    # get indicies from train split
    train_indices = train_ds.indices
    # generate class distributions [x, y]
    bin_count = np.bincount(csv_dataset.__get_target__()[train_indices])
    # weight gen
    weight = 1. / bin_count.astype(np.float32)
    # produce weights for each observation in the data set
    samples_weight = torch.tensor([weight[t] for t in csv_dataset.__get_target__()[train_indices]])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler

# create sampler for the training ds
train_sampler = create_sampler(train_ds)

# set loss
criterion = nn.CrossEntropyLoss()

# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


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
        b_input_ids = batch['input_ids'].squeeze(1).cuda()
        b_input_mask = batch['attn_mask'].squeeze(1).cuda()
        global cats_ids, cats_mask
        cats_ids = batch['cats_ids'].squeeze(1).cuda()
        cats_mask = batch['cats_mask'].squeeze(1).cuda()
        b_labels = batch['target'].cuda().long()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # loss
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
        b_input_ids = batch['input_ids'].squeeze(1).cuda()
        b_input_mask = batch['attn_mask'].squeeze(1).cuda()
        global cats_ids, cats_mask
        cats_ids = batch['cats_ids'].squeeze(1).cuda()
        cats_mask = batch['cats_mask'].squeeze(1).cuda()
        b_labels = batch['target'].cuda().long()


        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # loss
            loss = criterion(logits.view(-1, 2), b_labels.view(-1))

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


# Load DistilBERT_FE
model = DistillBERT_FE().cuda()

# optimizer
optimizer = AdamW(model.parameters(),
                  lr=3.2696465645595003e-06,
                  weight_decay=1.0
                )


# set number of epochs
epochs = 5

# create DataLoaders with samplers
train_dataloader = DataLoader(train_ds,
                              batch_size=8,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(valid_ds,
                              batch_size=8,
                              shuffle=True)

test_dataloader = DataLoader(test_ds,
                              batch_size=8,
                              shuffle=True)


# set LR scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=total_steps*0.1,
                                            num_training_steps=total_steps)


# lets check class balance for each batch to see how the sampler is working
for i, batch in enumerate(train_dataloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (batch['target'] == 0).sum(), (batch['target'] == 1).sum()))
    if i == 14:
        break

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



# with NLP embeddings: ~ 3 mins an epoch
# distilbert train: 0.265 loss, 0.895 f1
# distilbert valid: 0.320, 0.851 f1




# comparison -- nn.Embedding - 5 epochs
# distilbert train: 0.265 loss, 0.897 f1
# distilbert valid: 0.280, 0.881 f1


# BERT -- nlp embeddings ~ 5 min an epoch
# bert train: 0.218 loss, 0.919 f1
# bert valid: 0.300, 0.859 f1
