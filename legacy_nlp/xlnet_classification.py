import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import XLNetTokenizer, XLNetConfig, XLNetForSequenceClassification, AdamW
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import copy
pd.set_option('display.max_columns', 77)
pd.set_option('display.max_rows', 55)
pd.set_option('display.width', 0)
np.set_printoptions(suppress=True)

# prepare torch data set
class TorchDataSet(torch.utils.data.Dataset):
    '''
    This prepares the Dataset
    ----------
    transform : optionable, callable flag
        Whether or not we need to tokenize transform the data

    Returns
    -------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions

    '''
    def __init__(self, df_path, text_col, label_col, stratify=False, transform=None, truncate=None):

        # set init params
        self.df = pd.read_parquet(df_path)
        self.text_col = text_col
        self.label_col = label_col
        self.stratify = stratify
        self.transform = transform
        self.truncate = truncate

    # get len
    def __len__(self):
        if self.truncate:
            return len(self.df) // 1000
        return len(self.df)

    def __sss__(self, n_splits, test_size):
        # stratified split
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
        # get indices
        indices_dict = {}
        for i, (train, test) in enumerate(sss.split(self.df, self.df[self.label_col])):
            indices_dict[f'train_{i}'] = train
            indices_dict[f'test_{i}'] = test
        return indices_dict

    # pull a sample of data
    def __getitem__(self, idx):

        # get item
        text = self.df[self.text_col][idx]
        label = self.df[self.label_col][idx]

        # return sample
        sample = {'text': text,
                  'label': label,
                  'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


# tokenize text
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)



class Tokenize_Transform():
    '''
    This function tokenize transforms the data emitted from the data set.
    Parameters
    ----------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions
    Returns
    -------
    sample : dict
        A dictionary containing: (1) input tokens, (2) attention masks,
        (3) labels, and (4) data set index.
    '''
    def __init__(self, tokenizer):
        # instantiate the tokenizer
        self.tokenizer = tokenizer

    # retrieve sample and unpack it
    def __call__(self, sample):
        # transform text to input ids and attn masks
        encodings = self.tokenizer.encode_plus(
                            sample['text'],  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=128,  # set max length
                            truncation=True,  # truncate longer messages
                            padding='max_length',  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )

        # package up encodings
        return {'input_ids': encodings['input_ids'].squeeze(0),

                'token_type_ids': encodings['token_type_ids'].squeeze(0),

                'attn_masks': encodings['attention_mask'].squeeze(0),

                'labels': sample['label'],

                'idxs': sample['idx']
                                       }


# load data set
TorchDataSetDS = TorchDataSet(df_path=r'/mnt/c/Users/afogarty/Desktop/ML/SES/xlnet_training.parquet',
                                   text_col='concat',
                                   label_col='exit_code',
                                   transform=Tokenize_Transform(tokenizer=tokenizer),
                                   truncate=True)

len(TorchDataSetDS)

# view sample;
out = TorchDataSetDS.__getitem__(0)

# generate k-fold stratified indices
#index_dict = cottagelakeds.__sss__(n_splits=5, test_size=0.10)

# subset with indices
#sub_train = torch.utils.data.Subset(cottagelakeds, [1])
#sub_train.__getitem__(0)

# set train, valid, and test size
train_size = int(0.90 * len(TorchDataSetDS))
valid_size = int(0.10 * len(TorchDataSetDS)) + 1

# use random split to create two data sets;
train_set, val_set = torch.utils.data.random_split(TorchDataSetDS, [train_size, valid_size])

# init training info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 20
epochs = 4

# loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, collate_fn=None)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, collate_fn=None)


# load model
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_classes).to(device)

# optimizer
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-6
                  )


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(model, dataloader, optimizer, device):

    # tqdm stats
    pbar_total = len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Training", position=0, leave=True)

    # set model to train
    model.train()

    # info
    running_loss = 0.0
    running_corrects = 0
    tot_count = 0

    for data in dataloader:

        # send to device
        out = [v.to(device) for k, v in data.items()]

        # clear gradients
        optimizer.zero_grad()

        # forward
        outputs = model(out[0], token_type_ids=out[1], attention_mask=out[2], labels=out[3])

        # get loss/logits
        loss = outputs[0]
        logits = outputs[1]

        # logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        labels = out[3].cpu().numpy()

        # step acc
        step_acc = flat_accuracy(logits, labels)

        # statistics
        running_loss += loss.item() * out[0].size(0)
        running_corrects += step_acc
        tot_count += out[0].size(0)

        # backward
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()

    # epoch stats
    epoch_loss = running_loss / tot_count
    epoch_acc = running_corrects / tot_count

    # package
    train_results = {
        'loss': epoch_loss,
        'acc': epoch_acc
        }
    return train_results


def evaluate(model, dataloader, device):

    # tqdm stats
    pbar_total = len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluating", position=0, leave=True)

    # set model to train
    model.eval()

    # info
    running_loss = 0.0
    running_corrects = 0
    tot_count = 0

    for data in dataloader:
        with torch.no_grad():
            
            # send to device
            out = [v.to(device) for k, v in data.items()]

            # forward
            outputs = model(out[0], token_type_ids=out[1], attention_mask=out[2], labels=out[3])

            # get loss/logits
            loss = outputs[0]
            logits = outputs[1]

            # logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            labels = out[3].cpu().numpy()

            # step acc
            step_acc = flat_accuracy(logits, labels)

            # statistics
            running_loss += loss.item() * out[0].size(0)
            running_corrects += step_acc
            tot_count += out[0].size(0)

            pbar.update(1)
        pbar.close()

        # epoch stats
        epoch_loss = running_loss / tot_count
        epoch_acc = running_corrects / tot_count

    # package
    eval_results = {
        'loss': epoch_loss,
        'acc': epoch_acc
        }
    return eval_results


# train and evaluate loop
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
valid_interval = 1

for i, epoch in enumerate(range(1, epochs + 1)):
    train_results = train(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
    print(f"Epoch {epoch} | Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['acc']:.4f} ")

    if epoch % valid_interval == 0:
        eval_results = evaluate(model=model, dataloader=val_loader, device=device)
        print(f"Epoch {epoch} | Eval Loss: {eval_results['loss']:.4f} | Eval Acc: {eval_results['acc']:.4f} ")
        if eval_results['acc'] > best_acc:
            best_acc = eval_results['acc']
            best_model_wts = copy.deepcopy(model.state_dict())





#