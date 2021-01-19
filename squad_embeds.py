# packages
from argparse import ArgumentParser
import sys
sys.path.append("C:/utils/utils")
from tools import AverageMeter, ProgressBar, format_time
from compress_utils import AdapterPooler, BertConcat, TimeDistributed
from squad_preprocess import prepare_train_features, prepare_validation_features, postprocess_qa_predictions
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import random_split, DataLoader, RandomSampler, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler
import time, datetime, h5py
from datasets import load_dataset, load_metric
import collections


# prepare torch data set
class SquadDS(torch.utils.data.Dataset):
    '''
    This prepares the official Squad v2.0 JSON data
    (https://rajpurkar.github.io/SQuAD-explorer/) for traininig
    in PyTorch.

    Parameters
    ----------
    is_train : optionable, callable flag
        Whether or not we want Train or Dev data

    Returns
    -------
    train_sample : tuple
        A tuple containing tokenized inputs of: (1) input ids,
        (2) attention mask, (3) starting positions, and (4) ending positions.

    val_sample : tuple
        A tuple containing tokenized inputs of: (1) input ids,
        and (2) attention mask.
    '''
    def __init__(self, is_Train=True):
        self.is_Train = is_Train
        self.dataset = load_dataset("squad_v2")

        if self.is_Train:
            self.train = self.dataset['train'].map(prepare_train_features,
                                                   batched=True,
                                                   remove_columns=self.dataset["train"].column_names)

            self.train_mask = self.train['attention_mask']
            self.train_inputs = self.train['input_ids']
            self.start = self.train['start_positions']
            self.end = self.train['end_positions']
            self.token_type_ids = self.train['token_type_ids']

        else:
            self.val = self.dataset["validation"].map(prepare_validation_features,
                                                  batched=True,
                                                  remove_columns=self.dataset["validation"].column_names)

            self.val_inputs = self.val['input_ids']
            self.val_mask = self.val['attention_mask']
            self.val_id = self.val['example_id']
            self.val_offset = self.val['offset_mapping']

    # get len
    def __len__(self):
        if self.is_Train:
            return len(self.train)
        else:
            return len(self.val)

    # get items necessary for validation
    def __get_pred_train__(self):
        self.examples = self.dataset["train"]
        self.features = self.dataset["train"].map(
            prepare_validation_features,
            batched=True,
            remove_columns=self.dataset["train"].column_names
            )
        return self.examples, self.features

    # pull a sample of data
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_Train:
            sample = (self.train_inputs[idx],
                      self.train_mask[idx],
                      self.start[idx],
                      self.end[idx],
                      self.token_type_ids[idx])
            return sample

        else:
            sample = (self.val_inputs[idx],
                      self.val_mask[idx])

        return sample


def collate_train(batch):
    ''' This function retrieves the tuple emitted from
    SquadDS(is_Train=True) and packages them together for
    dataloader emission, example:
    batch['input_ids'] = [batch_sz, n_tokens]
    '''
    # containers
    input_ids = []
    attention_mask = []
    start = []
    end = []
    token_type_ids = []
    # loop over the batch
    for i in range(len(batch)):
        # get each item in the tuple
        input_ids.append(batch[i][0])
        attention_mask.append(batch[i][1])
        start.append(batch[i][2])
        end.append(batch[i][3])
        token_type_ids.append(batch[i][4])
    # package as a dict
    sample = {'input_ids': torch.as_tensor(input_ids),
              'attention_mask': torch.as_tensor(attention_mask),
              'start': torch.as_tensor(start),
              'end': torch.as_tensor(end),
              'token_type_ids': torch.as_tensor(token_type_ids)}
    return sample


def collate_val(batch):
    ''' This function retrieves the tuple emitted from
    SquadDS(is_Train=False) and packages them together for
    dataloader emission, example:
    batch['input_ids'] = [batch_sz, n_tokens]
    '''
    # containers
    input_ids = []
    attention_mask = []
    # loop over the batch
    for i in range(len(batch)):
        # get each item in the tuple
        input_ids.append(batch[i][0])
        attention_mask.append(batch[i][1])
    # package as dict
    sample = {'input_ids': torch.as_tensor(input_ids),
              'attention_mask': torch.as_tensor(attention_mask)}
    return sample


# train function
def train(model, dataloader, scaler, optimizer, device):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        # grab items from data loader and attach to GPU
        input_ids, attn_mask, start_pos, end_pos, token_type_ids = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['start'].to(device),
                                       batch['end'].to(device),
                                       batch['token_type_ids'].to(device))
        # clear gradients
        optimizer.zero_grad()
        # use mixed precision
        with autocast():
            # forward
            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        start_positions=start_pos,
                        end_positions=end_pos,
                        token_type_ids=token_type_ids)
        # backward
        scaler.scale(out[0]).backward()  # out[0] = loss
        scaler.step(optimizer)
        scaler.update()
        pbar(step=batch_idx, info={'loss': out[0].item()})
        train_loss.update(out[0].item(), n=1)
    train_log = {'train_loss': train_loss.avg}
    return train_log


# prepare embedding extraction
def emit_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = args.batch_size
    num_documents = len(train_dataset)
    # set a manipulable var to handle any batch size
    train_len = len(train_dataset)
    # check whether or not batch is divisible.
    if len(train_dataset) % args.batch_size != 0:
        remainder = len(train_dataset) % args.batch_size
        train_len = len(train_dataset) - remainder

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(train_len, 13, args.max_seq_length, 768),
                                maxshape=(None, 13, args.max_seq_length, 768),
                                chunks=(args.batch_size, 13, args.max_seq_length, 768),
                                dtype=np.float32)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_start_labels.h5', 'w') as s:
        # create empty data set; [batch_sz]
        start_dset = s.create_dataset('start_ids', shape=(train_len,),
                                      maxshape=(None,), chunks=(args.batch_size,),
                                      dtype=np.int64)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_end_labels.h5', 'w') as e:
        # create empty data set; [batch_sz]
        end_dset = e.create_dataset('end_ids', shape=(train_len,),
                                      maxshape=(None,), chunks=(args.batch_size,),
                                      dtype=np.int64)

    print('Generating embeddings for all {:,} documents...'.format(len(train_dataset)))
    for step, batch in enumerate(dataloader):
        # send necessary items to GPU
        input_ids, attn_mask, start_pos, end_pos, token_type_ids = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['start'].to(device),
                                       batch['end'].to(device),
                                       batch['token_type_ids'].to(device))

        if step % 20 == 0 and not batch_num == 0:
            # calc elapsed time
            elapsed = format_time(time.time() - t0)
            # calc time remaining
            rows_per_sec = (time.time() - t0) / batch_num
            remaining_sec = rows_per_sec * (num_documents - batch_num)
            remaining = format_time(remaining_sec)
            # report progress
            print('Documents {:>7,} of {:>7,}. Elapsed: {:}. Remaining: {:}'.format(batch_num, num_documents, elapsed, remaining))

        # get embeddings with no gradient calcs
        with torch.no_grad():
            # ['hidden_states'] is embeddings for all layers
            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        start_positions=start_pos,
                        end_positions=end_pos,
                        token_type_ids=token_type_ids)

        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32

        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_embeds.h5', 'a') as f:
            dset = f['embeds']
            # add chunk of rows
            start = step*args.batch_size
            # [batch_sz, layer, tokens, features]
            dset[start:start+args.batch_size, :, :, :] = embeddings[:, :, :, :]
            # Create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.batch_size

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_start_labels.h5', 'a') as s:
            start_dset = s['start_ids']
            # add chunk of rows
            start = step*args.batch_size
            # [batch_sz, layer, tokens, features]
            start_dset[start:start+args.batch_size] = start_pos.cpu().numpy()
            # Create attribute with last_index value
            start_dset.attrs['last_index'] = (step+1)*args.batch_size

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_end_labels.h5', 'a') as e:
            end_dset = e['end_ids']
            # add chunk of rows
            start = step*args.batch_size
            # [batch_sz, layer, tokens, features]
            end_dset[start:start+args.batch_size] = end_pos.cpu().numpy()
            # Create attribute with last_index value
            end_dset.attrs['last_index'] = (step+1)*args.batch_size

        batch_num += args.batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        # check the integrity of the embeddings
        x = f['embeds'][start:start+args.batch_size, :, :, :]
        assert np.array_equal(x, embeddings), 'not a match'
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    return None


def main():
    # training settings
    parser = ArgumentParser(description='SQuAD 2.0')
    parser.add_argument('--tokenizer', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--model', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--batch-size', type=int, default=14, metavar='N',
                         help='input batch size for training (default: 14)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                         help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                         help='learning rate (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                         help='number of CPU cores (default: 4)')
    parser.add_argument('--l2', type=float, default=1.0, metavar='LR',
                         help='l2 regularization weight (default: 1.0)')
    parser.add_argument('--max-seq-length', type=int, default=384, metavar='N',
                         help='max sequence length for encoding (default: 384)')
    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model)

    # set squad2.0 flag
    squad_v2 = True

    # set seeds and determinism
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.amp.autocast(enabled=True)

    # tokenize / load train
    train_ds = SquadDS(is_Train=True)

    # create train dataloader
    train_dataloader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=collate_train,
                                num_workers=args.num_workers,
                                drop_last=False)

    # load the model
    model = BertForQuestionAnswering.from_pretrained(args.model).to(device)

    # create gradient scaler for mixed precision
    scaler = GradScaler()

    # create optimizer
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.l2,
                    )

    # set epochs
    epochs = args.epochs

    # execute the model
    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        train_log = train(model, train_dataloader, scaler, optimizer, device)
        if train_log['train_loss'] < best_loss:
            # torch save
            torch.save(model.state_dict(), 'C:\\w266\\data2\\checkpoints\\BERT-QA' + '_epoch_{}.pt'.format(epoch))
            best_loss = train_log['train_loss']
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in train_log.items()])
        print(show_info)

    # now proceed to emit embeddings
    model = BertForQuestionAnswering.from_pretrained(args.model,
                                                     output_hidden_states=True).to(device)

    # load weights from 1 epoch
    model.load_state_dict(torch.load('C:\\w266\\data2\\checkpoints\\BERT-QA_epoch_1.pt'))

    # export embeddings
    emit_embeddings(train_dataloader, train_ds, model, device, args)

if __name__ == '__main__':
    main()

#
