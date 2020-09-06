# Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, datetime, random, re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel, BertTokenizer
import sys
sys.path.append("C:\\Users\\Andrew\\Desktop\\heinsen_routing")
import torchtext as tt
from heinsen_routing import Routing
from pytorch_extras import RAdam, SingleCycleScheduler



torch.manual_seed(44)
DEVICE = 'cuda:0'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lang_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
lang_model.cuda(device=DEVICE);
lang_model.eval();
print('BERT loaded.')


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

# my ver
def tokenized_texts_to_embs(tokenized_texts):
    tokenized_texts = [[*tok_seq] for tok_seq in tokenized_texts]
    lengths = [len(tok_seq) for tok_seq in tokenized_texts]

    max_length = max(lengths)
    input_toks = [t + [tokenizer.pad_token] * (max_length - l) for t, l in zip(tokenized_texts, lengths)]

    input_ids = [tokenizer.encode(tok_seq, add_special_tokens=True, truncation=True) for tok_seq in input_toks]
    input_ids = torch.tensor(input_ids).cuda()
    lengths = [len(tok_seq) for tok_seq in input_ids]
    max_length = max(lengths)

    mask = [[1.0] * length + [0.0] * (max_length - length) for length in lengths]
    mask = torch.tensor(mask).cuda() # [batch sz, num toks]

    with torch.no_grad():
        outputs = lang_model(input_ids=input_ids)
        embs = torch.stack(outputs[-1], -2)  # [batch sz, n toks, n layers, d emb]
        embs = embs[:, :, -4:, :]  # last 4 layers
    return mask, embs

df = df[['body', 'target']]
df.to_csv('working_csv.csv', index=False)

_stoi = {'Kabul': 0, 'US': 1}  # {'negative': 0, 'positive': 1}

TEXT = tt.data.RawField(
    preprocessing=tokenizer.tokenize,
    postprocessing=tokenized_texts_to_embs,
    is_target=False)

LABEL = tt.data.Field(sequential=False, use_vocab=False)  # use this if already numeric label

#LABEL = tt.data.RawField(
#    postprocessing=lambda samples: torch.tensor([_stoi[s] for s in samples], device=DEVICE),
#    is_target=True)
# use if not a numeric label

fields = [('body', TEXT), ('target', LABEL)]

# stratify split and pre/post proces the data to embeddings
raw_data = tt.data.TabularDataset('C:\\Users\\Andrew\\working_csv.csv', format='csv', fields=fields, skip_header=True)
trn_ds, val_ds, tst_ds = raw_data.split(split_ratio=[0.8, 0.1, 0.1], stratified=True, strata_field='target', random_state = random.seed(88))
print('Datasets ready.')
print('Number of samples: {:,} train phrases, {:,} valid sentences, {:,} test sentences.'\
      .format(len(trn_ds), len(val_ds), len(tst_ds)))

# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# sigmoid
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


class Classifier(nn.Module):
    """
    Args:
        d_depth: int, number of embeddings per token.
        d_emb: int, dimension of token embeddings.
        d_inp: int, number of features computed per embedding.
        d_cap: int, dimension 2 of output capsules.
        n_parts: int, number of parts detected.
        n_classes: int, number of classes.
    Input:
        mask: [..., n] tensor with 1.0 for tokens, 0.0 for padding.
        embs: [..., n, d_depth, d_emb] embeddings for n tokens.
    Output:
        a_out: [..., n_classes] class scores.
        mu_out: [..., n_classes, 1, d_cap] class capsules.
        sig2_out: [..., n_classes, 1, d_cap] class capsule variances.
    """
    def __init__(self, d_depth, d_emb, d_inp, d_cap, n_parts, n_classes):
        super().__init__()
        self.depth_emb = nn.Parameter(torch.zeros(d_depth, d_emb))
        self.detect_parts = nn.Sequential(nn.Linear(d_emb, d_inp), Swish(), nn.LayerNorm(d_inp))
        self.routings = nn.Sequential(
            Routing(d_cov=1, d_inp=d_inp, d_out=d_cap, n_out=n_parts),
            Routing(d_cov=1, d_inp=d_cap, d_out=d_cap, n_inp=n_parts, n_out=n_classes),
        )
        nn.init.kaiming_normal_(self.detect_parts[0].weight)
        nn.init.zeros_(self.detect_parts[0].bias)

    def forward(self, mask, embs):
        a = torch.log(mask / (1.0 - mask))                     # -inf to inf (logit)
        a = a.unsqueeze(-1).expand(-1, -1, embs.shape[-2])     # [bs, n, d_depth]
        a = a.contiguous().view(a.shape[0], -1)                # [bs, (n * d_depth)]

        mu = self.detect_parts(embs + self.depth_emb)          # [bs, n, d_depth, d_inp]
        mu = mu.view(mu.shape[0], -1, 1, mu.shape[-1])         # [bs, (n * d_depth), 1, d_inp]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)

        return a, mu, sig2



model = Classifier(d_depth=4, d_emb=768, d_inp=64, d_cap=2, n_parts=64, n_classes=2).cuda()
optimizer = RAdam(model.parameters(), lr=5e-4)
pct_warmup = 0.1
epochs = 5
n_iters = len(trn_ds) * epochs
scheduler = SingleCycleScheduler(
    optimizer, n_iters, frac=pct_warmup, min_lr=1e-5)

n_classes = 2
device = 'cuda:0'
mixup=(0.2, 0.2)
mixup_dist = torch.distributions.Beta(torch.tensor(mixup[0]), torch.tensor(mixup[1]))
onehot = torch.eye(n_classes, device=device)


# Make iterators for each split.
trn_itr, val_itr, tst_itr = tt.data.Iterator.splits(
    (trn_ds, val_ds, tst_ds),
    shuffle=True,
    sort=False,
    batch_size=16,
    device=DEVICE)

##########

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
    total_train_accuracy = 0

    # put model into eval mode
    model.eval()

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
        mask, embs, b_label = batch.body[0], batch.body[1], batch.target

        # clear previously calculated gradients
        optimizer.zero_grad()

        target_probs = onehot[b_label]
        # while training
        r = mixup_dist.sample([len(mask)]).to(device=device)
        idx = torch.randperm(len(mask))
        mask = mask.lerp(mask[idx], r[:, None])
        embs = embs.lerp(embs[idx], r[:, None, None, None])
        target_probs = target_probs.lerp(target_probs[idx], r[:, None])

        # preds
        pred_scores, _, _ = model(mask, embs)
        _, pred_ids = pred_scores.max(-1)
        accuracy = (pred_ids == b_label).float().mean()
        total_train_accuracy += accuracy.item()

        # for other metrics like f1
        predicted = pred_ids.detach().cpu().numpy()
        y_true = b_label.detach().cpu().numpy()
        total_train_f1 += f1_score(predicted, y_true, average='weighted', labels=np.unique(predicted))

        # loss
        losses = -target_probs * F.log_softmax(pred_scores, dim=-1)  # CE
        loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
        train_total_loss += loss.item()

        # back prop
        loss.backward()

        # optim updates
        optimizer.step()
        scheduler.step()

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # calculate the average f1 over all of the batches
    avg_train_f1 = total_train_f1 / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1': avg_train_f1,
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
        mask, embs, b_label = batch.body[0], batch.body[1], batch.target

        # clear previously calculated gradients
        optimizer.zero_grad()

        target_probs = onehot[b_label]
        # preds
        pred_scores, _, _ = model(mask, embs)
        _, pred_ids = pred_scores.max(-1)
        accuracy = (pred_ids == b_label).float().mean()
        total_valid_accuracy += accuracy.item()

        # for other metrics like f1
        predicted = pred_ids.detach().cpu().numpy()
        y_true = b_label.detach().cpu().numpy()
        total_valid_f1 += f1_score(predicted, y_true, average='weighted', labels=np.unique(predicted))

        # loss
        losses = -target_probs * F.log_softmax(pred_scores, dim=-1)  # CE
        loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
        total_valid_loss += loss.item()

        # back prop
        loss.backward()

        # optim updates
        optimizer.step()
        scheduler.step()

    # report final f1 of validation run
    global avg_val_f1
    avg_val_f1 = total_valid_f1 / len(dataloader)

    avg_val_acc = total_valid_accuracy / len(dataloader)

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val F1': avg_val_f1,
            'Val Acc': avg_val_acc
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val acc | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {avg_val_acc:.5f} | {training_time:}")

    return None


# create training result storage
training_stats = []
valid_stats = []
best_valid_loss = float('inf')

# this way does not erally learn
# for each epoch
for epoch in range(epochs):
    # train
    train(model, trn_itr, optimizer)
    # validate
    validating(model, val_itr)
    # check validation loss
    if valid_stats[epoch]['Val Loss'] < best_valid_loss:
        best_valid_loss = valid_stats[epoch]['Val Loss']
        # save best model for use later
        torch.save(model.state_dict(), 'capsule.pt')  # torch save

test_stats = []
model.load_state_dict(torch.load('capsule.pt'))
validating(model, tst_itr)



#
