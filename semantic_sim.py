# import packages
import sys
sys.path.append("C:/Users/Andrew/Desktop/Projects/Deep Learning/utils")
from tools import AverageMeter, ProgressBar
from radam import RAdam
import torch
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time, datetime

SEED = 15
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)

# set torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Create Dataset
class CSVDataset(Dataset):
    """Propganda dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # initialize
        self.data_frame = pd.read_csv(csv_file)
        # text col
        self.text_features = self.data_frame['body']
        # target col
        self.target = self.data_frame['target']
        # initialize the transform if specified
        self.transform = transform
        # encode outcome
        self.data_frame['target'] = LabelEncoder().fit_transform(self.data_frame['target'])

        # get length of df
    def __len__(self):
        return len(self.data_frame)

        # get target
    def __get_target__(self):
        return self.data_frame.target

        # get df filtered by indices
    def __get_values__(self, indices):
        return self.data_frame.iloc[indices]

        # pull a sample of data
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # hold sample in a dict
        sample = {'features': self.text_features.iloc[idx],
                  'target': self.target.iloc[idx],
                  'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize_Transform():
    # retrieve sample and unpack it
    def __call__(self, sample):
        text, target, idx = (sample['features'],
                              sample['target'],
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
        return {'features': torch.as_tensor(input_ids,
                                         dtype=torch.long,
                                         device=device).squeeze(1),
                'attn_mask': torch.as_tensor(attn_mask,
                                          dtype=torch.long,
                                          device=device).squeeze(1),
                'target': torch.as_tensor(target,
                                          dtype=torch.long,
                                          device=device),
                'idx': torch.as_tensor(idx,
                                       dtype=torch.int,
                                       device=device)}

# instantiate the lazy data set
csv_dataset = CSVDataset(csv_file='C:\\Users\\Andrew\\Desktop\\test_export.csv', transform=Tokenize_Transform())

# check data
for i, batch in enumerate(csv_dataset):
    if i == 0:
        break

# set train, valid, and test size
train_size = int(0.8 * len(csv_dataset))
valid_size = int(0.1 * len(csv_dataset))

# use random split to create three data sets; +1 for odd number of data
train_ds, valid_ds, test_ds = torch.utils.data.random_split(csv_dataset, [train_size, valid_size, valid_size+1])


# create distilbert
class DistillBERT(torch.nn.Module):
    def __init__(self):
        super(DistillBERT, self).__init__()
        # load model
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
        # pre-classifier layer
        self.pre_classifier = torch.nn.Linear(768, 768)
        # drop out
        self.dropout = torch.nn.Dropout(0.3)
        # final classification layer
        self.classifier = torch.nn.Linear(768, 2)  # [features, targets]

    def forward(self, input_ids, attention_mask):
        # generate outputs from BERT
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output_1[1]  # 2nd tuple is the embeddings
        # stack
        embeddings = torch.stack(embeddings)  # [layers, batch_sz, tokens, features]]
        # grab second to last layer
        embeddings = embeddings[-2]  # [batch_sz, tokens, features]]
        # just get the CLS token
        embeddings = embeddings[:, 0, :]  # [batch_sz, features]]
        # send through pre-classifying linear layer
        pooled_output = self.pre_classifier(embeddings)
        # relu
        pooled_output = torch.nn.ReLU()(pooled_output)
        # add dropout
        pooled_output = self.dropout(pooled_output)
        # final classifying layer to yield logits
        logits = self.classifier(pooled_output)
        return logits


# load the model
model = DistillBERT().to(device)

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

# create DataLoaders
train_dataloader = DataLoader(train_ds,
                              batch_size=16,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(valid_ds,
                              batch_size=16*2,
                              shuffle=True)

test_dataloader = DataLoader(test_ds,
                              batch_size=16*2,
                              shuffle=True)

# check sampler's integrity
for i, batch in enumerate(train_dataloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (batch['target'] == 0).sum(), (batch['target'] == 1).sum()))
    if i == 10:
        break

# create gradient scaler for mixed precision
scaler = GradScaler()

# create optimizer -- add L2 regularization
optimizer = RAdam(model.parameters(), lr=2e-6, weight_decay=0.5)

# set epochs
epochs = 4

# set LR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=2e-5,
                                            total_steps=len(train_dataloader)*epochs)

# train function
def train(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=batch['features'].squeeze(1), attention_mask=batch['attn_mask'].squeeze(1))
            loss = F.cross_entropy(logits.view(-1, 2), batch['target'].view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        pbar(step=batch_idx, info={'loss': loss.item()})
        train_loss.update(loss.item(), n=1)
    return {'loss': train_loss.avg}


# valid/test function
def test(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Testing')
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    valid_f1 = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            logits = model(input_ids=batch['features'].squeeze(1), attention_mask=batch['attn_mask'].squeeze(1))
            loss = F.cross_entropy(logits.view(-1, 2), batch['target'].view(-1))
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(batch['target'].view_as(pred)).sum().item()
            f1 = f1_score(pred.to("cpu").numpy(), batch['target'].to("cpu").numpy(), average='macro')
            valid_f1.update(f1, n=batch['features'].size(0))
            valid_loss.update(loss, n=batch['features'].size(0))
            valid_acc.update(correct, n=1)
            count += batch['features'].size(0)
            pbar(step=batch_idx)
    return {'valid_loss': valid_loss.avg,
            'valid_acc': valid_acc.sum /count,
            'valid_f1': valid_f1.avg}

# training
best_loss = 0
for epoch in range(1, epochs + 1):
    train_log = train(train_dataloader)
    valid_log = test(valid_dataloader)
    logs = dict(train_log, **valid_log)
    for key, value in logs.items():
        if key == 'valid_loss':
            if value.item() < best_loss:
                torch.save(model.state_dict(), 'distilbert-model1.pt')  # torch save
            best_loss = value.item()
    show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(show_info)

# testing
test_log = test(test_dataloader)
print(test_log)




#############
# export embeddings for semantic similary search
# create dataloader for whole data set -- batch size of 1 is fastest
train_dataloader = DataLoader(csv_dataset,
                              batch_size=1,
                              shuffle=False)


# tweak the model
class DistillBERT_Emb(torch.nn.Module):
    def __init__(self):
        super(DistillBERT_Emb, self).__init__()
        # load model
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
        # pre-classifier layer
        self.pre_classifier = torch.nn.Linear(768, 768)
        # drop out
        self.dropout = torch.nn.Dropout(0.3)
        # final classification layer
        self.classifier = torch.nn.Linear(768, 2)  # [features, targets]

    def forward(self, input_ids, attention_mask):
        # generate outputs from BERT
        with torch.no_grad():
            output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output_1[1]  # 2nd tuple is the embeddings
        # stack
        embeddings = torch.stack(embeddings)  # [layers, batch_sz, tokens, features]]
        # grab second to last layer
        embeddings = embeddings[-2]  # [batch_sz, tokens, features]]
        # just get the CLS token
        embeddings = embeddings[:, 0, :]  # [batch_sz, features]]
        return embeddings


# instantiate the model and load the weights
model = DistillBERT_Emb().to(device)
model.load_state_dict(torch.load('distilbert-model1.pt'))


# start time
t0 = time.time()
batch_num = 0
num_documents = len(csv_dataset)

# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# store embeddings
embeddings_container = []

print('Generating embeddings for all {:,} documents...'.format(len(csv_dataset)))
for step, batch in enumerate(train_dataloader):
    if step % 20 == 0 and not batch_num == 0:

        elapsed = format_time(time.time() - t0)

        # calc time remaining
        rows_per_sec = (time.time() - t0) / batch_num
        remaining_sec = rows_per_sec * (num_documents - batch_num)
        remaining = format_time(remaining_sec)

        # report progress
        print('Documents {:>7,} of {:>7,}. Elapsed: {:}. Remaining: {:}'.format(batch_num, num_documents, elapsed, remaining))

    # get embedding
    embeddings = model(input_ids=batch['features'].squeeze(1), attention_mask=batch['attn_mask'].squeeze(1))
    # send to cpu or GPU will run out of memory
    embeddings_container.append(embeddings.to('cpu').numpy())
    batch_num +=1
    torch.cuda.empty_cache()

embeddings_container = np.asarray(embeddings_container)
embeddings_container.shape
embeddings_container = embeddings_container.reshape(10045, 768)

# save document embeddings
np.save('semantic_search.npy', embeddings_container)



#
