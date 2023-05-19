import pytreebank, time, datetime, sys
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup, AdamW
from sklearn import metrics
import torch
import torch.nn as nn
from loguru import logger


SEED = 15
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)
logger.add("my_log_file.log", rotation="10 MB")


# gen utils
class AverageMeter(object):
    '''
    # computes and stores the average and current value
    # Example:
    #     >>> loss = AverageMeter()
    #     >>> for step,batch in enumerate(train_data):
    #     >>>     pred = self.model(batch)
    #     >>>     raw_loss = self.metrics(pred,target)
    #     >>>     loss.update(raw_loss.item(),n = 1)
    #     >>> cur_loss = loss.avg
    # '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


# tokenizer and GPU init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def get_binary_label(label):
    '''
    Convert fine-grained label to binary label

    labeled from: 0 -> very negative
                  2 -> neural
                  4 -> very positive

    '''
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError('Invalid label')

class SSTDataset(Dataset):
    """ configurable SST Dataset
        configuration:
            - split: train/val/test
            - nodes: root/all
            - binary/fine-grained
    """
    def __init__(self, split='train', root=True, binary=True):
        """Initializes the dataset with given configuration.
            - split: str
                Dataset split, [train, dev, test]
            - root: bool
                if true, only use root nodes; else, use all nodes
            - binary: bool
                if true, use binary labels; else, use fine-grained
        """
        logger.info(f"Loading SST {split} set")
        self.ds = pytreebank.load_sst()
        self.sst = self.ds[split]

        logger.info("Tokenizing")
        if root and binary:
            self.data = [
                (tokenizer(text=tree.to_lines()[0],
                                  add_special_tokens=True,
                                  padding='max_length',
                                  max_length=64,
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_token_type_ids=True,
                                  return_tensors='pt'),
                 get_binary_label(tree.label))
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            self.data = [
                (tokenizer(text=tree.to_lines()[0],
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=64,
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=True,
                                          return_tensors='pt'
                                          ),
                    tree.label)
                for tree in self.sst
                ]

        elif not root and not binary:
            self.data = [
                (tokenizer(text=tree.to_lines()[0],
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=64,
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=True,
                                          return_tensors='pt'
                                          ), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (tokenizer(text=tree.to_lines()[0],
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=64,
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=True,
                                          return_tensors='pt'
                                          ),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input_ids': self.data[idx][0]['input_ids'],
                  'token_type_ids': self.data[idx][0]['token_type_ids'],
                  'attention_mask': self.data[idx][0]['attention_mask'],
                  'label': self.data[idx][1]
                  }
        return sample


# initialize train and dev ds
train_ds = SSTDataset(split='train', root=True, binary=True)
dev_ds = SSTDataset(split='dev', root=False, binary=True)


# collate fn for SST
def collate_SST(batch):
    ''' This function packages the tokens and squeezes out the extra
    dimension.
    '''
    # turn data to tensors
    input_ids = torch.stack([torch.as_tensor(item['input_ids']) for item in batch]).squeeze(1)
    # get attn_mask
    attention_mask = torch.stack([torch.as_tensor(item['attention_mask']) for item in batch]).squeeze(1)
    # get token_type_ids
    token_type_ids = torch.stack([torch.as_tensor(item['token_type_ids']) for item in batch]).squeeze(1)
    # get labels
    labels = torch.stack([torch.as_tensor(item['label']) for item in batch])
    # repackage
    sample = {'input_ids': input_ids,
              'attention_masks': attention_mask,
              'token_type_ids': token_type_ids,
              'labels': labels}
    return sample


# create train and dev dataloaders
train_dataloader = DataLoader(train_ds,
                              batch_size=16,
                              shuffle=True,
                              num_workers=0,
                              drop_last=False,
                              collate_fn=collate_SST)

dev_dataloader = DataLoader(dev_ds,
                            batch_size=16*2,  # no grad; can 2x the speed
                            shuffle=True,
                            num_workers=0,
                            drop_last=False,
                            collate_fn=collate_SST)



class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * x.sigmoid()

d_features = 768
d_tokens = 64
d_inp = int(d_tokens / 2)
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """
    d_tokens       # number of tokens
    d_features     # dimension of token embeddings
    d_inp          # number of compressed features computed
    """
    def __init__(self, d_tokens, d_features, d_inp):
        super().__init__()
        ### ENCODER
        self.encode_params = nn.Parameter(torch.zeros(d_tokens, d_features))  # [tokens, BERT-features]
        self.encode = nn.Sequential(nn.Linear(d_features, d_inp), Swish(), nn.LayerNorm(d_inp))  # [BERT-features, compressed-features]
        nn.init.xavier_normal_(self.encode[0].weight)
        nn.init.zeros_(self.encode[0].bias)

        ### DECODER
        self.decode_params = nn.Parameter(torch.zeros(d_tokens, d_inp))  # [tokens, compressed-features]
        self.decode = nn.Sequential(nn.Linear(d_inp, d_features), Swish(), nn.LayerNorm(d_features))  # [tokens, BERT-features]

        ### CLASSIFICATION LAYER
        self.classification_layer = torch.nn.Linear(d_tokens*d_inp, 2)  # num_labels=2

        # bert emits ([layer, bs, tokens, features])
        return None

    def encoder(self, x):
        x = self.encode(x + self.encode_params)  # [layers=1, bs, tokens, features]
        x = x.squeeze(0)  # [bs, tokens, features=32]
        return x

    def decoder(self, x):
        x = self.decoder(x + self.decode_params)

    def forward(self, x):
        ### ENCODER
        x = self.encoder(x)

        ### CLASSIFIER
        x = x.view(-1, d_tokens*d_inp)
        x = self.classification_layer(x)
        return x

    def loss_fn(self, x, labels):
        loss = F.cross_entropy(x, labels)
        return loss


x = torch.randn(1, 16, 64, 768)

AE1 = AutoEncoder(d_tokens=d_tokens, d_features=d_features, d_inp=d_inp)
out = AE1(x)
out.shape


###

for batch_idx, batch in enumerate(train_dataloader):
    input_ids, token_type_ids, attn_masks, labels = (
        batch['input_ids'].to(device),
        batch['token_type_ids'].to(device),
        batch['attention_masks'].to(device),
        batch['labels'].to(device)
        )
    if batch_idx == 0: break


model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attn_masks)
out['last_hidden_state'].shape
out['pooler_output'].shape  # cls token
out['hidden_states'].shape

for i in range(13):
    print(out['hidden_states'][i].shape)  # torch.Size([16, 64, 768])



# initialize BERT
class BERT1(nn.Module):
    def __init__(self, n_labels):
        super(BERT1, self).__init__()
        self.n_labels = n_labels
        # load model
        self.l1 = BertModel.from_pretrained("bert-base-uncased",
                                            output_hidden_states=True)
        # freeze bert
        for param in self.l1.parameters():
            param.requires_grad = False

        # activiation
        self.GELU = torch.nn.GELU()
        # drop out
        self.dropout = torch.nn.Dropout(0.3)
        # initialize DP
        self.DP = DetectParts(d_depth=d_depth, d_emb=d_emb, d_inp=d_inp)
        # create linear layer
        self.classification = torch.nn.Linear(d_inp*n_tokens*d_depth, self.n_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # generate outputs from BERT
        output_1 = self.l1(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        # get embeddings
        embeddings = output_1.hidden_states
        # stack embeddings
        embeddings = torch.stack(embeddings)  # [layers, batch_sz, tokens, features]
        # detect parts
        embeddings = self.DP(embeddings)
        # reshape
        embeddings = embeddings.view(-1, d_inp*n_tokens*d_depth)
        # activation
        embeddings = self.GELU(embeddings)
        # dropout
        embeddings = self.dropout(embeddings)
        # classification
        embeddings = self.classification(embeddings)
        return embeddings


# init model
model = BERT1(n_labels=2).to(device)

# check params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# set epochs
epochs = 6

# init optimizer
optimizer = AdamW(model.parameters(),
                  lr=3e-5,
                  weight_decay=5.0,
                  correct_bias=False)

# create gradient scaler for mixed precision
scaler = GradScaler()

# set loss function
criterion = nn.CrossEntropyLoss()


# train function
def train(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        input_ids, token_type_ids, attn_masks, labels = (
            batch['input_ids'].to(device),
            batch['token_type_ids'].to(device),
            batch['attention_masks'].to(device),
            batch['labels'].to(device)
            )

        # clear gradients
        optimizer.zero_grad()
        # use fp16
        with autocast():
            logits = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attn_masks)
        loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #scheduler.step()
        pbar(step=batch_idx, info={'loss': train_loss.avg})
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
    # unpack dataloader
    for batch_idx, batch in enumerate(dataloader):
        input_ids, token_type_ids, attn_masks, labels = (
            batch['input_ids'].to(device),
            batch['token_type_ids'].to(device),
            batch['attention_masks'].to(device),
            batch['labels'].to(device)
            )

        # turn off gradient calcs:
        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attn_masks)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        f1 = metrics.f1_score(pred.to("cpu").numpy(), labels.to("cpu").numpy(), average='macro')
        valid_f1.update(f1, n=input_ids.size(0))
        valid_loss.update(loss, n=input_ids.size(0))
        valid_acc.update(correct, n=1)
        count += input_ids.size(0)
        pbar(step=batch_idx, info={'f1': valid_f1.avg})
    return {'valid_loss': valid_loss.avg,
            'valid_acc': valid_acc.sum /count,
            'valid_f1': valid_f1.avg}


# training
best_loss = 0
for epoch in range(1, epochs + 1):
    logger.info(f"epoch: {epoch}")
    train_log = train(train_dataloader)
    valid_log = test(dev_dataloader)
    logs = dict(train_log, **valid_log)
    for key, value in logs.items():
        if key == 'valid_loss':
            if value.item() < best_loss:
                torch.save(model.state_dict(), 'model1.pt')  # torch save
            best_loss = value.item()
    show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(show_info)


##
