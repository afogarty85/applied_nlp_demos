# GPT-2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datautils
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, LineByLineTextDataset
from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time, os, datetime, random, re
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)

# tell pytorch to use GPU
device = torch.device("cuda")


# prepare and load data
def prepare_df(pkl_location):
    # read pkl as pandas
    df = pd.read_pickle(pkl_location)
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    # split and shuffle data
    train, valid = train_test_split(df['body'], test_size=0.2)
    return train.reset_index(drop=True), valid.reset_index(drop=True)


# instantiate shuffled train and validation
train, valid = prepare_df('C:\\Users\\Andrew\\Desktop\\df.pkl')

# save to text for transformers TextDataset
np.savetxt('C:\\Users\\Andrew\\Desktop\\train.txt', train, fmt="%s")
np.savetxt('C:\\Users\\Andrew\\Desktop\\valid.txt', valid, fmt="%s")

# instantiate GPT2 tokenizer, byte-level encoding
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# add special tokens that otherwise all share the same id
tokenizer.add_special_tokens({'bos_token': '<bos>',
                              'eos_token': '<eos>',
                              'pad_token': '<pad>'})

# check token ids
tokenizer.eos_token_id
tokenizer.bos_token_id
tokenizer.unk_token_id
tokenizer.pad_token_id




# instantiate model GPT2 transformer with a language modeling head on top
model = GPT2LMHeadModel.from_pretrained('distilgpt2').cuda()  # to GPU

# Transfomer Data Set -- we need everything the same length
train_set = TextDataset(tokenizer=tokenizer,
                        file_path='C:\\Users\\Andrew\\Desktop\\train.txt',
                        block_size=1025)

valid_set = TextDataset(tokenizer=tokenizer,
                        file_path='C:\\Users\\Andrew\\Desktop\\valid.txt',
                        block_size=1025)

# prepare data loaders
train_dataloader = datautils.DataLoader(dataset=train_set,
                                        sampler=SequentialSampler(train_set),
                                        batch_size=3,
                                        drop_last=True,
                                        shuffle=False)


valid_dataloader = datautils.DataLoader(dataset=valid_set,
                                        sampler=SequentialSampler(valid_set),
                                        batch_size=3,
                                        drop_last=True,
                                        shuffle=False)


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
        #
        # `batch` contains our text in a PyTorch tensor
        #  that we need to slice opposite ends off
        slice_last = batch[:, :-1].cuda()
        slice_first = batch[:, 1:].cuda()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            logits = model(input_ids=slice_last)[0]

            loss = criterion(logits.flatten(0, 1), slice_first.flatten(0))
            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            train_total_loss += loss.item()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # clip the gradients to 1 to reduce exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # update the learning rate
        scheduler.step()

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss
        }
    )

    # training time end
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")

    return training_stats


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
    total_valid_loss = 0

    # evaluate data for one epoch
    for batch in dataloader:

        # Unpack this training batch from our dataloader:
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        #
        # `batch` contains our text in a PyTorch tensor
        #  that we need to slice opposite ends off
        slice_last = batch[:, :-1].cuda()
        slice_first = batch[:, 1:].cuda()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            logits = model(input_ids=slice_last)[0]

            loss = criterion(logits.flatten(0, 1), slice_first.flatten(0))
            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            total_valid_loss += loss.item()

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val PPL.': np.exp(avg_val_loss)
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val ppl | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {np.exp(avg_val_loss):.3f} | {training_time:}")

    return valid_stats


# create gradient scaler for mixed precision
scaler = GradScaler()

# training length
epochs = 10

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer: Adam w/ Weight Decay Fix
# set to optimizer_grouped_parameters or model.parameters()
optimizer = AdamW(model.parameters(),
                  lr=2e-5)


# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


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
        torch.save(model.state_dict(), 'gpt2-model1.pt')  # torch save
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('./model_save/gpt2/')  # transformers save
        tokenizer.save_pretrained('./model_save/gpt2/')  # transformers save



model.eval();
# beam search
text = "The Afghan National Army reported"
ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).cuda()
generated_ids = model.generate(
                        input_ids=ids,  # input
                        max_length=45,  # default 20
                        min_length=0,  # default 0
                        do_sample=True,  # don't use greedy decoding
                        early_stopping=False,  # search is stopped when at least num_beams sentences finished
                        temperature=2.45,  # default 1.0
                        top_k=45,  # default 50
                        top_p=0.7,  # default 1.0
                        repetition_penalty=2.0,  # rep. penalty
                        num_beams=6,
                        num_return_sequences=2, #  num ind. computed returned sequences
                        bos_token_id=tokenizer.bos_token_id
                        )

results = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

for i in results:
    print(i, end='\n \n')



#
