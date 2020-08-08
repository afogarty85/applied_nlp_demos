# Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import re
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# set seed
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
    # reset index
    df = df.reset_index(drop=True)
    return df


# load df
df = prepare_df('C:\\Users\\Andrew\\Desktop\\df.pkl')


# prepare data
def clean_df(df):
    # strip dash but keep a space
    df['body'] = df['body'].str.replace('-', ' ')
    # lower case the data
    df['body'] = df['body'].apply(lambda x: x.lower())
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    # generate a word count for body
    df['word_count'] = df['body'].apply(lambda x: len(x.split()))
    # generate a word count for summary
    df['word_count_summary'] = df['title_osc'].apply(lambda x: len(x.split()))
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))
    # lower case to body
    df['body'] = df['body'].apply(lambda x: x.lower())
    # lower case to summary
    df['title_osc'] = df['title_osc'].apply(lambda x: x.lower())
    # add " </s>" to end of body
    df['body'] = df['body'] + " </s>"
    # add " </s>" to end of target
    df['target'] = df['target'] + " </s>"
    return df


# clean df
df = clean_df(df)

# remove transliterated words that GloVe can't find
no_matches = np.load('translit_no_match.npy')
no_matches = dict(zip(set(no_matches), range(len(set(no_matches)))))

# remove transliterated words from corpus
df['body'] = df['body'].apply(lambda x: ' '.join(list(filter(lambda x: x not in no_matches.keys(), x.split()))))

# instantiate T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# check token ids
tokenizer.eos_token_id
tokenizer.bos_token_id
tokenizer.unk_token_id
tokenizer.pad_token_id


# tokenize the main text
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
                            add_special_tokens=True,  # add tokens relative to model
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
body_input_ids, body_attention_masks = tokenize_corpus(df['body'].values, tokenizer, 512)

# show document 7422, now as a list of token ids
print('Original: ', df.iloc[7422]['body'])
# to do - work on punctuation white space fix
print('Token IDs:', body_input_ids[7422])

# how long are tokenized targets
ls = []
for i in range(df.shape[0]):
    ls.append(len(tokenizer.tokenize(df.iloc[i]['target'])))

temp_df = pd.DataFrame({'len_tokens': ls})
temp_df['len_tokens'].mean()  # 2.7
temp_df['len_tokens'].median()  # 3
temp_df['len_tokens'].max()  # 3

# create tokenized targets
target_input_ids, target_attention_masks = tokenize_corpus(df['target'].values, tokenizer, 3)


# prepare tensor data sets
def prepare_dataset(body_tokens, body_masks, target_token, target_masks):
    # create tensor data sets
    tensor_df = TensorDataset(body_tokens, body_masks, target_token, target_masks)
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
train_dataset, val_dataset, test_dataset = prepare_dataset(body_input_ids,
                                                           body_attention_masks,
                                                           target_input_ids,
                                                           target_attention_masks
                                                           )


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
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: summary tokens
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_target_ids = batch[2].cuda()
        b_target_mask = batch[3].cuda()

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

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
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: summary tokens
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_target_ids = batch[2].cuda()
        b_target_mask = batch[3].cuda()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

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


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # measure training time
    t0 = time.time()

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    predictions = []
    actuals = []

    # evaluate data for one epoch
    for step, batch in enumerate(dataloader):
        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: summary tokens
        b_input_ids = batch[0].cuda()  # batch size, length
        b_input_mask = batch[1].cuda()  # batch size, length
        b_target_ids = batch[2].cuda()  # batch size, length
        b_target_mask = batch[3].cuda()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)


            loss, prediction_scores = outputs[:2]

            total_test_loss += loss.item()

            generated_ids = model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3
                    )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_target_ids]

            total_test_acc += accuracy_score(target, preds)
            total_test_f1 += f1_score(preds, target,
                                       average='weighted',
                                       labels=np.unique(preds))
            predictions.extend(preds)
            actuals.extend(target)

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    avg_test_acc = total_test_acc / len(test_dataloader)

    avg_test_f1 = total_test_f1 / len(test_dataloader)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test PPL.': np.exp(avg_test_loss),
            'Test Acc.': avg_test_acc,
            'Test F1': avg_test_f1
        }
    )
    global df2
    temp_data = pd.DataFrame({'predicted': predictions, 'actual': actuals})
    df2 = df2.append(temp_data)

    return test_stats


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# instantiate model T5 transformer with a language modeling head on top
model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()  # to GPU


# helper function to count target distribution inside tensor data sets
def target_count(tensor_dataset):
    # set empty count containers
    count0 = 0
    count1 = 0
    # set total container to turn into torch tensor
    total = []
    for i in tensor_dataset:
        # for kabul tensor
        if torch.all(torch.eq(i[2], torch.tensor([20716, 83, 1]))):
            count0 += 1
        # for us tensor
        elif torch.all(torch.eq(i[2], torch.tensor([837, 1, 0]))):
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
    new_batch = []
    # for each obs
    for i in tensor_dataset:
        # if i is equal to kabul
        if torch.all(torch.eq(i[2], torch.tensor([20716, 83, 1]))):
            # append 0
            new_batch.append(0)
            # elif equal to US
        elif torch.all(torch.eq(i[2], torch.tensor([837, 1, 0]))):
            # append 1
            new_batch.append(1)
    samples_weight = torch.tensor([weight[t] for t in new_batch])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler


# need to make them numeric now
train_sampler = create_sampler(target_count(train_dataset), train_dataset)


# check balancer
train_dataloader = DataLoader(train_dataset,
                              batch_size=24,
                              sampler=train_sampler,
                              shuffle=False)


# lets check class balance for each batch to see how the sampler is working
for i, (input_ids, input_masks, target_ids, target_masks) in enumerate(train_dataloader):
    count_kabul = 0
    count_us = 0
    if i in range(0, 25):
        for j in target_ids:
            if (torch.all(torch.eq(j, torch.tensor([20716, 83, 1])))):
                count_kabul += 1
            else:
                count_us += 1
        print("batch index {}, 0/1: {}/{}".format(i, count_kabul, count_us))


# create DataLoaders with samplers
train_dataloader = DataLoader(train_dataset,
                              batch_size=16,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=16,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                              batch_size=16,
                              shuffle=True)


# Adam w/ Weight Decay Fix
# set to optimizer_grouped_parameters or model.parameters()
optimizer = AdamW(model.parameters(),
                  lr=9.909733480089917e-06,
                  weight_decay=0.65)

# epochs
epochs = 5

# lr scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# create gradient scaler for mixed precision
scaler = GradScaler()


# measure time for whole run
total_t0 = time.time()


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
        torch.save(model.state_dict(), 't5-classification.pt')  # torch save
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('./model_save/t5-classification/')  # transformers save
        tokenizer.save_pretrained('./model_save/t5-classification/')  # transformers save

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
df2 = pd.DataFrame({'predicted': [], 'actual': []})
test_stats = []
model.load_state_dict(torch.load('t5-classification.pt'))
testing(model, test_dataloader)
df_test_stats = pd.DataFrame(data=test_stats)
df_test_stats  # .825

df2.to_csv('t5-class.csv')


########################################################################################
# fine tune T5
training_stats = []
valid_stats = []
epochs = 5


# create gradient scaler for mixed precision
scaler = GradScaler()
def objective(trial):
    model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()  # to GPU

    # instantiate model - attach to GPU
    model.cuda()

    batch_size = trial.suggest_int('batch_size', low=12, high=24, step=4)
    dropout = trial.suggest_float('dropout_rate', low=0.1, high=0.4, step=0.05)
    learning_rate = trial.suggest_loguniform('lr', 1e-7, 1e-5)
    weight_decay = trial.suggest_float('weight_decay', low=0.5, high=1, step=0.05)
    model.config.__dict__['dropout_rate'] = dropout


    # Generate the model.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=24,
                                  sampler=train_sampler,
                                  shuffle=False)

    valid_dataloader = DataLoader(val_dataset,
                                  batch_size=24,
                                  shuffle=True)

    # optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)

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
        for batch in train_dataloader:
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_target_ids = batch[2].cuda()
            b_target_mask = batch[3].cuda()

            # clear previously calculated gradients
            optimizer.zero_grad()

            # runs the forward pass with autocasting.
            with autocast():
                # forward propagation (evaluate model on training batch)
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_target_ids,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]
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

            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_target_ids = batch[2].cuda()
            b_target_mask = batch[3].cuda()

            # tell pytorch not to bother calculating gradients
            # as its only necessary for training
            with torch.no_grad():

                # forward propagation (evaluate model on training batch)
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_target_ids,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]

            total_valid_loss += loss.item()

        global avg_val_loss
        avg_val_loss = total_valid_loss / len(valid_dataloader)

    trial.report(avg_val_loss, epoch)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_val_loss


study = optuna.create_study(direction="minimize",
                            pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                  max_resource=5,
                                                                  reduction_factor=3,
                                                                  ))
study.optimize(objective, n_trials=25)


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
