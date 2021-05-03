
##############
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import copy
import pandas as pd
import plotly.express as px
from sklearn import metrics
import random
torch.set_printoptions(sci_mode=False)

# set seed for reproducibility
torch.backends.cudnn.deterministic = True
seed = 823
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def mask_generator(inject_p, inject, reject, weights):
    mask = {
                    name: (
                        torch.tensor(np.random.choice([False, True],
                                                      size=torch.numel(weight),
                                                      p=[inject_p, (1-inject_p)])
                                     .reshape(weight.shape))

                        if any(weight in name for weight in inject)
                        and not any(weight in name for weight in reject) else
                        torch.tensor(np.random.choice([False, True],
                                                      size=torch.numel(weight),
                                                      p=[0.0, 1.0])
                                     .reshape(weight.shape))
                      )
                    for name, weight in weights.items()
                    }
    return mask


def weight_generator(pre_trained, mask, trained):
    # create a new model state
    result = {}
    # for each key
    for key, value in pre_trained.items():
        # add the key
        result[key] = []
        # if False, replace initial value with trained value
        result[key] = pre_trained[key].cuda().where(mask[key].cuda(), trained[key].cuda())
    return result


def comparative_weights(pre_trained, trained, inject_p, inject, reject, seed):
    # set seed
    np.random.seed(seed=seed)
    # create mask
    mask = mask_generator(inject_p, inject, reject, pre_trained.state_dict())
    # generate weights
    injected_weights = weight_generator(pre_trained.state_dict(), mask, trained.state_dict())
    return injected_weights, mask


def find_diff(weight_one, weight_two, mask_one, mask_two):
    # find the diff
    diff = {key: weight_one[key].cuda() - weight_two.get(key, 0).cuda() for key in weight_one}
    # search layers
    search_layers = ['intermediate.dense.weight', 'output.dense.weight', 'classifier']
    # resulting sum
    layer_weight_diff = {
                    name: (
                        torch.sum(weight).item())
                        for name, weight in diff.items()
                        if any(weight in name for weight in search_layers)
                        and not any(weight in name for weight in reject)
                    }

    # get all weights flattened from weight1
    weight_one_weights = {
                    name: (
                        weight.view(-1))
                        for name, weight in weight_one.items()
                        if any(weight in name for weight in search_layers)
                        and not any(weight in name for weight in reject)
                    }

    # get all weights flattened from weight2
    weight_two_weights = {
                    name: (
                        weight.view(-1))
                        for name, weight in weight_two.items()
                        if any(weight in name for weight in search_layers)
                        and not any(weight in name for weight in reject)
                    }

    # get all masks flattened from weight1
    mask_one_masks = {
                    name: (
                        weight.view(-1))
                        for name, weight in mask_one.items()
                        if any(weight in name for weight in search_layers)
                        and not any(weight in name for weight in reject)
                    }

    # get all masks flattened from weight2
    mask_two_masks = {
                    name: (
                        weight.view(-1))
                        for name, weight in mask_two.items()
                        if any(weight in name for weight in search_layers)
                        and not any(weight in name for weight in reject)
                    }

    # total difference in weight values
    total_weight_diff = sum([value for key, value in layer_weight_diff.items()])

    # send results to df
    weight_diff_df = pd.DataFrame({k: v for k, v in layer_weight_diff.items()},
                      columns=[key for key in layer_weight_diff.keys()],
                      index=[0])

    return layer_weight_diff, total_weight_diff, weight_diff_df, weight_one_weights, weight_two_weights, mask_one_masks, mask_two_masks


def zero_injection(initial_weights, trained_weights, mask):
    ''' zeros all weights and then injects in masked selection '''
    # copy the weights
    initial_weights_copy = copy.deepcopy(initial_weights.state_dict())
    trained_weights_copy = copy.deepcopy(trained_weights.state_dict())

    # set all the values to zero
    for key, value in initial_weights_copy.items():
        initial_weights_copy[key][initial_weights_copy[key] < 0] = 0
        initial_weights_copy[key][initial_weights_copy[key] > 0] = 0

    state_dict = {}
    # for each key
    for key, value in initial_weights_copy.items():
        # add the key
        state_dict[key] = []
        # if False, replace initial value with trained value
        state_dict[key] = initial_weights_copy[key].cuda().where(mask[key].cuda(), trained_weights_copy[key].cuda())

    return state_dict


def pretrain_injection(initial_weights, trained_weights, mask):
    ''' injects weights over the pre-trained model '''
    # copy the weights
    initial_weights_copy = copy.deepcopy(initial_weights.state_dict())
    trained_weights_copy = copy.deepcopy(trained_weights.state_dict())

    state_dict = {}
    # for each key
    for key, value in initial_weights_copy.items():
        # add the key
        state_dict[key] = []
        # if False, replace initial value with trained value
        state_dict[key] = initial_weights_copy[key].cuda().where(mask[key].cuda(), trained_weights_copy[key].cuda())

    return state_dict


def edge_comparison(mask_one, mask_two):
    ''' return masks just for weights that are not shared '''
    # if True, the weight is different at that position
    shared_masks = {
                    name: (
                        torch.ne(mask_one[name], mask_two[name]))
                        for name, weight in mask_one.items()
                    }

    # calculate total number of different masks
    total_mask_diff = ([torch.sum(value).item() for key, value in shared_masks.items()])


    # do weight indices again, but make all non-matches false
    shared_masks_true = {
                    name: (
                        torch.eq(mask_one[name], mask_two[name]))
                        for name, weight in mask_one.items()
                        if any(weight in name for weight in ['intermediate.dense.weight', 'output.dense.weight', 'classifier'])
                    }

    return shared_masks_true, total_mask_diff


def diff_injection(initial_weights, trained_weights, shared_masks_true, zero=False):
    ''' inject the difference of weights between two edge cases
    into pre-trained BERT '''
    # copy the weights
    initial_weights_copy = copy.deepcopy(initial_weights.state_dict())
    trained_weights_copy = copy.deepcopy(trained_weights.state_dict())

    if zero:
        ''' turn all pre-trained to zero to see how these weights do alone '''
        # set all the values to zero
        for key, value in initial_weights_copy.items():
            initial_weights_copy[key][initial_weights_copy[key] < 0] = 0
            initial_weights_copy[key][initial_weights_copy[key] > 0] = 0

    # create a new model state
    reshape_output = [768, 3072]
    reshape_intermed = [3072, 768]
    reshape_class = [2, 768]

    state_dict = {}
    # for each key
    for key, value in initial_weights_copy.items():
        # add the key
        state_dict[key] = []
        if key not in shared_masks_true.keys():
            state_dict[key] = initial_weights_copy[key]
        elif key in shared_masks_true.keys() and not 'attention':
            if 'intermediate' in key:
                # if False, replace initial value with trained value
                state_dict[key] = initial_weights_copy[key].cuda().where(shared_masks_true[key].reshape(reshape_intermed).cuda(), trained_weights_copy[key].cuda())
            elif 'output' in key:
                # if False, replace initial value with trained value
                state_dict[key] = initial_weights_copy[key].cuda().where(shared_masks_true[key].reshape(reshape_output).cuda(), trained_weights_copy[key].cuda())
            elif 'classifier.weight' in key:
                state_dict[key] = initial_weights_copy[key].cuda().where(shared_masks_true[key].reshape(reshape_class).cuda(), trained_weights_copy[key].cuda())
            elif 'classifier.bias' in key:
                state_dict[key] = initial_weights_copy[key].cuda().where(shared_masks_true[key].reshape([2]).cuda(), trained_weights_copy[key].cuda())

    return state_dict



# preliminary items:
# initialize pre-train
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()

# make a copy
initial_weights = copy.deepcopy(model)

# make another copy
trained_weights = copy.deepcopy(initial_weights)
# load 4 epoch weights into it
trained_weights = torch.load('C:\\BERTVision\\code\\torch\\model_checkpoints\\bert-base-uncased\\RTE\\2021-03-11_20-14-05.pt')

# areas to inject and reject
inject = ['intermediate.dense', 'output.dense']
reject = ['attention']


# create a model state
duration = 1000
seed_store = np.zeros(duration)
inject_store = np.zeros(duration)
tweight_store = np.zeros(duration)

for step, i in enumerate(range(duration)):
    seed = i
    inject_p = np.random.uniform()
    inject_store[i] = inject_p
    seed_store[i] = seed

    base_case, base_mask = comparative_weights(pre_trained=initial_weights,
                                     trained=trained_weights,
                                     inject_p=0.0,
                                     inject=inject,
                                     reject=reject,
                                     seed=seed)

    compare_case, compare_mask = comparative_weights(pre_trained=initial_weights,
                                     trained=trained_weights,
                                     inject_p=inject_p,
                                     inject=inject,
                                     reject=reject,
                                     seed=seed)
    if step == 0:
        layer_weight_diff, total_weight_diff, weight_diff_df, weight_one_weights, weight_two_weights, mask_one_masks,  mask_two_masks = find_diff(base_case, compare_case, base_mask, compare_mask)

    elif step > 0:
        layer_weight_diff, total_weight_diff, temp_weight_df, weight_one_weights, weight_two_weights, mask_one_masks,  mask_two_masks = find_diff(base_case, compare_case, base_mask, compare_mask)
        weight_diff_df = weight_diff_df.append(temp_weight_df)

    tweight_store[i] = total_weight_diff




####


import pandas as pd
trials = pd.read_pickle('C:\\BERTVision\\code\\torch\\pfreezing_trials\\bert-base-uncased\\MSR\\2021-02-26_12-02-40.pkl')
dev_loss, dev_metric, freeze_p, no_freeze, seed = [], [], [], [], []
for i in range(len(trials)):
    dev_loss.append(getattr(trials, '_trials')[i]['result']['dev_loss'])
    dev_metric.append(getattr(trials, '_trials')[i]['result']['metric'])
    freeze_p.append(getattr(trials, '_trials')[i]['misc']['vals']['freeze_p'][0])
    no_freeze.append(getattr(trials, '_trials')[i]['misc']['vals']['no_freeze'][0])
    seed.append(getattr(trials, '_trials')[i]['misc']['vals']['seed'][0])
df = pd.DataFrame({'dev_loss': dev_loss, 'dev_acc': dev_metric,
                   'freeze_p': freeze_p, 'no_freeze': no_freeze, 'seed': seed})