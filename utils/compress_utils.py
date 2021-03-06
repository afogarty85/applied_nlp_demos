import torch
import torch.nn as nn


class BertConcat(torch.nn.Module):
    '''Implementation of learned pooler reported by Tenney 2019
       Original paper: https://arxiv.org/abs/1905.05950
    '''
    def __init__(self):
        super(BertConcat, self).__init__()
        self.n_layers = 13
        # weight shape: [num_layers,]
        self.weights = torch.zeros(self.n_layers, dtype=torch.float32).normal_(0.0, 0.1).cuda()
        self.bias = torch.ones(1, dtype=torch.float32).cuda()

    def forward(self, x):
        '''
        Parameters
        ----------
        embeddings : torch tensor
            An embedding tensor with shape: [layers, batch_sz, tokens, features]

        Returns
        -------
        x  : torch tensor
            A tensor containing reduced embeddings with the following
            dimensions: [batch_sz, tokens, features, layers = 1]
        '''
        # change emitted batch shape of [layers, batch_sz, tokens, features] to:
        x = x.permute(1, 2, 3, 0) # [batch_sz, tokens, features, layers]
        x = torch.softmax(x, dim=-1)  # last dim
        x = torch.sum(torch.mul(x, self.weights),
                               dim=-1, keepdims=True) * self.bias
        return x




class TimeDistributed(nn.Module):
    '''
    PyTorch port of Keras TimeDistributed module.
    https://discuss.pytorch.org/t/timedistributed-cnn/51707/11

    Parameters
    ----------
    None

    Returns
    -------
    embeddings  : torch tensor
        A tensor containing embeddings with the following
        dimensions: [batch_sz, tokens, features, tokens]
    '''
    def __init__(self):
        super(TimeDistributed, self).__init__()
        self.n_layers = 13
        self.n_batch_sz = 16
        self.n_tokens = 384
        self.n_features = 768
        self.module = torch.nn.Linear(self.n_layers, self.n_tokens)
        self.module2 = torch.nn.Linear(self.n_features, self.n_tokens)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.permute(1, 2, 3, 0)
            x = torch.reshape(x, shape=(self.n_batch_sz*self.n_tokens, self.n_features, -1))
            x = self.module(x)
            return x.view(self.n_batch_sz, self.n_tokens,*x.shape[1:])

        if len(x.size()) == 3:
            x = self.module2(x)
            return x


class AdapterPooler(nn.Module):
    '''
    Pooler inspired by Adapter BERT
    Original paper: https://arxiv.org/abs/1902.00751
    GitHub: https://github.com/google-research/adapter-bert

    Parameters
    ----------
    shared_weights : flag
        Whether to use the TimeDistributed module or a CNN (not implemented)

    Returns
    -------
    embeddings  : torch tensor
        A tensor containing embeddings with the following
        dimensions: [batch_sz, tokens, tokens, layers]
    '''
    def __init__(self, shared_weights=True):
        super(AdapterPooler, self).__init__()
        self.GELU = torch.nn.GELU()
        self.n_layers = 13
        self.n_batch_sz = 16
        self.n_tokens = 384
        self.n_features = 768
        if shared_weights:
            self.pooler_layer = TimeDistributed()
        else:
            raise NotImplementedError

    def forward(self, x):
        # reshape original batch from [layers, batch_sz, tokens, features], to:
        x = x.permute(1, 2, 3, 0)  # [batch_sz, tokens, features, layers]
        # store some shape dims
        x = torch.reshape(x, shape=(-1, self.n_features, self.n_layers))  # [batch_sz*tokens, features, layers]
        # move axis
        x = x.permute(0, 2, 1)  # [batch_sz*tokens, layers, features]
        # apply pooler_layer
        x = self.pooler_layer(x) # [tokens, layers, batch_sz*tokens]
        # apply GELU
        x = self.GELU(x)
        # move axis
        x = x.permute(0, 2, 1)  # [batch_sz*tokens, tokens, layers]
        # reshape
        x = torch.reshape(x, shape=(-1, self.n_tokens, self.n_tokens, self.n_layers))  # [batch_sz, tokens, tokens, layers]
        return x
