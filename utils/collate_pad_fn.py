# we can use collate_fn to dynamically pad our tokens in the data loader

def collate_fn_padd(batch):
    batch_inputs = list()
    max_size = 0
    # find the max length -- for every item
    for item in batch:
        # if len of the input ids > max_size:
        if len(item['input_ids'][0]) > max_size:
            # get a new max size
            max_size = len(item['input_ids'][0])
    # for every item
    for item in batch:
        # check current length of tokens
        current_len = len(item['input_ids'][0])
        # if tokens are smaller than the max_size
        if current_len < max_size:
            # find the difference
            len_diff = max_size - current_len
            # generate some zeros
            zeros = torch.zeros(len_diff).reshape(1, len_diff).long()
            # turn them to padding
            zeros[zeros == 0] = 100  # change to padding
            # cat them together
            temp_input = torch.cat([item['input_ids'][0].reshape(1, item['input_ids'][0].shape[0]), zeros], dim=1)
            # add the inputs to a list
            batch_inputs += list(temp_input)
        else:
            batch_inputs += list(item['input_ids'])
    return batch_inputs