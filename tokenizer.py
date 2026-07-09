import torch

#get the dataset with input and target features
def get_dataset(tokenized_text, num_examples, context_window_length, test_split=0.1):
    input_feat = []
    target_feat = []
    # create input and target features using a sliding window
    for i in range(0, len(tokenized_text), context_window_length):
        block = tokenized_text[i:i+context_window_length+1]
        input_seq = block[:-1]
        target_seq = block[1:]
        # eliminate blocks that are smaller than the context window length
        if len(block) < context_window_length + 1:
            continue
        input_feat.append(input_seq)
        target_feat.append(target_seq)
        # break the loop if we have enough examples
        if len(input_feat) >= num_examples:
            break
        
      # convert the input and target features to tensors and move them to the device
    inputs = torch.tensor(input_feat, dtype=torch.long)
    targets = torch.tensor(target_feat, dtype=torch.long)
    split_idx = int(num_examples * (1-test_split))
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    test_inputs = inputs[split_idx:]
    test_targets = targets[split_idx:]

    return train_inputs, train_targets, test_inputs, test_targets
