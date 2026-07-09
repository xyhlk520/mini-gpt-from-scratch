import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def evaluate(model, inputs, targets, batch_size, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
         x = inputs[i:i+batch_size].to(device)
         y = targets[i:i+batch_size].to(device)

         logits = model(x)
         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
         losses.append(loss.item())
    
    model.train()
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity
