from config import GPTConfig
from tokenizer import get_dataset
from model import GPTModel
from datasets import load_dataset
from torch.nn import functional as F
import tiktoken
import torch
#get tokenizer same as gpt2
tokenizer = tiktoken.get_encoding("gpt2")

#define hyperparameters
VOCAB_SIZE = tokenizer.vocab_size()
BATCH_SIZE = 10
SEQ_LEN = 128
N_EMBD = 256
N_HEAD = 8
N_LAYER = 4
NUM_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get the dataset with input and target features
dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train")
all_text = ""
all_data = dataset["page"]
for example in all_data:
    all_text += "<page>" + example + "</page>"

tokenized_text = tokenizer.encode(all_text)
train_inputs, train_targets, _, _ = get_dataset(10, 4, 0)
print(train_inputs.shape)

#create config object
config = GPTConfig(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER
)

model = GPTModel(config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=20, min_lr=5e-6, threshold=1e-4)

i = 1
losses = []
while i < NUM_STEPS:
    for j in range(0, len(train_inputs), BATCH_SIZE):
        x = train_inputs[j:j+BATCH_SIZE]
        y = train_targets[j:j+BATCH_SIZE]

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        losses.append(loss.item())
        
        optimizer.step()
        optimizer.zero_grad()
    

        loss = loss.item()
        scheduler.step(loss)

   
        # Print the average loss for the epoch
        lr = optimizer.param_groups[0]["lr"]
        print(f"Step {i+1}/{NUM_STEPS}, Loss: {loss}, LR: {lr}")

        i += 1



