import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import random
import json
from cleaner import clean_gutenberg_text
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=256, max_len=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, attn_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.fc_out(x)


class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length):
        self.data = [vocab[word] for word in text.split() if word in vocab]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long), torch.tensor(
            self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)


# Load dataset.
books = []
text = ""
directory = "./dataset/"
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        books.append(os.path.join(directory, filename))
for book in books:
    print(f"loading {book}.")
    text += clean_gutenberg_text(book)
vocab = {word: idx for idx, word in enumerate(set(text.split()))}
inv_vocab = {idx: word for word, idx in vocab.items()}  # Reverse mapping
vocab_size = len(vocab)
seq_length = 5
batch_size = 16

dataset = TextDataset(text, vocab, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = TransformerLanguageModel(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            x, y = x.permute(1, 0), y.permute(1, 0)
            output = model(x)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


# Train the model
print("Starting training...")
train(model, dataloader, optimizer, criterion)

# Save model and vocabulary
torch.save(model.state_dict(), "transformer_model.pth")
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

def generate_text(model, start_text, vocab, inv_vocab, max_length=50, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([vocab[word] for word in start_text.split() if word in vocab], dtype=torch.long).unsqueeze(
        1)
    generated = start_text.split()
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq.permute(1, 0))
            logits = output[-1, -1] / temperature  # Ensure correct indexing
            probabilities = torch.nn.functional.softmax(logits, dim=0)  # Adjust softmax dimension
            next_word_idx = torch.multinomial(probabilities, 1).item()
            if next_word_idx not in inv_vocab:
                break
            next_word = inv_vocab[next_word_idx]
            generated.append(next_word)
            input_seq = torch.cat([input_seq, torch.tensor([[next_word_idx]])], dim=0)
    return " ".join(generated)


# Load model and vocabulary when needed
with open("vocab.json", "r") as f:
    vocab = json.load(f)
inv_vocab = {idx: word for word, idx in vocab.items()}

model = TransformerLanguageModel(len(vocab))
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

# Example query
generated_text = generate_text(model, "simple sabotage", vocab, inv_vocab, max_length=20, temperature=0.8)
print(generated_text)

# simple sabotage sabotage sabotage sabotage sabotage incendiary it. it. If If If If If If If If If If If If If
# simple sabotage can can can can can can can can can can can can can can drivers Literary another unenforceability unenforceability frequent