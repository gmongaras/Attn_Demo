import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from Model import Model

# Define a simple Dataset class for translation
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
    
    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]

# Define a training function
def train_model(model, dataset, epochs, batch_size, vocab_size_dec, device='cpu'):
    # DataLoader to handle batches
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Move model to device (GPU/CPU)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (src, tgt) in enumerate(data_loader):
            # src is "English" and tgt is "Spanish"
            src, tgt = src.to(device), tgt.to(device)

            # The labels are the target, offset by one (starts at index 1, not 0)
            label = tgt[:, 1:]

            # Cut the target sequence by 1 as the last token has no label
            tgt = tgt[:, :-1]

            # Input and target padding masks (this is just a simple example
            # where we mask 0 tokens. In practice, you should do this for tokens
            # where the ID is the padding ID)
            padding_mask_src = (src != 0).bool()
            padding_mask_tgt = (tgt != 0).bool()
            padding_mask_src[:, 0] = True
            padding_mask_tgt[:, 0] = True

            # Forward pass through the model
            output = model(src, tgt, padding_mask_src, padding_mask_tgt)

            # Reshape the output and target for computing loss
            output = output.reshape(-1, vocab_size_dec)
            label = label.reshape(-1)

            # Compute loss
            loss = criterion(output, label)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}')

# Example usage

if __name__ == '__main__':
    # Hyperparameters
    vocab_size_enc = 10
    vocab_size_dec = 12
    max_seq_len = 10
    d_model = 4
    d_key = 2
    d_value = 2
    num_heads = 2
    causal = True
    dropout = 0.1
    epochs = 10
    batch_size = 2

    # Dummy data for training (source and target sequences)
    source_sentences = torch.randint(0, vocab_size_enc, (100, 7))
    target_sentences = torch.randint(0, vocab_size_dec, (100, 6))

    # Create dataset and model
    dataset = TranslationDataset(source_sentences, target_sentences)
    model = Model(vocab_size_enc, vocab_size_dec, max_seq_len, d_model, d_key, d_value, num_heads, causal, dropout)

    # Train the model
    train_model(model, dataset, epochs, batch_size, vocab_size_dec, device='cpu')