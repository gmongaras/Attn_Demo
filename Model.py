import torch
from torch import nn

from Transformer import Transformer
from PositionalEncodings import PositionalEncodings



class Model(nn.Module):
    def __init__(self, vocab_size_enc, vocab_size_dec, max_seq_len, d_model, d_key, d_value, num_heads, causal, dropout=0.1):
        super(Model, self).__init__()
        self.vocab_size_enc = vocab_size_enc
        self.vocab_size_dec = vocab_size_dec
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.causal = causal
        self.dropout = dropout

        # Embedding layer is just a lookup table - linear layer no bias
        self.emb_encoder = nn.Linear(vocab_size_enc, d_model, bias=False)
        self.emb_decoder = nn.Linear(vocab_size_dec, d_model, bias=False)

        # Precalculate the positional encoding
        self.pos_enc = PositionalEncodings(d_model)(torch.arange(max_seq_len))

        # Encoder layers
        self.encoder = nn.Sequential(
            Transformer(d_model, d_key, d_value, num_heads, dropout, encoder=True),
            Transformer(d_model, d_key, d_value, num_heads, dropout, encoder=True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            Transformer(d_model, d_key, d_value, num_heads, dropout, encoder=False),
            Transformer(d_model, d_key, d_value, num_heads, dropout, encoder=False)
        )

        # Output layer goes form dim d_model to vocab_size
        self.fc = nn.Linear(d_model, vocab_size_dec)

    # Inputs:
    # cond: (bsz, cond_len)
    # X: (bsz, X_len)
    # padding_mask_enc: (bsz, cond_len)
    # padding_mask_dec: (bsz, X_len)
    def forward(self, cond, X, padding_mask_enc=None, padding_mask_dec=None):
        # If the input are integers, one-hot encode them
        if cond.dtype == torch.long:
            cond = torch.nn.functional.one_hot(cond, self.vocab_size_enc).float()
        if X.dtype == torch.long:
            X = torch.nn.functional.one_hot(X, self.vocab_size_dec).float()

        # Embedding
        cond = self.emb_encoder(cond)
        X = self.emb_decoder(X)

        # Add positional encodings
        cond = cond + self.pos_enc[:cond.size(1)].to(cond.device)[None, :, :]
        X = X + self.pos_enc[:X.size(1)].to(X.device)[None, :, :]

        # Encoder
        for layer in self.encoder:
            cond = layer(X=cond, padding_mask_X=padding_mask_enc)

        # Decoder
        for layer in self.decoder:
            # Note that we pass both padding masks in.
            # For self attention, we use the padding mask from output sequence
            # For cross attention, we use the padding mask from the input sequence
            # Basically, whatever the keys are, we use that padding mask
            X = layer(X=X, y=cond, padding_mask_X=padding_mask_dec, padding_mask_y=padding_mask_enc)

        # Output layer
        X = self.fc(X)

        return X
    



if __name__ == '__main__':
    # Test Model
    bsz = 2
    cond_len = 7
    X_len = 5
    vocab_size_enc = 10
    vocab_size_dec = 12
    max_seq_len = 10
    d_model = 4
    d_key = 2
    d_value = 2
    num_heads = 2
    causal = True
    dropout = 0.1

    cond = torch.randint(0, vocab_size_enc, (bsz, cond_len))
    X = torch.randint(0, vocab_size_dec, (bsz, X_len))
    padding_mask_cond = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]]).bool()
    padding_mask_X = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]).bool()

    model = Model(vocab_size_enc, vocab_size_dec, max_seq_len, d_model, d_key, d_value, num_heads, causal, dropout)
    out = model(cond, X, padding_mask_cond, padding_mask_X)
    print(out.shape)
    print(out)