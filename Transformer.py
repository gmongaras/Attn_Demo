import torch
from torch import nn

from Attn import MultiHeadAttention



class Transformer(nn.Module):
    def __init__(self, d_model, d_key, d_value, num_heads, dropout=0.1, encoder=True):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.encoder = encoder
        self.dropout = dropout

        # Encoder layer is bidirectional attention
        if encoder:
            self.self_attn = MultiHeadAttention(d_model, d_key, d_value, num_heads, causal=False, dropout=dropout)
        # Decoder layer is causal attention and then cross attention
        else:
            # Causal self attention
            self.self_attn = MultiHeadAttention(d_model, d_key, d_value, num_heads, causal=True, dropout=dropout)
            # Cross attention
            self.cross_attn = MultiHeadAttention(d_model, d_key, d_value, num_heads, causal=False, dropout=dropout)


        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Linear(4*d_model, d_model)
        )

        self.norm_attn = nn.LayerNorm(d_model)
        if not encoder:
            self.norm_cross = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

    # Inputs:
    # X: (bsz, X_len, d_model)
    # y: (bsz, y_len, d_model)
    # mask: (bsz, X_len)
    def forward(self, X, y=None, padding_mask_X=None, padding_mask_y=None):
        # If this is a decoder, we expect y to be provided
        if not self.encoder:
            assert y is not None

        # If we have padding_mask_y, we expect this to be a decoder layer
        # and for y to be provided
        if padding_mask_y is not None:
            assert y is not None

        # Self-attention (causal if decoder, bidirectional if encoder)
        X = self.self_attn(self.norm_attn(X), padding_mask=padding_mask_X) + X

        # Cross-attention (note that this is always bidirectional)
        if not self.encoder:
            X = self.cross_attn(self.norm_cross(X), y, padding_mask=padding_mask_y) + X

        # Feedforward
        X = self.feedforward(self.norm_mlp(X)) + X

        return X
    




if __name__ == '__main__':
    # Test Transformer
    bsz = 2
    X_len = 5
    d_model = 4
    d_key = 2
    d_value = 2
    num_heads = 2
    causal = True
    dropout = 0.1

    X = torch.randn(bsz, X_len, d_model)
    padding_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]).bool()

    transformer = Transformer(d_model, d_key, d_value, num_heads, causal, dropout)
    out = transformer(X, padding_mask)
    print(out.shape)
    # torch.Size([2, 5, 4])