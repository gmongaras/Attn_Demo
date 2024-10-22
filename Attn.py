import torch
from torch import nn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_key, d_value, num_heads, causal, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.causal = causal
        self.dropout = dropout

        self.q_linear = nn.Linear(d_model, d_key*num_heads)
        self.k_linear = nn.Linear(d_model, d_key*num_heads)
        self.v_linear = nn.Linear(d_model, d_value*num_heads)
        self.o_linear = nn.Linear(d_value*num_heads, d_model)

    # Inputs:
    # X: (bsz, X_len, d_model)
    # y: (bsz, y_len, d_model)
    # mask: (bsz, y_len)
    def forward(self, X, y=None, padding_mask=None):
        # If this is causal, we expect y to not be provided
        if self.causal:
            assert y is None

        # Easily handle the case where y is not provided
        if y is None:
            y = X

        # The padding mask should be of shape (bsz, y_len)
        assert padding_mask is None or padding_mask.size() == (y.size(0), y.size(1))

        # Project the query, key, and value
        query = self.q_linear(X)
        key = self.k_linear(y)
        value = self.v_linear(y)

        # Get dimensions
        bsz, X_len, d_model = query.size()
        _, y_len, _ = key.size()

        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # Attention mask of shape (bsz, num_heads, k_len, X_len)
        causal_mask = torch.triu(torch.ones(bsz, self.num_heads, X_len, y_len)).bool().mT if self.causal \
            else torch.ones(bsz, self.num_heads, X_len, y_len).bool()
        # Add padding mask. Note that this is added along the columns, not the rows
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & padding_mask

        # Split the query, key, and value into multiple heads
        # (bsz, __len, d_model) -> (bsz, __len, num_heads, dim_per_head)
        # Note that the query and key have the same dimension. The value can be different.
        query = query.view(bsz, X_len, self.num_heads, self.d_key).transpose(1, 2)
        key = key.view(bsz, y_len, self.num_heads, self.d_key).transpose(1, 2)
        value = value.view(bsz, y_len, self.num_heads, self.d_value).transpose(1, 2)

        # Compute the attention score
        scores = query @ key.mT / (self.d_key ** 0.5)

        # Mask out the future tokens
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Normalize the scores
        attn = scores.softmax(dim=-1)

        # Apply dropout
        attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)

        # Project the values
        out = attn @ value

        # Combine the heads
        # (bsz, num_heads, X_len, dim_per_head) -> (bsz, X_len, num_heads * dim_per_head)
        out = out.transpose(1, 2).contiguous().view(bsz, X_len, self.num_heads * self.d_value)

        # Output projection to combine the heads
        out = self.o_linear(out)

        return out






if __name__ == '__main__':
    mha = MultiHeadAttention(512, 64, 64, 8, False)
    X = torch.randn(16, 20, 512)
    y = torch.randn(16, 15, 512)
    padding_mask = torch.rand(16, 15).round().bool()
    out = mha(X, y, padding_mask)
    print(out.size())