class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, x)
        return output
