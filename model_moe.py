import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class GLUExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, hidden_dim)
        self.w_val = nn.Linear(input_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.act(self.w_gate(x))
        val = self.w_val(x)
        out = self.w_out(gate * val)
        return self.dropout(out)


class TopKRouter(nn.Module):
    def __init__(self, input_dim, n_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(input_dim, n_experts)
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.gate(x)
        weights = self.softmax(logits)
        top_k_vals, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_vals = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-9)
        return top_k_vals, top_k_indices, weights


class AffineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale + self.shift


class AttnFusionMoE(nn.Module):
    def __init__(self, vocab_size, bert_dim, bio_dim, d_model, n_experts, top_k, dropout):
        super().__init__()
        self.enc_u5 = SequenceEncoder(vocab_size, d_model, dropout)
        self.enc_cds = SequenceEncoder(vocab_size, d_model, dropout)
        self.enc_u3 = SequenceEncoder(vocab_size, d_model, dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(bert_dim + bio_dim, d_model),
            nn.LayerNorm(d_model), nn.ReLU()
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([GLUExpert(d_model, d_model * 2, dropout) for _ in range(n_experts)])
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
        self.output_adapter = AffineHead()

    def forward(self, u5, cds, u3, bert_x, bio_x):
        s_u5, s_cds, s_u3 = self.enc_u5(u5), self.enc_cds(cds), self.enc_u3(u3)
        local_seq = torch.cat([s_u5, s_cds, s_u3], dim=1)
        global_vec = self.global_proj(torch.cat([bert_x, bio_x], dim=1)).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=global_vec, key=local_seq, value=local_seq)
        x = self.norm(global_vec + attn_out).squeeze(1)

        weights, indices, gate_weights = self.router(x)
        batch_size, top_k = indices.size()
        final_output = torch.zeros_like(x)
        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)
        batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(top_k)

        for e_id, expert in enumerate(self.experts):
            mask = (flat_indices == e_id)
            if mask.any():
                input_indices = batch_indices[mask]
                expert_out = expert(x[input_indices])
                final_output.index_add_(0, input_indices, expert_out * flat_weights[mask].unsqueeze(1))

        out = self.regressor(final_output).squeeze(-1)
        return self.output_adapter(out), gate_weights, indices