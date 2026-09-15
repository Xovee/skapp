"""Permutation-invariant multimodal retrieval with an optional soft RRCP prior."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SKAPP(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=64, relation_dim=4,
                 dropout=.1, prediction_mean_views=False):
        super().__init__()
        self.visual = nn.Linear(feature_dim, hidden_dim)
        self.textual = nn.Linear(feature_dim, hidden_dim)
        self.query = nn.Sequential(nn.Linear(feature_dim*2, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.GELU())
        self.query_prediction = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))
        self.query_key = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
        self.neighbor_key = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
        self.relation_score = nn.Sequential(nn.Linear(relation_dim,32),nn.GELU(),nn.Linear(32,1))
        self.rrcp_gate = nn.Linear(hidden_dim*2,1)
        nn.init.zeros_(self.rrcp_gate.weight);nn.init.constant_(self.rrcp_gate.bias,-2.)
        self.rrcp_strength = nn.Parameter(torch.tensor(0.))
        self.correction = nn.Sequential(nn.Linear(hidden_dim*4+4,hidden_dim*2),nn.GELU(),
                                        nn.Dropout(dropout),nn.Linear(hidden_dim*2,1))
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim*2+4,32),nn.GELU(),nn.Linear(32,1))
        self.dropout = nn.Dropout(dropout)
        self.prediction_mean_views = prediction_mean_views
        if prediction_mean_views:
            state = torch.get_rng_state()
            self.direct_metadata = nn.Linear(1, hidden_dim*2, bias=False)
            nn.init.zeros_(self.direct_metadata.weight)
            torch.set_rng_state(state)

    def forward(self, query_visual, query_text, neighbor_visual, neighbor_text,
                neighbor_labels, relations, rrcp, mean_views=None, return_details=False):
        if not torch.isfinite(rrcp).all():
            raise ValueError('RRCP must be finite.')
        q = self.query(torch.cat([query_visual,query_text],dim=-1))
        if self.prediction_mean_views:
            if mean_views is None or mean_views.shape != (len(q), 1) or not torch.isfinite(mean_views).all():
                raise ValueError('Expected normalized query mean_views with shape (batch, 1).')
            q = q + self.direct_metadata(mean_views)
        values = torch.cat([F.gelu(self.visual(neighbor_visual)),F.gelu(self.textual(neighbor_text))],dim=-1)
        logits = (self.neighbor_key(values)*self.query_key(q).unsqueeze(1)).sum(-1)/math.sqrt(values.shape[-1])
        logits = logits + self.relation_score(relations).squeeze(-1)
        # Multiplicative positive prior in probability space, additive in
        # logit space. Negative contributions attenuate, never erase nodes.
        strength = F.softplus(self.rrcp_strength)*torch.sigmoid(self.rrcp_gate(q))
        logits = logits + strength*torch.tanh(rrcp)
        weights = torch.softmax(logits,dim=1)
        context = (weights.unsqueeze(-1)*values).sum(1)
        label_mean = (weights*neighbor_labels).sum(1,keepdim=True)
        label_var = (weights*(neighbor_labels-label_mean).square()).sum(1,keepdim=True)
        entropy = -(weights*weights.clamp_min(1e-12).log()).sum(1,keepdim=True)/math.log(max(2,weights.shape[1]))
        max_weight = weights.max(1,keepdim=True).values
        stats = torch.cat([label_mean,label_var.clamp_min(1e-12).sqrt(),entropy,max_weight],dim=1)
        query_pred = self.query_prediction(q)
        context_pred = label_mean + self.correction(self.dropout(torch.cat([q,context,stats],dim=1)))
        gate = torch.sigmoid(self.context_gate(torch.cat([q,stats],dim=1)))
        output = (1-gate)*query_pred+gate*context_pred
        if return_details:
            return output,query_pred,{'attention':weights,'context_gate':gate}
        return output
