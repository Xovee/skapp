import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


def graph_norm_ours(A, mask, batch=False, self_loop=True, symmetric=True):
    A = A * mask.unsqueeze(1)
    d = A.sum(-1) 
    if symmetric:
        d = torch.pow(d + 1e-8, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
        d = torch.pow(d + 1e-8, -1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A)

    return norm_A


def cal_edge_emb(x, mask, p=2, dim=1):

    x = F.normalize(x, p=p, dim=dim)
    x = x * mask.unsqueeze(-1)  
    A = torch.bmm(x, x.transpose(1, 2))
    return A


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True,
                 dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device = device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim)).to(device)
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj, mask):
        x = feat
        node_size = adj.size()[1]
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device=self.device).unsqueeze(dim=0).expand(adj.size(0), -1, -1)
        adj = adj + I
        adj = graph_norm_ours(adj, mask, batch=True, self_loop=True, symmetric=True)

        pre_sup = torch.matmul(x, self.gcn_weights)
        output = torch.matmul(adj, pre_sup)

        if self.bias:
            output += self.gcn_bias.unsqueeze(0)
        if self.act is not None:
            output = self.act(output)
        output = output * mask.unsqueeze(-1)

        return output.transpose(1, 2) 


class GraphLearner(nn.Module):
    def __init__(self, device, hidden_dim, class_num):
        super().__init__()
        self.device = device
        self.alpha = 0.1
        self.alpha_it = 0.7
        self.beta_it = 0.5
        self.node_num = 1
        self.hidden_dim = hidden_dim
        self.class_num = class_num

        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=class_num + 1)
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=class_num + 1)

    def forward(self, input_text, input_img, base_text_features, base_img_features, text_mask, img_mask):

        with torch.no_grad():
            node_cluster_t = base_text_features
            node_cluster_i = base_img_features

        graph_o_t_all = []

        for index in range(1):
            with torch.no_grad():
                inputs_text = input_text
                inputs_img = input_img

                node_cluster_tt = node_cluster_t
                node_cluster_it = node_cluster_i

                feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)
                feat_it = torch.cat([inputs_img, node_cluster_it], dim=1)

                edge_tt = cal_edge_emb(feat_tt, text_mask)
                edge_it = cal_edge_emb(feat_it, img_mask)

            graph_o_tt = self.GCN_tt(feat_tt, edge_tt, text_mask)
            graph_o_it = self.GCN_it(feat_it, edge_it, img_mask)

            graph_o_t = graph_o_tt * self.alpha_it + (1 - self.alpha_it) * graph_o_it
            graph_o_t_all.append(graph_o_t[:, :, 0])

        graph_o_t = torch.stack(graph_o_t_all, dim=0).transpose(1, 0)

        text_mask_expanded = text_mask[:, :self.class_num].unsqueeze(-1)
        img_mask_expanded = img_mask[:, :self.class_num].unsqueeze(-1)

        graph_o_t_expanded = graph_o_t.repeat(1, self.class_num, 1)

        fused_text = self.beta_it * base_text_features + (1 - self.beta_it) * graph_o_t_expanded
        fused_img = self.beta_it * base_img_features + (1 - self.beta_it) * graph_o_t_expanded

        fused_text = fused_text * text_mask_expanded
        fused_img = fused_img * img_mask_expanded

        return fused_text, fused_img
