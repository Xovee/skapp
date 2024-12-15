import torch
import torch.nn as nn
from graph_variable_length import GraphLearner




class Model(nn.Module):

    def __init__(self, retrieval_num ,alpha=0.5, frame_num=1, feature_dim=768):

        super(Model, self).__init__()
        self.alpha = alpha
        self.frame_num = frame_num
        self.feature_dim = feature_dim

        self.visual_embedding = nn.Linear(feature_dim, feature_dim)
        self.textual_embedding = nn.Linear(feature_dim, feature_dim)
        self.retrieval_visual_embedding = nn.Linear(feature_dim, feature_dim)
        self.retrieval_textual_embedding = nn.Linear(feature_dim, feature_dim)

        self.tahn = nn.Tanh()
        self.relu = nn.ReLU()

        self.dual_attention_linear_1 = nn.Linear(feature_dim, feature_dim)
        self.dual_attention_linear_2 = nn.Linear(feature_dim, feature_dim)
        self.retrieval_dual_attention_linear_1 = nn.Linear(feature_dim * 2, feature_dim)
        self.retrieval_dual_attention_linear_2 = nn.Linear(feature_dim * 2, feature_dim)
        self.cross_modal_linear_1 = nn.Linear(feature_dim * 2, feature_dim)
        self.cross_modal_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.retrieval_cross_modal_linear_1 = nn.Linear(feature_dim * 2, feature_dim)
        self.retrieval_cross_modal_linear_2 = nn.Linear(feature_dim * 2, feature_dim)
        self.uni_modal_linear_1 = nn.Linear(feature_dim, 1)
        self.uni_modal_linear_2 = nn.Linear(feature_dim, 1)

        self.retrieval_uni_modal_linear_1 = nn.Linear(feature_dim, 1)
        self.retrieval_uni_modal_linear_2 = nn.Linear(feature_dim, 1)

        self.predict_linear_1 = nn.Linear(feature_dim, feature_dim)
        self.predict_linear_2 = nn.Linear(feature_dim * 2, 1)

        self.label_embedding_linear = nn.Linear(retrieval_num, feature_dim)

        self.graph = GraphLearner(device='cuda:0', hidden_dim=768, class_num = retrieval_num)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8,batch_first=True)


    def forward(self, retrieved_label_list,
                mean_pooling_vec,merge_text_vec,
                retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, text_mask, img_mask, CXMI):

        textual_feature_emb, visual_feature_emb = self.graph(merge_text_vec, mean_pooling_vec,
                                                                     retrieved_textual_feature_embedding,
                                                                     retrieved_visual_feature_embedding_cls, text_mask, img_mask)

        packed_feature = torch.cat([visual_feature_emb, textual_feature_emb], dim=1)

        CXMI = torch.cat([CXMI, CXMI], dim=1).unsqueeze(-1)
        CXMI = CXMI / CXMI.sum(dim=1, keepdim=True)

        output = torch.matmul(packed_feature.permute(0, 2, 1), CXMI).squeeze(-1)

        output = self.predict_linear_1(output)
        output = self.relu(output)

        label = self.label_embedding_linear(retrieved_label_list)

        output = torch.cat([output, label], dim=1)
        output = self.predict_linear_2(output)

        return output

