import torch
import torch.nn as nn
from graph import GraphLearner

class RRCP_Model(nn.Module):

    def __init__(self, retrieval_num ,alpha=0.5, frame_num=1, feature_dim=768):

        super(RRCP_Model, self).__init__()
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.retrieval_num = retrieval_num
        self.predict_linear_1 = nn.Linear(feature_dim * retrieval_num * 2, feature_dim)
        self.predict_linear_2 = nn.Linear(feature_dim * 2, 1)
        self.relu = nn.ReLU()
        self.label_embedding_linear = nn.Linear(retrieval_num, feature_dim)
        self.GraphLearner = GraphLearner(device='cuda:0', hidden_dim=768, class_num=self.retrieval_num)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8,batch_first=True)


    def forward(self, mean_pooling_vec, merge_text_vec,
                retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls, retrieved_label_list):

        if retrieved_textual_feature_embedding.dim() == 3:
            retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.unsqueeze(2)
            retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls.unsqueeze(2)
            retrieved_label_list = retrieved_label_list.unsqueeze(1)

        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:, :self.retrieval_num, :, :]
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls[:, :self.retrieval_num, :, :]
        retrieved_label_list = retrieved_label_list[:, :self.retrieval_num]

        textual_feature_emb, visual_feature_emb = self.GraphLearner(merge_text_vec, mean_pooling_vec,
                                                                     retrieved_textual_feature_embedding.squeeze(2),
                                                                     retrieved_visual_feature_embedding_cls.squeeze(2))
        packed_feature = torch.cat([visual_feature_emb, textual_feature_emb], dim=1)

        output = self.multihead_attn(packed_feature, packed_feature, packed_feature)
        values = output[0]
        output = values.reshape(values.shape[0], -1)
        output = self.predict_linear_1(output)
        output = self.relu(output)
        label = self.label_embedding_linear(retrieved_label_list)
        label = label.squeeze(1)
        output = torch.cat([output, label], dim=1)
        output = self.predict_linear_2(output)

        return output