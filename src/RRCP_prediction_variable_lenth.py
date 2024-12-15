import torch
import torch.nn as nn
from graph_attention import Model as graph_attention
import time



class RRCP_prediction(nn.Module):

    def __init__(self, retrieval_num, threshold_of_RRCP, alpha=0.5, frame_num=1, feature_dim=768):
        super(RRCP_prediction, self).__init__()
        self.retrieval_num = retrieval_num
        self.threshold_of_RRCP = threshold_of_RRCP
        self.graph_attention = graph_attention(retrieval_num, alpha, frame_num, feature_dim)

    def preprocess_data(self, base_text_features, base_img_features, input_text, input_img, RRCP):
        device = base_text_features.device
        batch_size, max_nodes, hidden_dim = base_text_features.shape
        text_mask = torch.ones(batch_size, max_nodes + 1, device=device)
        img_mask = torch.ones(batch_size, max_nodes + 1, device=device)
        max_valid_nodes = max_nodes
        base_text_features_processed = torch.zeros(batch_size, max_valid_nodes, hidden_dim, device=device)
        base_img_features_processed = torch.zeros(batch_size, max_valid_nodes, hidden_dim, device=device)
        for i in range(batch_size):
            valid_nodes = int(RRCP[i].sum().item())

            valid_indices = torch.where(RRCP[i] == 1)[0]
            base_text_features_processed[i, :valid_nodes] = base_text_features[i, valid_indices]
            text_mask[i, valid_nodes + 1:] = 0

            base_img_features_processed[i, :valid_nodes] = base_img_features[i, valid_indices]
        img_mask[i, valid_nodes + 1:] = 0

        return base_text_features_processed, base_img_features_processed, input_text, input_img, text_mask, img_mask

    def forward(self, mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label_list, RRCP):
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls.squeeze(2)
        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.squeeze(2)

        RRCP = RRCP[:, :self.retrieval_num]
        RRCP_binary = torch.where(RRCP > self.threshold_of_RRCP, torch.tensor(1.0, device=RRCP.device),
                                  torch.tensor(0.0, device=RRCP.device))
        RRCP_binary = RRCP_binary.to(torch.int32)
        RRCP[RRCP < self.threshold_of_RRCP] = 0

        zero_rows = torch.all(RRCP == 0, dim=1)
        RRCP[zero_rows, 0] = 1

        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:, :self.retrieval_num, :]
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls[:, :self.retrieval_num, :]
        retrieved_label_list = retrieved_label_list[:, :self.retrieval_num]

        retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls, merge_text_vec, mean_pooling_vec, text_mask, img_mask = self.preprocess_data(
            retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding, mean_pooling_vec,
            merge_text_vec, RRCP_binary)

        output = self.graph_attention(retrieved_label_list,
                                      mean_pooling_vec, merge_text_vec,
                                      retrieved_visual_feature_embedding_cls,
                                      retrieved_textual_feature_embedding, text_mask, img_mask, RRCP)

        return output
