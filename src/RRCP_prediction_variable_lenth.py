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
        batch_size = base_text_features.shape[0]
        current_node_mask = torch.ones(batch_size, 1, device=device, dtype=base_text_features.dtype)
        retrieval_mask = RRCP.to(dtype=base_text_features.dtype)
        text_mask = torch.cat([current_node_mask, retrieval_mask], dim=1)
        img_mask = text_mask.clone()

        return base_text_features, base_img_features, input_text, input_img, text_mask, img_mask

    def forward(self, mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label_list, RRCP):
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls.squeeze(2)
        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.squeeze(2)

        RRCP = RRCP[:, :self.retrieval_num]
        RRCP_binary = (RRCP > self.threshold_of_RRCP).to(torch.int32)
        RRCP[RRCP < self.threshold_of_RRCP] = 0

        zero_rows = torch.all(RRCP == 0, dim=1)
        RRCP[zero_rows, 0] = 1
        RRCP_binary[zero_rows, 0] = 1

        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:, :self.retrieval_num, :]
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls[:, :self.retrieval_num, :]
        retrieved_label_list = retrieved_label_list[:, :self.retrieval_num]

        retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls, merge_text_vec, mean_pooling_vec, text_mask, img_mask = self.preprocess_data(
            retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls, merge_text_vec,
            mean_pooling_vec, RRCP_binary)

        output = self.graph_attention(retrieved_label_list,
                                      mean_pooling_vec, merge_text_vec,
                                      retrieved_visual_feature_embedding_cls,
                                      retrieved_textual_feature_embedding, text_mask, img_mask, RRCP)

        return output
