import torch
import torch.nn as nn
from teacher_graph import GraphLearner

class Teacher(nn.Module):

    def __init__(self, retrieval_num ,alpha=0.5, frame_num=1, feature_dim=768, efficient=False):

        super(Teacher, self).__init__()
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.retrieval_num = retrieval_num
        self.retrieval_semantics_version = 2
        if retrieval_num == 1:
            self.query_head = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1)
            )
        self.predict_linear_1 = nn.Linear(feature_dim, feature_dim)
        self.predict_linear_2 = nn.Linear(feature_dim * 2, 1)
        self.relu = nn.ReLU()
        self.label_embedding_linear = nn.Linear(retrieval_num, feature_dim)
        self.GraphLearner = GraphLearner(device=None, hidden_dim=feature_dim, class_num=self.retrieval_num)
        self.efficient = efficient
        self.GraphLearner.GCN_tt.efficient_query_only = efficient
        self.GraphLearner.GCN_it.efficient_query_only = efficient
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8,batch_first=True)
        self.feature_attention = nn.Linear(feature_dim, 1)


    def forward_query_only(self, mean_pooling_vec, merge_text_vec):
        """A candidate-independent baseline, trained alongside the retrieved prediction."""
        if self.retrieval_num != 1:
            raise ValueError('The query-only head belongs to the single-item auxiliary model.')
        with torch.autocast(device_type=mean_pooling_vec.device.type, dtype=torch.bfloat16,
                            enabled=getattr(self, 'mixed_precision', False) and mean_pooling_vec.is_cuda):
            query = torch.cat([mean_pooling_vec.squeeze(1), merge_text_vec.squeeze(1)], dim=-1)
            return self.query_head(query)

    def forward(self, mean_pooling_vec, merge_text_vec,
                retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding, retrieved_label_list):

        with torch.autocast(device_type=mean_pooling_vec.device.type, dtype=torch.bfloat16,
                            enabled=getattr(self, 'mixed_precision', False) and mean_pooling_vec.is_cuda):
            return self._forward(mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                                 retrieved_textual_feature_embedding, retrieved_label_list)

    def _forward(self, mean_pooling_vec, merge_text_vec,
                 retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding, retrieved_label_list):

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

        output = self.multihead_attn(packed_feature, packed_feature, packed_feature,
                                    need_weights=not getattr(self, 'efficient', False))
        values = output[0]
        attention_score = self.feature_attention(values).squeeze(-1)
        attention_weight = torch.softmax(attention_score, dim=1).unsqueeze(-1)
        output = torch.sum(values * attention_weight, dim=1)
        output = self.predict_linear_1(output)
        output = self.relu(output)
        label = self.label_embedding_linear(retrieved_label_list)
        label = label.squeeze(1)
        output = torch.cat([output, label], dim=1)
        output = self.predict_linear_2(output)
        if self.retrieval_num == 1:
            output = self.forward_query_only(mean_pooling_vec, merge_text_vec) + output

        return output


def predict_single_item_delta(model, current_visual, current_textual, retrieved_visual,
                               retrieved_textual, retrieved_label, chunk_size=8192):
    if getattr(model, 'retrieval_semantics_version', None) != 2 or not hasattr(model, 'query_head'):
        raise ValueError('RRCP requires a single-neighbor teacher with a query-only head.')
    batch_size, retrieval_num = retrieved_label.shape
    # Compute once per query, independently of candidate features and labels.
    baseline = model.forward_query_only(current_visual, current_textual).reshape(batch_size, 1)
    flat_label = retrieved_label.reshape(batch_size * retrieval_num, 1)
    current_visual_flat = current_visual.unsqueeze(1).expand(
        -1, retrieval_num, *current_visual.shape[1:]
    ).reshape(batch_size * retrieval_num, *current_visual.shape[1:])
    current_textual_flat = current_textual.unsqueeze(1).expand(
        -1, retrieval_num, *current_textual.shape[1:]
    ).reshape(batch_size * retrieval_num, *current_textual.shape[1:])
    retrieved_visual_flat = retrieved_visual.reshape(batch_size * retrieval_num, *retrieved_visual.shape[2:])
    retrieved_textual_flat = retrieved_textual.reshape(batch_size * retrieval_num, *retrieved_textual.shape[2:])

    labels_with = []
    for start in range(0, flat_label.size(0), chunk_size):
        end = start + chunk_size
        label_chunk = flat_label[start:end]
        current_visual_chunk = current_visual_flat[start:end]
        current_textual_chunk = current_textual_flat[start:end]

        labels_with.append(
            model(current_visual_chunk, current_textual_chunk,
                  retrieved_visual_flat[start:end], retrieved_textual_flat[start:end],
                  label_chunk).squeeze(-1)
        )

    return (
        baseline.expand(-1, retrieval_num),
        torch.cat(labels_with, dim=0).reshape(batch_size, retrieval_num),
    )

def load_teacher(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Teacher(**checkpoint['config']).to(device)
    model.mixed_precision = checkpoint['mixed_precision']
    model.dataset_manifest_sha256 = checkpoint.get('dataset_manifest_sha256')
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()
