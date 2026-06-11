from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClozeAnalyzer(nn.Module):
    def __init__(self, tokenizer, bert, device, visualize=False):
        super(ClozeAnalyzer, self).__init__()
        self.tokenizer = tokenizer
        self.bert = bert
        self.visualize = visualize
        self.device = device
        
    def forward(self, x, groundtruth):
        # 1. Trích xuất ngữ cảnh qua cơ chế điền khuyết (Fill-in token embedding)
        batch_size = x['input_ids'].size(0)
        batch_indices = torch.arange(batch_size, device=self.device)
        mask_token_index = (x['input_ids'] == self.tokenizer.mask_token_id).int().argmax(dim=1)
        
        token_logits = self.bert(**x).logits
        mask_token_logits = token_logits[batch_indices, mask_token_index, :]
        soft_probs = torch.softmax(mask_token_logits, dim=-1) 

        word_embeddings = self.bert.get_input_embeddings().weight
        predicted_embeds = torch.matmul(soft_probs, word_embeddings)

        inputs_embeds = self.bert.get_input_embeddings()(x['input_ids'])
        inputs_embeds[batch_indices, mask_token_index] = predicted_embeds.to(inputs_embeds.dtype)

        x_outputs = self.bert.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=x['attention_mask']
        ).last_hidden_state
                
        cloze_feature = x_outputs[batch_indices, mask_token_index, :] # [B, d_model]
        
        # 2. Trích xuất ma trận đặc trưng ngữ cảnh động từ chuỗi nguyên bản "groundtruth"
        gt_outputs = self.bert.base_model(**groundtruth).last_hidden_state # [B, seq_len, d_model]
        
        return cloze_feature, gt_outputs


class NodeAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super(NodeAttention, self).__init__()
        self.W = nn.Linear(in_size, out_size)
        self.attn = nn.Linear(out_size * 2, 1)

    def forward(self, h, adj):
        # h: [B, N, D]
        Wh = self.W(h) 
        a1 = self.attn.weight[:, :Wh.shape[-1]]
        a2 = self.attn.weight[:, Wh.shape[-1]:]
        attn_i = torch.matmul(Wh, a1.T)
        attn_j = torch.matmul(Wh, a2.T)
        e = F.leaky_relu(attn_i + attn_j.transpose(1, 2)) 
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        return torch.bmm(attention, Wh)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # z: [B, num_meta_paths, D]
        w = self.project(z).mean(0) # [num_meta_paths, 1]
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape) # [B, num_meta_paths, 1]
        return (beta * z).sum(1) # [B, D]

class HeterogeneousGraphAttentionNetwork(nn.Module):
    def __init__(self, in_size, out_size, num_meta_paths=2):
        super(HeterogeneousGraphAttentionNetwork, self).__init__()
        # Chú ý cấp nút (Node-level attention)
        self.node_attentions = nn.ModuleList([NodeAttention(in_size, out_size) for _ in range(num_meta_paths)])
        # Chú ý cấp ngữ nghĩa (Semantic-level attention)
        self.semantic_attention = SemanticAttention(out_size)

    def forward(self, h, graph_data=None, input_ids=None, e1_id=None, e2_id=None):
        batch_size, seq_len, _ = h.shape
        
        # Nếu chưa có cấu trúc đồ thị từ data loader, tạo các ma trận kề mặc định (self-loop và fully connected)
        if graph_data is None or 'adjs' not in graph_data:
            adjs = [
                torch.eye(seq_len, device=h.device).unsqueeze(0).expand(batch_size, -1, -1),
                torch.ones((seq_len, seq_len), device=h.device).unsqueeze(0).expand(batch_size, -1, -1)
            ]
        else:
            adjs = graph_data['adjs']

        semantic_embeddings = []
        for i, node_attn in enumerate(self.node_attentions):
            z = node_attn(h, adjs[i]) # Quá trình lan truyền đa hop dọc theo cạnh
            
            # Trích xuất đặc trưng của 2 sự kiện e1 và e2
            if input_ids is not None and e1_id is not None and e2_id is not None:
                batch_indices = torch.arange(batch_size, device=h.device)
                
                e1_indices = (input_ids == e1_id).int().argmax(dim=1)
                e2_indices = (input_ids == e2_id).int().argmax(dim=1)
                
                e1_repr = z[batch_indices, e1_indices]
                e2_repr = z[batch_indices, e2_indices]
                
                # Tính trung bình cộng đặc trưng 2 event
                z_pooled = (e1_repr + e2_repr) / 2.0
            else:
                z_pooled = z.mean(dim=1) 
                
            semantic_embeddings.append(z_pooled)
            
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1) # [B, num_meta_paths, out_size]
        return self.semantic_attention(semantic_embeddings) # [B, out_size]


class FeatureFusionLayer(nn.Module):
    def __init__(self, text_dim, graph_dim, out_dim):
        super(FeatureFusionLayer, self).__init__()
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.graph_proj = nn.Linear(graph_dim, out_dim)
        
        # Cơ chế cổng điều tiết (Gating mechanism)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + graph_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, text_feat, graph_feat):
        t_proj = self.text_proj(text_feat)
        g_proj = self.graph_proj(graph_feat)
        gate_val = self.gate(torch.cat([text_feat, graph_feat], dim=-1))
        
        # Hợp nhất tương tác bằng cơ chế gating
        fused = gate_val * t_proj + (1 - gate_val) * g_proj
        return fused


class Discriminator(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(Discriminator, self).__init__()
        # Mạng neural truyền thẳng đa lớp kết hợp với kích hoạt phi tuyến (MLP)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, fused_feature):
        # Bộ trọng tài xử lý tương tác nhân quả cấp cao
        out = self.net(fused_feature)
        return out


class Causal_Model(nn.Module):
    def __init__(self, bert_path, d_model, num_heads, dropout_rate, device, special_tokens=None, visualize=False):
        super(Causal_Model, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids('<e1>')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('<e2>')
        
        # Bốn giai đoạn phối hợp
        self.cloze_analyzer = ClozeAnalyzer(self.tokenizer, self.bert, device, visualize)
        self.han = HeterogeneousGraphAttentionNetwork(in_size=d_model, out_size=d_model, num_meta_paths=2)
        self.feature_fusion = FeatureFusionLayer(text_dim=d_model, graph_dim=d_model, out_dim=d_model)
        self.discriminator = Discriminator(d_model, dropout_rate)
        

    def forward(self, x, groundtruth, graph_data=None):
        # 1. Khối trích xuất ngữ cảnh
        cloze_feature, gt_outputs = self.cloze_analyzer(x, groundtruth) 
        
        # 2. Mạng đồ thị dị thể (HAN)
        graph_feature = self.han(gt_outputs, graph_data, groundtruth['input_ids'], self.e1_id, self.e2_id)
        
        # 3. Tầng dung hợp đặc trưng
        fused_feature = self.feature_fusion(cloze_feature, graph_feature)
        
        # 4. Khối phân biệt tương tác
        out = self.discriminator(fused_feature)
        
        # (Softmax được áp dụng bên ngoài khi tính Loss/Metrics)
        return out