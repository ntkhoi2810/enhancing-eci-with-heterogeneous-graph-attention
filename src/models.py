from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn


class ClozeAnalyzer(nn.Module):
    def __init__(self, tokenizer, bert, device, visualize = False):
        super(ClozeAnalyzer, self).__init__()
        self.tokenizer = tokenizer
        self.bert = bert
        self.visualize = visualize
        self.device = device
        
    def forward(self, x, groundtruth):

        token_logits = self.bert(**x).logits
        # mask_token_index = (x['input_ids'] == self.tokenizer.mask_token_id).int().argmax(dim=1)
        # mask_token_logits = torch.stack( 
        #     [ token_logits[idx, mask_token_index[idx], :] for idx in range(x['input_ids'].size(0))], 
        #     dim=0)
        batch_size = x['input_ids'].size(0)
        batch_indices = torch.arange(batch_size, device=self.device)
        mask_token_index = (x['input_ids'] == self.tokenizer.mask_token_id).int().argmax(dim=1)
        
        mask_token_logits = token_logits[batch_indices, mask_token_index, :]

        # predicted_token_ids = torch.argmax(mask_token_logits, dim=-1)
    
        # new_input_ids = x['input_ids'].clone()
        # new_input_ids[batch_indices, mask_token_index] = predicted_token_ids
        
        # outputs = self.bert.base_model(
        #     input_ids=new_input_ids,
        #     attention_mask=x['attention_mask']
        # ).last_hidden_state

        soft_probs = torch.softmax(mask_token_logits, dim=-1) 

        word_embeddings = self.bert.get_input_embeddings().weight
        predicted_embeds = torch.matmul(soft_probs, word_embeddings)

        inputs_embeds = self.bert.get_input_embeddings()(x['input_ids'])
        inputs_embeds[batch_indices, mask_token_index] = predicted_embeds.to(inputs_embeds.dtype)

        outputs = self.bert.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=x['attention_mask']
        ).last_hidden_state
                
        ret = outputs[batch_indices, mask_token_index, :].unsqueeze(1)
        
        return ret
        
        # gen_token = [ self.tokenizer.decode(token_id) for token_id in torch.argmax(mask_token_logits, dim=-1) ]
        # og_mask_sentences = [ self.tokenizer.decode(sequence[1:-1]) for sequence in x['input_ids'] ] 
        # gen_sentences = [sentence.replace(self.tokenizer.mask_token, gen_token[idx]) for idx, sentence in enumerate( og_mask_sentences)]
        # if self.visualize:
        #     og_sentences=[ self.tokenizer.decode(sequence[1:-1]) for sequence in groundtruth['input_ids'] ] 
        #     print(f"generated tokens: {gen_token}")
        #     print(f"original sentences: {og_sentences}")
        #     print(f"original mask sentences: {og_mask_sentences}")
        #     print(f"generated sentences: {gen_sentences}")

        # outputs = self.bert.base_model(
        #     **self.tokenizer(gen_sentences, return_tensors="pt", padding=True).to(self.device)
        #     ).last_hidden_state
        
        # ret = torch.stack( 
        #     [outputs[idx, mask_token_index[idx], :] for idx in range(x['input_ids'].size(0))], 
        #     dim=0).unsqueeze(1)
        
        # return ret 
        

class Discriminator(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, tokenizer, bert, device):
        super(Discriminator, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.tokenizer = tokenizer
        self.bert = bert
        self.device = device
        # FFN
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.fc3 = nn.Linear(d_model, 2)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, groundtruth):
        
        key = self.bert.base_model(**groundtruth).last_hidden_state.permute(1, 0, 2)
        value =key
        x = x.permute(1, 0, 2)
        attn_output, attn_weights = self.mha(x, key, value)
        attn_output = attn_output.permute(1, 0, 2)

        # FFN
        out=self.dropout(self.relu(self.fc1(attn_output)))
        out=self.fc2(out)
        out=self.layer_norm(attn_output+out)
        out=self.fc3(out) 
        
        return out
      


# class Causal_Model(nn.Module):
#     def __init__(self, bert_path, d_model, num_heads, dropout_rate, device, special_tokens=None, visualize=False):
#         super(Causal_Model, self).__init__()
        
#         self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
#         special_tokens_dict = {'additional_special_tokens': special_tokens}
#         self.tokenizer.add_special_tokens(special_tokens_dict)
#         self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
#         self.bert.resize_token_embeddings(len(self.tokenizer))
        
#         self.generator = ClozeAnalyzer(self.tokenizer, self.bert, device, visualize)
#         self.discriminator = Discriminator(d_model, num_heads, dropout_rate, self.tokenizer, self.bert, device)
        

#     def forward(self, x, groundtruth):
        
#         out=self.generator(x, groundtruth) 
#         out=self.discriminator(out, groundtruth)
#         return out

class Causal_Model(nn.Module):
    def __init__(self, bert_path, d_model, num_heads, dropout_rate, device, special_tokens=None, visualize=False):
        super(Causal_Model, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        
        self.generator = ClozeAnalyzer(self.tokenizer, self.bert, device, visualize)
        self.discriminator = Discriminator(d_model, num_heads, dropout_rate, self.tokenizer, self.bert, device)
        
        # ---> MỚI: Thiết lập Mạng Heterogeneous Graph Attention Network (HAN)
        self.device = device
        # Định nghĩa các loại Node và Cạnh (Metadata cho HANConv)
        self.metadata = (
            ['word'], 
            [('word', 'nsubj', 'word'), ('word', 'prep', 'word'), 
             ('word', 'pobj', 'word'), ('word', 'dobj', 'word'), 
             ('word', 'amod', 'word'), ('word', 'ROOT', 'word'), ('word', 'other', 'word')]
        )
        
        # d_model // 2 vì lát nữa ta sẽ nối (concat) e1 và e2 -> size d_model
        self.han = HANConv(
            in_channels=d_model, 
            out_channels=d_model // 2, 
            metadata=self.metadata, 
            heads=2,
            dropout=dropout_rate
        )
        
        # Lớp để Fusion (Kết hợp Output của BERT Mask + Output của Mạng Đồ thị)
        # d_model (Cloze) + d_model (HAN) = d_model * 2
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x, groundtruth, graph_data):
        """
        graph_data: dictionary chứa các features về đồ thị theo batch (được map từ src/data.py)
        - graph_data['edges']: List các dictionary mô tả edge_index_dict cho HANConv
        - graph_data['e1_idx'], graph_data['e2_idx']: Vị trí node của e1, e2
        """
        
        # 1. Output gốc từ ClozeAnalyzer (Shape: [batch_size, 1, d_model])
        cloze_out = self.generator(x, groundtruth) 
        
        # 2. Xử lý Đồ thị ngữ pháp (Syntax-Aware Graph)
        # Lấy base features từ BERT làm features ban đầu cho các Nodes
        bert_features = self.bert.base_model(**groundtruth).last_hidden_state
        batch_size = cloze_out.size(0)
        
        e_graph_reps = []
        
        for i in range(batch_size):
            # Tạo dictionary input cho HANConv
            x_dict = {'word': bert_features[i]} 
            
            # Khôi phục đồ thị của câu hiện tại dạng PyTorch Tensors
            edges = graph_data['edges'][i]
            edge_index_dict = {
                rel: torch.tensor(indices, dtype=torch.long).to(self.device)
                for rel, indices in edges.items()
            }
            
            # Forward qua Mạng Đồ Thị (HAN)
            han_out = self.han(x_dict, edge_index_dict)
            word_embs = han_out['word'] # Node embeddings đã mang thông tin ngữ pháp
            
            # Trích xuất biểu diễn của Node e1 và Node e2
            idx_1 = min(graph_data['e1_idx'][i], word_embs.size(0) - 1)
            idx_2 = min(graph_data['e2_idx'][i], word_embs.size(0) - 1)
            
            e1_emb = word_embs[idx_1] # Shape: [d_model // 2]
            e2_emb = word_embs[idx_2] # Shape: [d_model // 2]
            
            # Nối (Concatenate) 2 events để tạo vector quan hệ qua GNN
            e_pair = torch.cat([e1_emb, e2_emb], dim=-1).unsqueeze(0) # [1, d_model]
            e_graph_reps.append(e_pair)
            
        e_graph_reps = torch.stack(e_graph_reps, dim=0) # [batch_size, 1, d_model]
        
        # 3. MỚI: Feature Fusion
        # Nối đặc trưng của Cloze Text và HAN Graph
        fused_features = torch.cat([cloze_out, e_graph_reps], dim=-1) # [batch_size, 1, d_model * 2]
        
        # Nén lại về chiều d_model và thêm Residual Connection
        fused_features = self.relu(self.fusion_fc(fused_features))
        fused_features = self.layer_norm(cloze_out + fused_features) # Identity mapping với raw cloze
        
        # 4. Đưa vector đã dung hợp vào Discriminator
        out = self.discriminator(fused_features, groundtruth)
        
        return out