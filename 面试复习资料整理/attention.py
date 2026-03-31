import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    在MHA中，每个注意力头都有自己独立的Query、Key和Value投影矩阵
    
    公式:
    Attention(Q, K, V) = softmax(Q·K^T/√d_k)·V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)·W^O
    where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保模型维度可以被头数整除
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)  # Query投影
        self.k_proj = nn.Linear(d_model, d_model)  # Key投影
        self.v_proj = nn.Linear(d_model, d_model)  # Value投影
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性投影并分割成多个头
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置以便进行注意力计算
        # 形状: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        # 缩放点积注意力: Attention(Q, K, V) = softmax(QK^T/√d_k)V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到values
        out = torch.matmul(attn_weights, v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out


class MQA(nn.Module):
    """
    多查询注意力机制 (Multi-Query Attention)
    
    在MQA中，所有头共享相同的Key和Value投影矩阵，但每个头有自己独立的Query投影矩阵
    
    公式与MHA相似，但是Key和Value投影矩阵在所有头之间共享：
    K_shared = K·W^K
    V_shared = V·W^V
    head_i = Attention(Q·W_i^Q, K_shared, V_shared)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保模型维度可以被头数整除
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # Query有多个头，但Key和Value只有一个头
        self.q_proj = nn.Linear(d_model, d_model)  # 多个Query投影
        self.k_proj = nn.Linear(d_model, self.head_dim)  # 单个Key投影
        self.v_proj = nn.Linear(d_model, self.head_dim)  # 单个Value投影
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len, kv_len = q.size(1), k.size(1)
        
        # 对Query进行多头投影
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim)
        
        # Key和Value只有一个共享投影
        # 形状: (batch_size, seq_len, head_dim)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 转置Query以便进行注意力计算
        # 形状: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        
        # 扩展Key和Value以适应多头格式
        # 形状: (batch_size, 1, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        k = k.unsqueeze(1).expand(batch_size, self.num_heads, kv_len, self.head_dim)
        v = v.unsqueeze(1).expand(batch_size, self.num_heads, kv_len, self.head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到values
        out = torch.matmul(attn_weights, v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out


class GQA(nn.Module):
    """
    分组查询注意力机制 (Grouped-Query Attention)
    
    在GQA中，将注意力头分成几个组，每组共享相同的Key和Value投影矩阵
    
    这是MHA和MQA之间的折中方案：
    - 分组数量 = 1 时，等同于MQA
    - 分组数量 = num_heads 时，等同于MHA
    """
    def __init__(self, d_model, num_heads, num_groups, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.num_groups = num_groups  # KV组的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保条件满足
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        assert num_heads % num_groups == 0, "num_heads必须能被num_groups整除"
        
        self.heads_per_group = num_heads // num_groups  # 每组的头数量
        
        # 投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)  # Query投影 (所有头)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_groups)  # Key投影 (每组一个)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_groups)  # Value投影 (每组一个)
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len, kv_len = q.size(1), k.size(1)
        
        # 对Query进行多头投影
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim)
        
        # 对Key和Value进行分组投影
        # 形状: (batch_size, seq_len, num_groups * head_dim)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 调整Key和Value的形状以适应分组
        # 形状: (batch_size, seq_len, num_groups, head_dim)
        k = k.view(batch_size, kv_len, self.num_groups, self.head_dim)
        v = v.view(batch_size, kv_len, self.num_groups, self.head_dim)
        
        # 转置形状以便计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_groups, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_groups, kv_len, head_dim)
        
        # 现在需要将q与对应组的k和v匹配
        # 我们需要为每个query头找到对应的key/value组
        
        # 初始化输出张量
        out = torch.zeros(batch_size, self.num_heads, q_len, self.head_dim, device=q.device)
        
        # 对每个组计算注意力
        for i in range(self.num_groups):
            # 获取当前组的queries
            start_idx = i * self.heads_per_group
            end_idx = (i + 1) * self.heads_per_group
            group_q = q[:, start_idx:end_idx]
            
            # 获取当前组的key和value
            group_k = k[:, i:i+1].expand(-1, self.heads_per_group, -1, -1)  # 复制以匹配num_heads
            group_v = v[:, i:i+1].expand(-1, self.heads_per_group, -1, -1)  # 复制以匹配num_heads
            
            # 计算注意力分数
            scores = torch.matmul(group_q, group_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用mask（如果提供）
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # 应用softmax获得注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力权重到values
            out[:, start_idx:end_idx] = torch.matmul(attn_weights, group_v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out

class MultiheadLatentAttention(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()
        
        # 基本参数设置
        self.n_heads = args.n_heads # 头数=128
        self.d_k = args.d_k  # 非RoPE部分的维度 (128)
        self.d_r = args.d_r  # RoPE部分的维度 (64)
        self.d_c = args.d_c  # 低秩投影维度 (512)
        self.d_c_prime = args.d_c_prime  # Q的低秩投影维度 (1536)
        self.d_v = args.d_v  # V的输出维度 (128)

        
        
        # 定义投影矩阵
        # 低秩投影 d_c_prime * d_c
        self.W_c = nn.Linear(args.dim, self.d_c, bias=False)  # W_c
        # 低秩投影 d_c_prime * d_c_prime
        self.W_c_prime = nn.Linear(args.dim, self.d_c_prime, bias=False)  # W_c'
        
        # Q的投影矩阵 d_c_prime * d_k
        self.W_qc = nn.ModuleList([
            nn.Linear(self.d_c_prime, self.d_k, bias=False) 
            for _ in range(self.n_heads)
        ])  # W_qc^(s)
        # Q的RoPE投影矩阵 d_c_prime * d_r
        self.W_qr = nn.ModuleList([
            nn.Linear(self.d_c_prime, self.d_r, bias=False)
            for _ in range(self.n_heads)
        ])  # W_qr^(s)
        
        # K的投影矩阵 d_c * d_k
        self.W_kc = nn.ModuleList([
            nn.Linear(self.d_c, self.d_k, bias=False)
            for _ in range(self.n_heads)
        ])  # W_kc^(s)
        # K的RoPE投影矩阵 d_c * d_r
        self.W_kr = nn.Linear(args.dim, self.d_r, bias=False)  # W_kr
        
        # V的投影矩阵 d_c * d_v
        self.W_v = nn.ModuleList([
            nn.Linear(self.d_c, self.d_v, bias=False)
            for _ in range(self.n_heads)
        ])  # W_v^(s)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        # KV缓存
        self.c_cache = None  # 低秩投影后的缓存
        self.x_cache = None  # 原始输入的缓存
        
        # 注意力掩码
        max_seq_len = args.max_seq_len
        attn_mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        self.register_buffer("attn_mask", torch.triu(attn_mask, diagonal=1), persistent=False)

    def precompute_matrices(self):
        # 推理阶段使用的预计算矩阵, W_qc^(s) * W_kc^(s)
        self.merged_W_qc_kc = [
            torch.matmul(self.W_qc[i].weight, self.W_kc[i].weight.t())
            for i in range(self.n_heads)
        ]

    def forward(self, x, rotary_emb=None, kv_cache=False):
        batch_size, seq_len, _ = x.shape
        
        # 低秩投影
        c = self.W_c(x)  # [batch_size, seq_len, d_c]
        c_prime = self.W_c_prime(x)  # [batch_size, seq_len, d_c_prime]
        
        # 缓存处理
        if kv_cache and not self.training:
            has_cache = self.c_cache is not None and self.x_cache is not None
            if seq_len == 1 and has_cache:
                c = torch.cat((self.c_cache, c), dim=1)
                x = torch.cat((self.x_cache, x), dim=1)
            self.c_cache = c
            self.x_cache = x
            
        outputs = []
        for head in range(self.n_heads):
            if not self.training:  # 推理模式
                # 使用预计算的矩阵计算非RoPE部分的注意力得分
                q_c = torch.matmul(c_prime, self.merged_W_qc_kc[head])  # [batch_size, seq_len, d_c]
                # 注意: 这里直接使用c而不是k_c，因为我们已经预计算了W_q和W_k的乘积
                k_c = c
            else:  # 训练模式
                # 计算Q和K的非RoPE部分
                q_c = self.W_qc[head](c_prime)  # c'W_qc^(s)
                k_c = self.W_kc[head](c)  # cW_kc^(s)
            
            # 计算RoPE部分
            q_r = self.W_qr[head](c_prime)  # c'W_qr^(s)
            k_r = self.W_kr(x)  # xW_kr
            
            # 应用RoPE
            if rotary_emb is not None:
                q_r = apply_rope(q_r, rotary_emb)  # c'W_qr^(s)R_i
                k_r = apply_rope(k_r, rotary_emb)  # xW_krR_i
            
            # 拼接Q和K的两个部分
            q = torch.cat([q_c, q_r], dim=-1)  # [batch_size, seq_len, d_k + d_r]
            k = torch.cat([k_c, k_r], dim=-1)  # [batch_size, seq_len, d_k + d_r]
            
            # 计算注意力分数
            scale = 1.0 / math.sqrt(self.d_k + self.d_r)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # 添加注意力掩码
            attn_scores = attn_scores + self.attn_mask[:, :, :seq_len, :seq_len]
            
            # 计算注意力概率
            attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(q_c)
            attn_probs = self.attn_dropout(attn_probs)
            
            # 计算V和输出
            v = self.W_v[head](c)  # cW_v^(s)
            head_output = torch.matmul(attn_probs, v)
            
            outputs.append(head_output)
        
        # 拼接所有头的输出
        output = torch.cat(outputs, dim=-1)
        return self.resid_dropout(output)