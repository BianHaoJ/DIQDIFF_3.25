# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionwiseFeedForward(nn.Module):
    """位置前馈网络层，用于Transformer架构中"""
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 第一个卷积层，将输入维度从d_in转换到d_hid
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        # 第二个卷积层，将维度从d_hid转换回d_in
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        # Layer Normalization，用于稳定训练过程
        self.layer_norm = nn.LayerNorm(d_in)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播"""
        # 保存残差连接的输入
        residual = x
        # 转置输入，使卷积操作可以在正确维度上进行
        output = x.transpose(1, 2)
        # 通过第一层卷积和ReLU激活函数，然后通过第二层卷积
        output = self.w_2(F.relu(self.w_1(output)))
        # 转置回来以匹配期望的输出形状
        output = output.transpose(1, 2)
        # 应用dropout
        output = self.dropout(output)
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        return output



class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 确保隐藏层大小能被头数整除
        assert hidden_size % num_heads == 0
        
        # 线性变换层，用于生成Q、K、V向量
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # Softmax激活函数
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]  # 查询张量，形状为[N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]     # 键张量，形状为[N, T_k, C_k]
        
        :return: A 3d tensor with shape of (N, T_q, C)          # 输出张量，形状为(N, T_q, C)
        
        """
        # 通过线性层生成Q、K、V向量
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat - 分割和拼接，为多头注意力做准备
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication - 计算Q和K的点积，得到注意力分数
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking - 键掩码，屏蔽掉填充的键
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # 因果性掩码 - 防止未来信息泄露（上三角掩码）
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation - Softmax激活
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking - 查询掩码，屏蔽掉填充的查询
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout - 应用dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum - 加权求和，得到多头注意力输出
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape - 重新组合多头输出
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection - 残差连接
        output_res = output + queries
        
        return output_res
        
