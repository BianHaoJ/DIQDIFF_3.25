# -*- coding: utf-8 -*-
import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from Modules import *




class CoDiffu(nn.Module):
    """
   基于扩散模型的序列推荐模型 (Diffusion-based Sequential Recommendation Model)
   该模型结合了扩散模型和意图聚类来提高序列推荐的准确性

   CoDiffu: Conditional Diffusion Model for Sequential Recommendation
   """
    def __init__(self, args):
        super(CoDiffu, self).__init__()
        # 模型维度参数
        self.hidden_size = args.hidden_size  # 隐藏层维度
        self.schedule_sampler_name = args.schedule_sampler_name  # 采样策略名称
        self.diffusion_steps = args.diffusion_steps  # 扩散步数

        # 扩散模型核心参数
        self.noise_schedule = args.noise_schedule  # 噪声调度方案
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)  # 获取beta值
        # Use float64 for accuracy.
        # 使用float64确保精度
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"  # 确保beta是一维数组
        assert (betas > 0).all() and (betas <= 1).all()  # 确保beta在(0,1]范围
        alphas = 1.0 - betas   # 计算alpha值
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # alpha的累积乘积
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # 前一个alpha累积乘积

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 用于扩散过程的计算 q(x_t | x_{t-1}) 等
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  # sqrt(alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # sqrt(1-alpha_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 扩散过程计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)   # sqrt(alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # sqrt(1-alpha_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  # sqrt(1/alpha_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # sqrt(1/alpha_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 后验分布均值系数
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])  # 时间步总数

        # 创建时间步采样器，用于训练时选择时间步
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)

        self.rescale_timesteps = args.rescale_timesteps  # 是否重新缩放时间步
        # 时间嵌入层，将时间步信息编码为向量
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4), SiLU(), nn.Linear(self.hidden_size * 4, self.hidden_size))
        # Transformer编码器，用于序列建模
        self.att = Transformer_rep(args)
        # Transformer编码器，用于序列建模
        self.lambda_history = args.lambda_history  # 历史序列权重
        self.lambda_intent = args.lambda_intent  # 意图权重
        self.dropout = nn.Dropout(args.dropout)  # dropout层
        self.norm_diffu_rep = LayerNorm(self.hidden_size)  # 扩散表示的层归一化
        self.item_num = args.item_num+1  # 项目数量（+1用于padding）
        # 项目嵌入层
        self.item_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.embed_dropout = nn.Dropout(args.emb_dropout)  # 嵌入层的dropout
        # self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        # 层归一化
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # 交叉熵损失函数
        self.loss_ce = nn.CrossEntropyLoss()
        # self.encoder=SASRecModel(args,self.item_num)
        # 聚类参数
        self.n_clusters = int(args.num_cluster)   # 聚类数量
        self.n_iters = int(args.num_iter)  # 聚类迭代次数
        # 用于意图聚类的MLP
        self.mlp= nn.Sequential(nn.Linear(self.hidden_size*args.max_len, int(self.n_clusters)))
        # self.centroids = torch.zeros(self.n_clusters,args.max_len,args.hidden_size).cuda()

        

        
      
       
          
      
        # self.mlp= nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size))
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        """
        创建正弦时间步嵌入 (Create sinusoidal timestep embeddings)
        
        Args:
            timesteps: 一维张量，包含N个索引，每个批次元素一个
            dim: 输出维度
            max_period: 控制最小频率的参数
            
        Returns:
            [N x dim] 的位置嵌入张量
        """

        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        """
        对给定数量的扩散步骤进行数据扩散
        
        换句话说，从 q(x_t | x_0) 中采样
        
        Args:
            x_start: 初始数据批次
            t: 扩散步数（减1）。0表示一步
            noise: 指定的高斯噪声
            mask: 锚定掩码位置
            
        Returns:
            x_start 的噪声版本
        """

        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t


    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output= self.denoise(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        
        x_0 = model_output  ##output predict
        # model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t 
    def denoise(self,item_rep, x_t, t, mask_seq):
        """
        去噪函数，用于预测原始信号

        Args:
            item_rep: 项目表示
            x_t: 带噪声的输入
            t: 时间步
            mask_seq: 序列掩码

        Returns:
            去噪后的表示
        """
        # 获取时间步嵌入
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t   # 将时间信息加入噪声数据
        # 结合历史表示、噪声和意图表示进行注意力计算
        # item_rep*self.lambda_history: 历史序列表示加权
        # 0.001 * x_t.unsqueeze(1): 噪声项（加小权重以避免数值问题）
        # self.centroids[self.labels]*self.lambda_intent: 意图表征加权
        res= self.att(item_rep*self.lambda_history + 0.001 * x_t.unsqueeze(1)+self.centroids[self.labels]*self.lambda_intent, mask_seq)
        res= self.norm_diffu_rep(self.dropout(res))  # 归一化和dropout
       
        # out=self.mlp(x_t)
        out=res[:, -1, :]  # 取序列最后一个位置的表示作为输出
        return out
    


    def diffu(self, item_rep, item_tag, mask_seq):        
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler
        x_t = self.q_sample(item_tag, t, noise=noise)      
        x_0 = self.denoise(item_rep, x_t, self._scale_timesteps(t), mask_seq) ##output predict
        return x_0
    
    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        contra=self.contra_loss(rep_diffu)
        return self.loss_ce(scores, labels.squeeze(-1)),contra
    
    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    def intent_cluster(self,input,n_clusters):
        """
        意图聚类函数，用于发现用户的不同意图

        Args:
            input: 输入序列表示
            n_clusters: 聚类数量

        Returns:
            聚类中心和聚类标签
        """
        X=input.view(input.shape[0],-1)  # 将输入展平
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)  # 随机初始化聚类中心
        labels=self.mlp(X)  # 通过MLP获取初始标签
        labels = F.gumbel_softmax(labels, tau=0.1, hard=True)  # 使用Gumbel-Softmax获取离散标签
        # labels = torch.argmax(labels, dim=1)
        # K-means聚类迭代
        for i in range(self.n_clusters):
            if torch.sum(labels[:, i]) == 0:  # 如果某个聚类为空
                centers[i] = X[torch.randint(0, X.size(0), (1,))]  # 随机选择一个点作为中心
            else:
                centers[i] = torch.mean(X[labels[:, i].bool()], dim=0)  # 计算聚类中心
        return centers.view(n_clusters,input.shape[1],input.shape[-1]), torch.argmax(labels, dim=1)

    def contra_loss(self,diffu):
        """
        对比学习损失，用于增强表示的区分性

        Args:
            diffu: 扩散模型的输出表示

        Returns:
            对比学习损失值
        """
        temperature=0.07
        # 计算表示间的余弦相似度
        similarities = (F.cosine_similarity(diffu.unsqueeze(1), diffu.unsqueeze(0), dim=2) / temperature).to(diffu.device)
        
        # Positive pairs
        # 正样本对（对角线元素）
        positives = torch.diag(similarities).to(diffu.device)
        
        # Negative pairs
        # 负样本对
        mask = torch.eye(len(diffu)).bool().to(diffu.device)
        mask = ~mask.to(diffu.device)
        negatives = similarities[mask].view(len(diffu), -1).to(diffu.device)
        
        # Calculate NCE Loss
        # 计算NCE损失
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(len(diffu), dtype=torch.long).to(diffu.device)
        loss = F.cross_entropy(logits, labels)        
        return loss


    def forward(self, sequence, tag, train_flag):
        """
        前向传播函数

        Args:
            sequence: 输入序列
            tag: 目标项目
            train_flag: 是否为训练模式

        Returns:
            推荐表示和聚类中心
        """
        # 获取项目嵌入
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        item_embeddings = self.LayerNorm(item_embeddings)  # 层归一化
        # input=self.encoder(sequence)[:,-1,:]
        # self.centroids, self.labels = KMeans(item_embeddings, self.n_clusters, self.n_iters)
        # 执行意图聚类，获取聚类中心和标签
        self.centroids, self.labels = self.intent_cluster(item_embeddings, self.n_clusters)

        mask_seq = (sequence>0).float()  # 序列掩码
        
        if train_flag:  # 训练模式
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H  # 目标项目嵌入
            rep_diffu= self.diffu(item_embeddings, tag_emb, mask_seq)  # 扩散过程
          
        else:  # 推理模式
            noise_x_t = th.randn_like(item_embeddings[:,-1,:]) # 随机噪声
            rep_diffu = self.reverse_p_sample(item_embeddings, noise_x_t, mask_seq)  # 反向采样

        return rep_diffu,self.centroids  # 返回推荐表示和聚类中心




