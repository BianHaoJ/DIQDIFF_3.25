# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod
import torch as th
import torch.distributed as dist


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """
    # 时间步采样器抽象基类，用于定义扩散过程中的时间步分布，减少目标函数的方差

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """
        # 获取权重数组，每个扩散步一个权重值，不需要归一化，但必须为正数

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        # 重要性采样，为批次采样时间步
        w = self.weights() 
        p = w / np.sum(w)  # 计算概率分布
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # 根据概率分布采样时间步索引
        indices = th.from_numpy(indices_np).long().to(device)  # 转换为PyTorch张量并移到指定设备
        weights_np = 1 / (len(p) * p[indices_np])  # 计算重要性权重
        weights = th.from_numpy(weights_np).float().to(device)  # 转换为PyTorch张量并移到指定设备
        return indices, weights  # 返回采样的时间步索引和对应的权重


class UniformSampler(ScheduleSampler):
    # 均匀采样器，每个时间步的采样概率相等
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])  # 所有时间步权重相同
        
    def weights(self):
        return self._weights  # 返回均匀分布的权重


class LossAwareSampler(ScheduleSampler):
    # 损失感知采样器，根据模型损失动态调整采样权重
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        # 使用模型损失更新重加权
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        # 将all_gather批次填充到最大批次大小
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """
        # 使用模型损失更新重加权，子类需要重写此方法


class LossSecondMomentResampler(LossAwareSampler):
    # 基于损失二阶矩的重采样器
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term  # 每个时间步保留的历史损失数量
        self.uniform_prob = uniform_prob  # 均匀采样的概率
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )  # 损失历史记录
        self._loss_counts = np.zeros([self.num_timesteps], dtype=int)  # 每个时间步的损失计数

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)  # 如果还没预热完成，返回均匀权重
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))  # 计算损失的平方根（二阶矩）
        weights /= np.sum(weights)  # 归一化
        weights *= 1 - self.uniform_prob  # 调整权重
        weights += self.uniform_prob / len(weights)  # 添加均匀分布成分
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                # 移除最旧的损失项，保留最新的损失历史
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()  # 检查是否所有时间步都有足够的历史损失


class FixSampler(ScheduleSampler):
    # 固定采样器，允许自定义时间步采样权重
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        # 你可以在这里自定义时间步采样权重
        self._weights = np.concatenate([np.ones([num_timesteps//2]), np.zeros([num_timesteps//2]) + 0.5])

    def weights(self):
        return self._weights  # 返回自定义的权重


def create_named_schedule_sampler(name, num_timesteps):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    # 根据名称创建预定义的时间步采样器
    if name == "uniform":
        return UniformSampler(num_timesteps)  # 均匀采样器
    elif name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)  ## default setting  # 默认设置：损失感知采样器
    elif name == "fixstep":
        return FixSampler(num_timesteps)  # 固定采样器
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")  # 未知采样器类型
