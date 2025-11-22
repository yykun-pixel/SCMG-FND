import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class MultiGranularityContrast(nn.Module):
    """多粒度对比学习模块
    
    该模块实现了针对多模态数据的多粒度对比学习损失，通过以下三个层次的对比:
    1. 全局对比: 对整体特征表示进行对比学习
    2. 时空对比: 对分割为时空区域的特征进行对比学习
    3. 模态对比: 对不同模态的特征分量进行对比学习
    
    这种多粒度设计帮助模型在不同层次上学习更强大的表示，增强模型对多模态数据的理解能力。
    """
    
    def __init__(self, 
                 feature_dim=128, 
                 projection_dim=64, 
                 temperature=0.1,
                 spatial_regions=4,
                 temporal_segments=8,
                 modal_components=3):
        """初始化多粒度对比学习模块
        
        Args:
            feature_dim: 输入特征维度
            projection_dim: 投影空间维度
            temperature: 温度参数，控制对比学习的敏感度
            spatial_regions: 空间区域的数量
            temporal_segments: 时间段的数量
            modal_components: 模态组件的数量
        """
        super(MultiGranularityContrast, self).__init__()
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        # 改进4：自适应温度参数 - 为不同粒度使用不同温度参数
        self.temperatures = {
            'global': temperature,
            'modal': temperature * 0.7,  # 模态对比使用较小温度
            'temporal': temperature * 0.5,  # 时间对比使用更小温度
            'spatial': temperature * 0.8   # 空间对比使用中等温度
        }
        self.spatial_regions = spatial_regions
        self.temporal_segments = temporal_segments
        self.modal_components = modal_components
        
        # 全局特征适配器 - 用于处理不同维度的输入
        self.global_feature_adapter = None  # 懒加载，只在需要时创建
        
        # 全局特征投影
        self.global_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # 模态特征投影
        self.modal_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # 时间特征投影
        self.temporal_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # 空间特征投影
        self.spatial_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
    def forward(self, text_features, audio_features, video_features, global_features, labels):
        """前向传播过程
        
        Args:
            text_features: 文本特征，形状为 [B, D]
            audio_features: 音频特征，形状为 [B, T, D]
            video_features: 视频特征，形状为 [B, T, S, D]，其中S是空间维度
            global_features: 全局特征，形状为 [B, D]
            labels: 标签，形状为 [B]
            
        Returns:
            对比损失的加权总和
        """
        batch_size = text_features.size(0)
        device = text_features.device
        
        # 1. 全局对比损失
        global_loss = self._compute_global_loss(global_features, labels)
        
        # 2. 模态对比损失
        modal_loss = self._compute_modal_loss(text_features, audio_features, video_features, labels)
        
        # 3. 时间对比损失（使用音频和视频的时间维度）
        temporal_loss = self._compute_temporal_loss(audio_features, video_features, labels)
        
        # 4. 空间对比损失（使用视频的空间维度）
        spatial_loss = self._compute_spatial_loss(video_features, labels)
        
        # 改进1：损失函数权重平衡 - 使用加权平均替代简单相加
        total_loss = self._compute_weighted_loss(global_loss, modal_loss, temporal_loss, spatial_loss)
        
        return total_loss
    
    def _compute_weighted_loss(self, global_loss, modal_loss, temporal_loss, spatial_loss):
        """改进1：计算加权损失，平衡不同粒度损失的贡献
        
        Args:
            global_loss: 全局对比损失
            modal_loss: 模态对比损失  
            temporal_loss: 时间对比损失
            spatial_loss: 空间对比损失
            
        Returns:
            加权后的总损失
        """
        # 损失权重配置，总和为1
        weights = {
            'global': 0.4,    # 全局损失权重最高
            'modal': 0.3,     # 模态损失权重次之
            'temporal': 0.2,  # 时间损失权重中等
            'spatial': 0.1    # 空间损失权重最低
        }
        
        # 计算加权损失
        weighted_loss = (weights['global'] * global_loss + 
                        weights['modal'] * modal_loss + 
                        weights['temporal'] * temporal_loss + 
                        weights['spatial'] * spatial_loss)
        
        return weighted_loss
    
    def _compute_global_loss(self, global_features, labels):
        """计算全局对比损失
        
        对整体特征表示进行对比学习，同一类的样本应当更加接近
        
        Args:
            global_features: 全局特征表示，形状为[batch_size, feature_dim]
            labels: 标签，形状为[batch_size]
            
        Returns:
            全局对比损失
        """
        # 检查并适配特征维度
        if global_features.size(1) != self.feature_dim:
            input_dim = global_features.size(1)
            print(f"警告: global_features 维度 ({input_dim}) 与预期维度 ({self.feature_dim}) 不匹配")
            
            # 处理低维特征 - 填充到所需维度
            if input_dim < self.feature_dim:
                padding = torch.zeros(global_features.size(0), self.feature_dim - input_dim, 
                                     device=global_features.device)
                global_features = torch.cat([global_features, padding], dim=1)
                print(f"已填充global_features到维度: {global_features.shape}")
            # 处理高维特征 - 使用线性投影降维
            else:
                # 创建或重用适配器
                if self.global_feature_adapter is None or self.global_feature_adapter.in_features != input_dim:
                    self.global_feature_adapter = nn.Linear(input_dim, self.feature_dim).to(global_features.device)
                    # 使用Xavier初始化确保输出分布合理
                    nn.init.xavier_uniform_(self.global_feature_adapter.weight)
                    nn.init.zeros_(self.global_feature_adapter.bias)
                    print(f"已创建新的全局特征适配器: {input_dim} -> {self.feature_dim}")
                
                # 应用适配器进行降维
                global_features = self.global_feature_adapter(global_features)
                print(f"已调整global_features到维度: {global_features.shape}")
        
        # 投影全局特征
        global_proj = self.global_projector(global_features)
        global_proj = F.normalize(global_proj, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(global_proj, global_proj.t()) / self.temperatures['global']
        
        # 创建标签矩阵（相同标签的样本为正对，不同标签的样本为负对）
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(global_features.device)
        
        # 对角线掩码（排除自己与自己的对比）
        mask_diag = torch.eye(labels.size(0)).to(global_features.device)
        mask_pos = mask_pos - mask_diag
        
        # 使用掩码计算NCE损失
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 计算正样本对的平均对数似然
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos.sum(1).clamp(min=1.0)
        
        # 对比损失
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _compute_modal_loss(self, text_features, audio_features, video_features, labels):
        """计算跨模态对比损失

        Args:
            text_features: 文本特征 [B, T, D]
            audio_features: 音频特征 [B, T, D]
            video_features: 视频特征 [B, T, D]
            labels: 标签，形状为[batch_size]
        """
        batch_size = text_features.size(0)
        
        # 先处理每个模态，确保每个模态的特征都是二维张量 [B, D]
        # 处理音频特征
        if len(audio_features.shape) > 2:
            print(f"音频特征形状: {audio_features.shape}，进行平均池化")
            if len(audio_features.shape) == 3:  # [B, T, D]
                audio_global = torch.mean(audio_features, dim=1)  # 时间维度平均池化 [B, D]
            elif len(audio_features.shape) == 4:  # [B, C, T, D] 或其他
                audio_global = torch.mean(audio_features, dim=(1, 2))  # 对中间两个维度平均池化
            else:
                audio_global = audio_features.reshape(batch_size, -1)  # 将所有维度展平成特征
        else:
            audio_global = audio_features
        
        # 处理视频特征
        if len(video_features.shape) > 2:
            print(f"视频特征形状: {video_features.shape}，进行平均池化")
            if len(video_features.shape) == 3:  # [B, T, D]
                video_global = torch.mean(video_features, dim=1)  # 时间维度平均池化 [B, D]
            elif len(video_features.shape) == 4:  # [B, T, S, D] 或其他
                video_global = torch.mean(video_features, dim=(1, 2))  # 对中间两个维度平均池化
            else:
                video_global = video_features.reshape(batch_size, -1)  # 将所有维度展平成特征
        else:
            video_global = video_features
        
        # 处理文本特征
        if len(text_features.shape) > 2:
            print(f"文本特征形状: {text_features.shape}，进行平均池化")
            if len(text_features.shape) == 3:  # [B, T, D]
                text_features = torch.mean(text_features, dim=1)  # 时间维度平均池化 [B, D]
            else:
                text_features = torch.mean(text_features.reshape(batch_size, -1, text_features.shape[-1]), dim=1)
        
        print(f"模态特征维度: text={text_features.shape}, audio={audio_global.shape}, video={video_global.shape}")
        
        # 额外确认文本特征是二维的
        if len(text_features.shape) > 2:
            print(f"警告：文本特征仍然是多维的，尝试强制转换为二维")
            # 如果平均池化后特征仍然不是二维，尝试直接重塑
            if len(text_features.shape) == 3:  # [B, T, D]
                text_dim = text_features.size(2)
                text_features = text_features.reshape(batch_size, -1)  # 将所有时间步展平
                
                # 如果展平后的维度太大，只保留前text_dim个特征
                if text_features.size(1) > text_dim:
                    text_features = text_features[:, :text_dim]
            elif len(text_features.shape) == 4:  # [B, T, S, D]
                text_dim = text_features.size(3)
                text_features = text_features.reshape(batch_size, -1)
                if text_features.size(1) > text_dim:
                    text_features = text_features[:, :text_dim]
        
        # 确保所有特征具有相同的维度
        text_dim = text_features.size(1)
        
        # 对音频特征进行维度调整
        if audio_global.size(1) != text_dim:
            print(f"音频特征维度 ({audio_global.size(1)}) 与文本特征维度 ({text_dim}) 不匹配")
            if audio_global.size(1) > text_dim:
                # 截断
                audio_global = audio_global[:, :text_dim]
            else:
                # 填充
                padding = torch.zeros(batch_size, text_dim - audio_global.size(1), device=audio_global.device)
                audio_global = torch.cat([audio_global, padding], dim=1)
        
        # 对视频特征进行维度调整
        if video_global.size(1) != text_dim:
            print(f"视频特征维度 ({video_global.size(1)}) 与文本特征维度 ({text_dim}) 不匹配")
            if video_global.size(1) > text_dim:
                # 截断
                video_global = video_global[:, :text_dim]
            else:
                # 填充
                padding = torch.zeros(batch_size, text_dim - video_global.size(1), device=video_global.device)
                video_global = torch.cat([video_global, padding], dim=1)
        
        print(f"调整后模态特征维度: text={text_features.shape}, audio={audio_global.shape}, video={video_global.shape}")
        
        # 再次检查所有特征维度是否一致
        if text_features.shape != audio_global.shape or text_features.shape != video_global.shape:
            print(f"警告：调整后特征维度仍不一致，将执行额外处理")
            
            # 获取共同的批次大小
            common_batch_size = min(text_features.size(0), audio_global.size(0), video_global.size(0))
            
            # 如果批次大小不一致，进行截断
            if text_features.size(0) != common_batch_size:
                text_features = text_features[:common_batch_size]
            if audio_global.size(0) != common_batch_size:
                audio_global = audio_global[:common_batch_size]
            if video_global.size(0) != common_batch_size:
                video_global = video_global[:common_batch_size]
            
            # 获取共同的特征维度
            common_dim = min(text_features.size(1), audio_global.size(1), video_global.size(1))
            
            # 如果特征维度不一致，进行截断
            if text_features.size(1) != common_dim:
                text_features = text_features[:, :common_dim]
            if audio_global.size(1) != common_dim:
                audio_global = audio_global[:, :common_dim]
            if video_global.size(1) != common_dim:
                video_global = video_global[:, :common_dim]
                
            print(f"特征强制调整后: text={text_features.shape}, audio={audio_global.shape}, video={video_global.shape}")
        
        # 按模态排列特征 [B, 3, D]，顺序为：文本、音频、视频
        try:
            modality_features = torch.stack([text_features, audio_global, video_global], dim=1)
            print(f"成功堆叠模态特征: shape={modality_features.shape}")
        except Exception as e:
            print(f"堆叠模态特征时出错: {e}")
            print(f"尝试备选方案...")
            
            # 获取当前批次大小和特征维度
            current_batch_size = text_features.size(0)
            feature_dim = text_features.size(1)
            
            # 创建形状统一的张量
            modality_features = torch.zeros((current_batch_size, 3, feature_dim), device=text_features.device)
            
            # 手动填充张量
            modality_features[:, 0, :] = text_features
            modality_features[:, 1, :] = audio_global
            modality_features[:, 2, :] = video_global
            print(f"备选方案成功: shape={modality_features.shape}")
        
        # 改进2：修复模态对比损失 - 使用正确的InfoNCE损失
        loss = self._compute_infonce_modal_loss(modality_features, labels)
        
        # 只返回损失，不返回模态特征
        return loss
    
    def _compute_infonce_modal_loss(self, modality_features, labels):
        """改进2：计算正确的InfoNCE模态对比损失
        
        Args:
            modality_features: 模态特征，形状为 [B, 3, D]（文本、音频、视频）
            labels: 标签，形状为 [B]
            
        Returns:
            InfoNCE对比损失
        """
        batch_size, num_modalities, feature_dim = modality_features.shape
        device = modality_features.device
        
        # 对特征进行L2归一化
        modality_features_norm = F.normalize(modality_features, dim=-1)
        
        total_loss = 0.0
        num_pairs = 0
        
        # 计算不同模态间的对比损失
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                modal_i = modality_features_norm[:, i, :]  # [B, D]
                modal_j = modality_features_norm[:, j, :]  # [B, D]
                
                # 计算相似度矩阵
                sim_matrix = torch.matmul(modal_i, modal_j.t()) / self.temperatures['modal']  # [B, B]
                
                # 创建正样本掩码（相同类别为正样本）
                labels_expanded = labels.view(-1, 1)
                pos_mask = torch.eq(labels_expanded, labels_expanded.t()).float()
                
                # 对角线掩码（同一样本不同模态为强正样本）
                diag_mask = torch.eye(batch_size, device=device)
                
                # 组合掩码：对角线权重为1.0，相同类别的不同样本权重为0.5
                combined_mask = diag_mask + 0.5 * (pos_mask - diag_mask)
                
                # 计算InfoNCE损失
                exp_sim = torch.exp(sim_matrix)
                log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
                
                # 计算正样本对的平均对数似然
                pos_log_prob = (combined_mask * log_prob).sum(1) / combined_mask.sum(1).clamp(min=1.0)
                loss = -pos_log_prob.mean()
                
                total_loss += loss
                num_pairs += 1
        
        # 返回平均损失
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)
    
    def _compute_temporal_loss(self, audio_features, video_features, labels):
        """计算时间对比损失"""
        # 检查特征是否包含时间维度
        if len(audio_features.shape) < 3 or len(video_features.shape) < 3:
            return torch.tensor(0.0).to(audio_features.device)
        
        # 获取音频的时间特征 [B, T, D]
        if len(audio_features.shape) == 3:
            audio_temporal = audio_features
        else:
            return torch.tensor(0.0).to(audio_features.device)
            
        # 获取视频的时间特征 [B, T, D]
        if len(video_features.shape) == 4:  # [B, T, S, D]
            video_temporal = video_features.mean(dim=2)  # [B, T, D]
        elif len(video_features.shape) == 3:  # [B, T, D]
            video_temporal = video_features
        else:
            return torch.tensor(0.0).to(audio_features.device)
        
        # 使用插值/自适应池化将时间维度对齐到相同长度
        audio_time = audio_temporal.shape[1]
        video_time = video_temporal.shape[1]
        target_len = min(audio_time, video_time)
        if target_len <= 0:
            return torch.tensor(0.0).to(audio_features.device)
        
        # 线性插值对齐（避免整除分段造成的错误）
        audio_aligned = F.interpolate(audio_temporal.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
        video_aligned = F.interpolate(video_temporal.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
        
        # 将时间段数调整为不超过实际帧数，并进行均匀分段
        T = target_len
        temporal_segments = min(self.temporal_segments, T)
        if temporal_segments <= 0:
            return torch.tensor(0.0).to(audio_features.device)
        
        if T > temporal_segments:
            # 使用自适应平均池化到指定段数
            audio_segments = F.adaptive_avg_pool1d(audio_aligned.transpose(1, 2), temporal_segments).transpose(1, 2)  # [B, S, D]
            video_segments = F.adaptive_avg_pool1d(video_aligned.transpose(1, 2), temporal_segments).transpose(1, 2)  # [B, S, D]
        else:
            audio_segments = audio_aligned
            video_segments = video_aligned
        
        # 投影时间段特征
        batch_size, n_segments, dim = audio_segments.shape
        audio_segments_flat = audio_segments.reshape(-1, dim)  # [B*S, D]
        audio_proj = self.temporal_projector(audio_segments_flat)  # [B*S, P]
        audio_proj = F.normalize(audio_proj, dim=1)
        audio_proj = audio_proj.view(batch_size, n_segments, -1)  # [B, S, P]
        
        video_segments_flat = video_segments.reshape(-1, dim)  # [B*S, D]
        video_proj = self.temporal_projector(video_segments_flat)  # [B*S, P]
        video_proj = F.normalize(video_proj, dim=1)
        video_proj = video_proj.view(batch_size, n_segments, -1)  # [B, S, P]
        
        # 计算时间一致性损失 - 同一时间段的音频和视频特征应该更相似
        temporal_loss = 0.0
        
        for t in range(n_segments):
            audio_t = audio_proj[:, t, :]  # [B, P]
            video_t = video_proj[:, t, :]  # [B, P]
            
            # 相似度矩阵 [B, B]
            sim_matrix = torch.matmul(audio_t, video_t.t()) / self.temperatures['temporal']
            
            # 创建标签矩阵
            labels_expanded = labels.view(-1, 1)
            mask_pos = torch.eq(labels_expanded, labels_expanded.t()).float()
            mask_diag = torch.eye(batch_size, device=audio_features.device)
            
            # 对角线是同一个样本不同模态的对比，这是正例；同类不同样本为弱正
            diff_sample_weight = 0.5
            combined_mask = mask_diag + diff_sample_weight * (mask_pos - mask_diag)
            
            # 使用logsumexp实现数值稳定的InfoNCE
            log_denom = torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # [B, 1]
            log_prob = sim_matrix - log_denom  # [B, B]
            
            # 正样本对的平均对数似然（加权）
            pos_log_prob = (combined_mask * log_prob).sum(1) / combined_mask.sum(1).clamp(min=1.0)
            segment_loss = -pos_log_prob.mean()
            temporal_loss += segment_loss
        
        # 显式 1/S 平均
        temporal_loss = temporal_loss / max(1, n_segments)
        
        return temporal_loss
    
    def _compute_spatial_loss(self, video_features, labels):
        """计算空间对比损失"""
        # 检查视频特征是否包含空间维度
        if len(video_features.shape) != 4:  # 应该是 [B, T, S, D]
            return torch.tensor(0.0).to(video_features.device)
        
        # 获取视频的空间特征
        batch_size, time_steps, spatial_dim, feature_dim = video_features.shape
        
        # 直接使用时间平均的空间特征 [B, S, D]
        video_spatial = video_features.mean(dim=1)
        
        # 投影空间区域特征
        batch_size, n_regions, dim = video_spatial.shape
        video_spatial_flat = video_spatial.reshape(-1, dim)  # [B*S, D]
        spatial_proj = self.spatial_projector(video_spatial_flat)  # [B*S, P]
        spatial_proj = F.normalize(spatial_proj, dim=1)
        spatial_proj = spatial_proj.view(batch_size, n_regions, -1)  # [B, S, P]
        
        # 参考区域：样本内区域均值（最小化改动下保留原策略）
        reference_region = spatial_proj.mean(dim=1)  # [B, P]
        
        spatial_loss = 0.0
        for s in range(n_regions):
            region_s = spatial_proj[:, s, :]  # [B, P]
            
            # 相似度矩阵 [B, B]
            sim_matrix = torch.matmul(region_s, reference_region.t()) / self.temperatures['spatial']
            
            # 标签矩阵
            labels_expanded = labels.view(-1, 1)
            mask_pos = torch.eq(labels_expanded, labels_expanded.t()).float()
            mask_diag = torch.eye(batch_size).to(video_features.device)
            
            # 同一样本为强正，相同类别不同样本为弱正
            diff_sample_weight = 0.3
            combined_mask = mask_diag + diff_sample_weight * (mask_pos - mask_diag)
            
            # 数值稳定的InfoNCE
            log_denom = torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # [B, 1]
            log_prob = sim_matrix - log_denom
            mean_log_prob_pos = (combined_mask * log_prob).sum(1) / combined_mask.sum(1).clamp(min=1.0)
            region_loss = -mean_log_prob_pos.mean()
            spatial_loss += region_loss
        
        # 显式 1/P 平均
        spatial_loss = spatial_loss / max(1, n_regions)
        
        return spatial_loss 