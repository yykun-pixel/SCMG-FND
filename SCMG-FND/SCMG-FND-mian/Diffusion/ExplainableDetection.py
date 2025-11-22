import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.autograd import Function
from typing import Tuple, List, Dict, Optional, Union


class GradientExtractor(Function):
    """用于提取梯度的自定义自动求导函数"""
    
    @staticmethod
    def forward(ctx, input, pred_logits, target_class):
        ctx.save_for_backward(input, pred_logits, target_class)
        return input
        
    @staticmethod
    def backward(ctx, grad_output):
        input, pred_logits, target_class = ctx.saved_tensors
        batch_size = pred_logits.size(0)
        
        # 只返回与目标类别相关的梯度
        grad_input = torch.zeros_like(input)
        for i in range(batch_size):
            grad_input[i] = grad_output[i] * pred_logits[i, target_class[i]]
            
        return grad_input, None, None


class ExplainableDetection(nn.Module):
    """
    可解释性和虚假区域定位模块
    
    此模块通过计算模态特征的注意力权重，生成可视化热图，
    帮助定位视频中可能的虚假区域，并提供多模态决策的可解释性。
    """
    
    def __init__(self, unified_size: int, vertex_num: int):
        """
        初始化ExplainableDetection模块
        
        Args:
            unified_size: 统一特征维度的大小
            vertex_num: 图结构中顶点的数量
        """
        super().__init__()
        
        # 注意力机制用于计算各模态贡献
        self.modality_attention = nn.Sequential(
            nn.Linear(unified_size * 3, unified_size),
            nn.ReLU(),
            nn.Linear(unified_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # 用于生成特征重要性图的卷积层
        self.importance_conv = nn.Conv1d(unified_size, 1, kernel_size=1)
        
        # 用于精细定位的多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=unified_size, 
            num_heads=4,
            batch_first=True
        )
        
        # CAM（类激活映射）生成层
        self.cam_generator = nn.Linear(vertex_num, 1)
        
        # 特征维度
        self.unified_size = unified_size
        self.vertex_num = vertex_num

    def forward(self, x_t: torch.Tensor, x_a: torch.Tensor, x_v: torch.Tensor, 
                prediction_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生成模型决策的可解释性结果
        
        Args:
            x_t: 文本模态特征 [batch_size, vertex_num, feature_dim]
            x_a: 音频模态特征 [batch_size, vertex_num, feature_dim]
            x_v: 视频模态特征 [batch_size, vertex_num, feature_dim]
            prediction_logits: 模型最终预测结果 [batch_size, num_classes]
            
        Returns:
            包含可解释性信息的字典，包括各模态贡献度、特征重要性图和虚假区域热图
        """
        batch_size = x_t.shape[0]
        
        # 确保输入特征形状一致
        if x_t.shape != x_a.shape or x_t.shape != x_v.shape:
            # 调整为相同形状
            min_batch = min(x_t.shape[0], x_a.shape[0], x_v.shape[0])
            x_t = x_t[:min_batch]
            x_a = x_a[:min_batch]
            x_v = x_v[:min_batch]
            
            print(f"调整特征形状: x_t={x_t.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
        
        # 1. 计算各模态对最终决策的贡献度
        # 采用全局池化获取模态级特征表示
        t_global = torch.mean(x_t, dim=1)  # [batch_size, feature_dim]
        a_global = torch.mean(x_a, dim=1)
        v_global = torch.mean(x_v, dim=1)
        
        # 拼接全局特征表示
        concat_features = torch.cat([t_global, a_global, v_global], dim=-1)  # [batch_size, 3*feature_dim]
        
        # 计算模态间注意力权重
        modality_weights = self.modality_attention(concat_features)  # [batch_size, 3]
        
        # 2. 生成特征重要性图 - 指示每个顶点对决策的重要性
        # 为文本、音频和视频分别生成重要性图
        t_importance = self.generate_importance_map(x_t)  # [batch_size, vertex_num]
        a_importance = self.generate_importance_map(x_a)
        v_importance = self.generate_importance_map(x_v)
        
        # 3. 对视频模态，生成可视化的热图用于定位虚假区域
        fake_region_heatmap = self.generate_fake_region_heatmap(x_v, prediction_logits)
        
        # 4. 使用多头注意力找出模态间交互的关键区域
        # 计算模态间注意力，找出文本、音频特征如何影响视频特征判断
        t_v_attn_output, t_v_attn_weights = self.multihead_attn(x_v, x_t, x_t)
        a_v_attn_output, a_v_attn_weights = self.multihead_attn(x_v, x_a, x_a)
        
        # 规范化注意力权重，以便可视化
        t_v_attn_weights = t_v_attn_weights.mean(dim=1)  # 平均头之间的注意力 [batch_size, v_len, t_len]
        a_v_attn_weights = a_v_attn_weights.mean(dim=1)  # [batch_size, v_len, a_len]
        
        # 返回可解释性结果
        return {
            "modality_weights": modality_weights,  # 各模态贡献权重 [batch_size, 3]
            "text_importance": t_importance,  # 文本特征重要性 [batch_size, vertex_num]
            "audio_importance": a_importance,  # 音频特征重要性 [batch_size, vertex_num]
            "video_importance": v_importance,  # 视频特征重要性 [batch_size, vertex_num]
            "fake_region_heatmap": fake_region_heatmap,  # 视频中虚假区域热图 [batch_size, H, W]
            "text_video_attention": t_v_attn_weights,  # 文本-视频注意力 [batch_size, v_len, t_len]
            "audio_video_attention": a_v_attn_weights,  # 音频-视频注意力 [batch_size, v_len, a_len]
        }
    
    def generate_importance_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入特征生成重要性图
        
        Args:
            x: 输入特征 [batch_size, vertex_num, feature_dim]
            
        Returns:
            特征重要性图 [batch_size, vertex_num]
        """
        # 转换维度以适应Conv1d的输入要求
        x_transposed = x.transpose(1, 2)  # [batch_size, feature_dim, vertex_num]
        
        # 应用1D卷积降维
        importance = self.importance_conv(x_transposed)  # [batch_size, 1, vertex_num]
        importance = importance.squeeze(1)  # [batch_size, vertex_num]
        
        # 使用Softmax使重要性分数归一化，便于解释
        importance = F.softmax(importance, dim=-1)
        
        return importance
    
    def generate_fake_region_heatmap(self, 
                                    video_features: torch.Tensor, 
                                    prediction_logits: torch.Tensor) -> torch.Tensor:
        """
        为视频生成热图，用于定位可能的虚假区域
        
        这里使用类似于Grad-CAM的方法，但直接基于特征而非梯度
        
        Args:
            video_features: 视频特征 [batch_size, vertex_num, feature_dim]
            prediction_logits: 模型预测 [batch_size, num_classes]
            
        Returns:
            虚假区域热图 [batch_size, H, W]，H和W会在应用中被确定
        """
        batch_size = video_features.shape[0]
        
        # 1. 获取预测的虚假概率 (假设二分类，第二个元素是虚假类的概率)
        fake_probabilities = F.softmax(prediction_logits, dim=-1)
        
        if fake_probabilities.shape[1] >= 2:
            # 二分类情况，取第二个元素（索引1）作为虚假类概率
            fake_prob = fake_probabilities[:, 1:2]  # [batch_size, 1]
        else:
            # 单输出情况，直接使用
            fake_prob = fake_probabilities  # [batch_size, 1]
        
        # 2. 使用线性层生成类激活映射（CAM）权重
        # 为视频顶点创建权重
        cam_weights = self.cam_generator(video_features.transpose(1, 2))  # [batch_size, feature_dim, 1]
        cam_weights = cam_weights.transpose(1, 2)  # [batch_size, 1, feature_dim]
        
        # 3. 与视频特征加权组合生成初步热图
        # [batch_size, 1, feature_dim] x [batch_size, vertex_num, feature_dim] -> [batch_size, vertex_num]
        heatmap = torch.bmm(cam_weights, video_features.transpose(1, 2)).squeeze(1)
        
        # 4. 应用ReLU激活以突出正向贡献
        heatmap = F.relu(heatmap)
        
        # 5. 归一化热图方便可视化
        # 防止除零
        heatmap_max = torch.max(heatmap, dim=1, keepdim=True)[0]
        heatmap_min = torch.min(heatmap, dim=1, keepdim=True)[0]
        heatmap_range = heatmap_max - heatmap_min + 1e-8  # 避免除零
        normalized_heatmap = (heatmap - heatmap_min) / heatmap_range
        
        # 默认返回尺寸为[batch_size, vertex_num]的热图
        # 注意：此热图仍需在使用时转换为实际视频帧的二维图像空间
        return normalized_heatmap

    def visualize_explanation(self, 
                             explanation_dict: Dict[str, torch.Tensor], 
                             video_frames: Optional[torch.Tensor] = None, 
                             save_path: Optional[str] = None) -> List[np.ndarray]:
        """
        可视化解释结果
        
        Args:
            explanation_dict: 包含可解释性信息的字典
            video_frames: 原始视频帧 [batch_size, num_frames, H, W, C]
            save_path: 保存可视化结果的路径
            
        Returns:
            可视化结果图像列表
        """
        batch_size = explanation_dict["modality_weights"].shape[0]
        results = []
        
        for i in range(batch_size):
            # 1. 绘制模态贡献度
            modal_weights = explanation_dict["modality_weights"][i].detach().cpu().numpy()
            modal_names = ["文本", "音频", "视频"]
            
            # 创建模态贡献图
            plt.figure(figsize=(10, 5))
            plt.bar(modal_names, modal_weights)
            plt.title("模态贡献度")
            plt.ylim(0, 1)
            
            # 保存模态贡献图
            if save_path:
                plt.savefig(f"{save_path}_modality_weights_{i}.png")
                plt.close()
            
            # 2. 创建虚假区域热图可视化
            heatmap = explanation_dict["fake_region_heatmap"][i].detach().cpu().numpy()
            
            # 如果提供了视频帧，将热图叠加到视频帧上
            if video_frames is not None and i < len(video_frames):
                # 选择中间帧作为代表
                middle_frame_idx = len(video_frames[i]) // 2
                frame = video_frames[i][middle_frame_idx].detach().cpu().numpy()
                
                # 将热图调整为与视频帧相同大小
                h, w = frame.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (w, h))
                
                # 转换为热力图颜色
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # 叠加热力图和原始帧
                result_frame = cv2.addWeighted(
                    frame, 0.6, 
                    heatmap_colored, 0.4, 
                    0
                )
                
                # 保存结果
                if save_path:
                    cv2.imwrite(f"{save_path}_fake_region_{i}.png", result_frame)
                
                results.append(result_frame)
            else:
                # 仅热图情况
                heatmap_colored = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                if save_path:
                    cv2.imwrite(f"{save_path}_fake_region_{i}.png", heatmap_colored)
                
                results.append(heatmap_colored)
            
            # 3. 可视化特征重要性
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(explanation_dict["text_importance"][i].detach().cpu().numpy())
            plt.title("文本特征重要性")
            
            plt.subplot(1, 3, 2)
            plt.plot(explanation_dict["audio_importance"][i].detach().cpu().numpy())
            plt.title("音频特征重要性")
            
            plt.subplot(1, 3, 3)
            plt.plot(explanation_dict["video_importance"][i].detach().cpu().numpy())
            plt.title("视频特征重要性")
            
            if save_path:
                plt.savefig(f"{save_path}_feature_importance_{i}.png")
                plt.close()
            
            # 4. 可视化模态间注意力
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            plt.imshow(explanation_dict["text_video_attention"][i].detach().cpu().numpy(), 
                      aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title("文本-视频注意力图")
            plt.xlabel("文本序列")
            plt.ylabel("视频特征")
            
            plt.subplot(2, 1, 2)
            plt.imshow(explanation_dict["audio_video_attention"][i].detach().cpu().numpy(), 
                      aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title("音频-视频注意力图")
            plt.xlabel("音频序列")
            plt.ylabel("视频特征")
            
            if save_path:
                plt.savefig(f"{save_path}_cross_modal_attention_{i}.png")
                plt.close()
        
        return results


def get_explanation_metrics(explanation_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    计算解释结果的量化指标
    
    Args:
        explanation_dict: 包含可解释性信息的字典
        
    Returns:
        包含量化指标的字典
    """
    metrics = {}
    
    # 1. 各模态对决策的贡献度
    modality_weights = explanation_dict["modality_weights"]
    metrics["text_contribution"] = float(torch.mean(modality_weights[:, 0]).item())
    metrics["audio_contribution"] = float(torch.mean(modality_weights[:, 1]).item())
    metrics["video_contribution"] = float(torch.mean(modality_weights[:, 2]).item())
    
    # 2. 计算模态间一致性 - 使用注意力权重的标准差作为指标
    # 注意力权重波动大表示模态间不一致性高
    t_v_std = torch.std(explanation_dict["text_video_attention"], dim=-1).mean().item()
    a_v_std = torch.std(explanation_dict["audio_video_attention"], dim=-1).mean().item()
    metrics["text_video_inconsistency"] = float(t_v_std)
    metrics["audio_video_inconsistency"] = float(a_v_std)
    
    # 3. 特征重要性分布熵 - 分散的特征重要性具有高熵值
    def calc_entropy(tensor):
        tensor = tensor + 1e-10  # 防止log(0)
        entropy = -torch.sum(tensor * torch.log(tensor), dim=-1)
        return entropy.mean().item()
    
    metrics["text_feature_entropy"] = float(calc_entropy(explanation_dict["text_importance"]))
    metrics["audio_feature_entropy"] = float(calc_entropy(explanation_dict["audio_importance"]))
    metrics["video_feature_entropy"] = float(calc_entropy(explanation_dict["video_importance"]))
    
    # 4. 虚假区域集中度 - 热图中高亮区域的集中程度
    # 高集中度表示虚假区域更明确
    heatmap = explanation_dict["fake_region_heatmap"]
    threshold = 0.7  # 虚假区域的阈值
    high_attention_ratio = (heatmap > threshold).float().mean().item()
    metrics["fake_region_concentration"] = float(high_attention_ratio)
    
    return metrics


def visualize_explanation(video_frames, heatmaps, save_path):
    """
    将热力图可视化并叠加到视频帧上
    
    参数:
        video_frames: 视频帧列表，每帧为numpy数组 [height, width, 3]
        heatmaps: 热力图张量 [num_frames]
        save_path: 保存可视化结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    for i, (frame, heatmap) in enumerate(zip(video_frames, heatmaps.detach().cpu().numpy())):
        # 将热力图缩放到帧的尺寸
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # 将热力图转换为彩色图
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # 叠加到原始帧
        alpha = 0.4  # 透明度
        overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        # 保存结果
        output_path = os.path.join(save_path, f"frame_{i:03d}_explanation.png")
        cv2.imwrite(output_path, overlayed)
        
    print(f"可视化结果已保存到 {save_path}")


def compute_localization_metrics(pred_heatmap, gt_mask):
    """
    计算定位准确率指标
    
    参数:
        pred_heatmap: 预测的热力图 [batch_size, num_elements]
        gt_mask: 真实标注区域 [batch_size, num_elements]
        
    返回:
        dict: 包含各种定位指标
    """
    # 将预测热力图二值化(阈值0.5)
    pred_binary = (pred_heatmap > 0.5).float()
    
    # 计算交并比(IoU)
    intersection = (pred_binary * gt_mask).sum(dim=1)
    union = pred_binary.sum(dim=1) + gt_mask.sum(dim=1) - intersection
    iou = intersection / (union + 1e-6)
    
    # 计算精确率
    precision = (pred_binary * gt_mask).sum(dim=1) / (pred_binary.sum(dim=1) + 1e-6)
    
    # 计算召回率
    recall = (pred_binary * gt_mask).sum(dim=1) / (gt_mask.sum(dim=1) + 1e-6)
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # 以字典形式返回所有指标
    metrics = {
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }
    
    return metrics 