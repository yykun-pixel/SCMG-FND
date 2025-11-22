import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class AdversarialVerifier(nn.Module):
    """
    对抗性验证框架
    
    主要功能：
    1. 生成对抗样本以挑战模型
    2. 验证模型预测与解释之间的一致性
    3. 提供对抗鲁棒性评分
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 eps: float = 0.01,
                 alpha: float = 0.004,
                 steps: int = 5,
                 consistency_threshold: float = 0.7):
        """
        初始化对抗性验证框架
        
        Args:
            model: 被验证的模型
            eps: 扰动大小上限
            alpha: PGD攻击步长
            steps: PGD攻击步数
            consistency_threshold: 一致性评分阈值
        """
        super().__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.consistency_threshold = consistency_threshold
        
        # 边界样本缓存池
        self.boundary_samples_pool = {
            'samples': [],    # 存储边界样本
            'labels': [],     # 存储真实标签
            'confidence': []  # 存储预测置信度
        }
        
        # 最大边界样本数量
        self.max_boundary_samples = 100
    
    def generate_adversarial_samples(self, 
                                    inputs: Dict[str, torch.Tensor], 
                                    labels: torch.Tensor,
                                    targeted: bool = False,
                                    target_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        生成对抗样本
        
        Args:
            inputs: 输入字典，包含不同模态的输入
            labels: 真实标签
            targeted: 是否为目标攻击
            target_labels: 目标标签（仅在targeted=True时使用）
            
        Returns:
            包含扰动后输入的字典
        """
        perturbed_inputs = {}
        
        # 对每个模态分别进行对抗性扰动
        for modality, x in inputs.items():
            # 只对具有梯度的tensor进行扰动
            if isinstance(x, torch.Tensor) and x.requires_grad:
                # 克隆输入，避免修改原始输入
                perturbed_x = x.clone().detach().requires_grad_(True)
                
                # PGD攻击
                if targeted:
                    # 目标攻击：让模型预测为目标标签
                    target = target_labels if target_labels is not None else (1 - labels)  # 默认翻转标签
                    perturbed_x = self._pgd_targeted(perturbed_x, target, modality)
                else:
                    # 非目标攻击：让模型预测错误
                    perturbed_x = self._pgd_untargeted(perturbed_x, labels, modality)
                
                perturbed_inputs[modality] = perturbed_x
            else:
                # 不扰动没有梯度的输入
                perturbed_inputs[modality] = x
        
        return perturbed_inputs
    
    def _pgd_untargeted(self, 
                       x: torch.Tensor, 
                       y: torch.Tensor, 
                       modality: str) -> torch.Tensor:
        """
        非目标PGD攻击
        
        Args:
            x: 输入特征
            y: 真实标签
            modality: 模态名称
            
        Returns:
            扰动后的输入
        """
        # 保存原始输入
        x_orig = x.clone().detach()
        
        # 初始随机扰动
        delta = torch.zeros_like(x, requires_grad=True)
        if self.eps > 0:
            delta = torch.rand_like(x) * 2 * self.eps - self.eps
            delta = torch.clamp(x + delta, 0, 1) - x
        
        # 多步PGD攻击
        for _ in range(self.steps):
            # 构建完整输入
            adv_inputs = {modality: x + delta}
            for other_modality, other_input in self.model.last_inputs.items():
                if other_modality != modality:
                    adv_inputs[other_modality] = other_input
            
            # 前向传播
            self.model.zero_grad()
            loss, outputs = self.model(**adv_inputs)
            
            # 计算对抗损失（最大化与真实标签的差异）
            adv_loss = -F.cross_entropy(outputs, y)
            
            # 反向传播
            adv_loss.backward()
            
            # 更新扰动
            with torch.no_grad():
                delta = delta + self.alpha * delta.grad.sign()
                delta = torch.clamp(delta, -self.eps, self.eps)
                delta = torch.clamp(x_orig + delta, 0, 1) - x_orig
            
            # 重置梯度
            delta = delta.detach().requires_grad_(True)
        
        # 返回扰动后的输入
        return x_orig + delta
    
    def _pgd_targeted(self, 
                     x: torch.Tensor, 
                     target_y: torch.Tensor, 
                     modality: str) -> torch.Tensor:
        """
        目标PGD攻击
        
        Args:
            x: 输入特征
            target_y: 目标标签
            modality: 模态名称
            
        Returns:
            扰动后的输入
        """
        # 保存原始输入
        x_orig = x.clone().detach()
        
        # 初始随机扰动
        delta = torch.zeros_like(x, requires_grad=True)
        if self.eps > 0:
            delta = torch.rand_like(x) * 2 * self.eps - self.eps
            delta = torch.clamp(x + delta, 0, 1) - x
        
        # 多步PGD攻击
        for _ in range(self.steps):
            # 构建完整输入
            adv_inputs = {modality: x + delta}
            for other_modality, other_input in self.model.last_inputs.items():
                if other_modality != modality:
                    adv_inputs[other_modality] = other_input
            
            # 前向传播
            self.model.zero_grad()
            loss, outputs = self.model(**adv_inputs)
            
            # 计算目标对抗损失（最小化与目标标签的差异）
            adv_loss = F.cross_entropy(outputs, target_y)
            
            # 反向传播
            adv_loss.backward()
            
            # 更新扰动
            with torch.no_grad():
                delta = delta - self.alpha * delta.grad.sign()  # 注意这里是减号，与非目标攻击相反
                delta = torch.clamp(delta, -self.eps, self.eps)
                delta = torch.clamp(x_orig + delta, 0, 1) - x_orig
            
            # 重置梯度
            delta = delta.detach().requires_grad_(True)
        
        # 返回扰动后的输入
        return x_orig + delta
    
    def verify_consistency(self, 
                          inputs: Dict[str, torch.Tensor], 
                          explanations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        验证预测与解释的一致性
        
        Args:
            inputs: 输入字典，包含不同模态的输入
            explanations: 解释结果字典
            
        Returns:
            包含一致性评分的字典
        """
        consistency_scores = {}
        
        # 模型预测
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs, dim=1)
        
        # 提取解释信息
        modality_weights = explanations.get('modality_weights', None)
        feature_importances = {
            'text': explanations.get('text_importance', None),
            'audio': explanations.get('audio_importance', None),
            'video': explanations.get('video_importance', None)
        }
        
        # 计算预测-模态贡献一致性
        if modality_weights is not None:
            pred_modality_consistency = self._compute_prediction_modality_consistency(
                inputs, predictions, modality_weights
            )
            consistency_scores['pred_modality_consistency'] = pred_modality_consistency
        
        # 计算预测-特征重要性一致性
        pred_feature_consistency = {}
        for modality, importance in feature_importances.items():
            if importance is not None and modality in inputs:
                pred_feature_consistency[modality] = self._compute_prediction_feature_consistency(
                    inputs[modality], predictions, importance
                )
        
        if pred_feature_consistency:
            # 计算平均特征一致性
            avg_feature_consistency = sum(pred_feature_consistency.values()) / len(pred_feature_consistency)
            consistency_scores['pred_feature_consistency'] = avg_feature_consistency
            # 各模态一致性分数
            for modality, score in pred_feature_consistency.items():
                consistency_scores[f"{modality}_feature_consistency"] = score
        
        # 总体一致性分数（各项平均）
        if consistency_scores:
            consistency_scores['overall_consistency'] = sum(consistency_scores.values()) / len(consistency_scores)
        
        return consistency_scores
    
    def _compute_prediction_modality_consistency(self, 
                                               inputs: Dict[str, torch.Tensor], 
                                               predictions: torch.Tensor, 
                                               modality_weights: torch.Tensor) -> float:
        """
        计算预测与模态贡献的一致性
        
        通过屏蔽某个模态并观察预测变化，检验模态重要性与解释是否一致
        
        Args:
            inputs: 输入字典
            predictions: 原始预测结果
            modality_weights: 模态贡献权重
            
        Returns:
            一致性分数 [0,1]
        """
        batch_size = predictions.size(0)
        device = predictions.device
        modalities = ['text', 'audio', 'video']
        
        # 获取模态权重
        weights = modality_weights.detach().cpu().numpy()
        
        consistency_scores = []
        for i in range(batch_size):
            # 计算各模态的一致性
            sample_scores = []
            
            # 遍历各模态
            for j, modality in enumerate(modalities):
                if modality in inputs:
                    # 模态屏蔽：创建一个新的输入，屏蔽当前模态
                    masked_inputs = {k: v.clone() for k, v in inputs.items()}
                    masked_inputs[modality] = torch.zeros_like(inputs[modality])
                    
                    # 使用屏蔽后的输入进行预测
                    with torch.no_grad():
                        masked_outputs = self.model(**masked_inputs)
                        masked_predictions = torch.argmax(masked_outputs, dim=1)
                    
                    # 预测变化率
                    prediction_change = (predictions[i] != masked_predictions[i]).float().item()
                    
                    # 计算一致性：模态权重与预测变化的相关性
                    # 权重越高，预测变化应越明显
                    modality_weight = weights[i, j]
                    
                    # 理想情况下，预测变化应与权重成正比
                    # 当权重高且预测变化大，或权重低且预测变化小时，一致性高
                    consistency = 1.0 - abs(modality_weight - prediction_change)
                    sample_scores.append(consistency)
            
            if sample_scores:
                consistency_scores.append(sum(sample_scores) / len(sample_scores))
        
        # 返回平均一致性
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _compute_prediction_feature_consistency(self, 
                                              x: torch.Tensor, 
                                              predictions: torch.Tensor, 
                                              feature_importance: torch.Tensor) -> float:
        """
        计算预测与特征重要性的一致性
        
        通过扰动重要特征并观察预测变化，检验特征重要性与解释是否一致
        
        Args:
            x: 输入特征
            predictions: 原始预测结果
            feature_importance: 特征重要性
            
        Returns:
            一致性分数 [0,1]
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # 获取特征重要性
        importance = feature_importance.detach().cpu().numpy()
        
        consistency_scores = []
        for i in range(batch_size):
            # 获取最重要的特征索引（前10%）
            feature_dim = importance[i].shape[0]
            top_k = max(1, int(feature_dim * 0.1))
            top_indices = np.argsort(importance[i])[-top_k:]
            
            # 创建扰动输入：仅扰动重要特征
            perturbed_x = x.clone()
            for idx in top_indices:
                if idx < perturbed_x.shape[1]:  # 确保索引在范围内
                    perturbed_x[i, idx] = torch.zeros_like(perturbed_x[i, idx])
            
            # 使用扰动后的输入进行预测
            with torch.no_grad():
                perturbed_outputs = self.model(perturbed_x)
                perturbed_predictions = torch.argmax(perturbed_outputs, dim=1)
            
            # 计算预测变化
            prediction_change = (predictions[i] != perturbed_predictions[i]).float().item()
            
            # 高预测变化表示特征重要性解释一致
            consistency_scores.append(prediction_change)
        
        # 返回平均一致性
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def mine_boundary_samples(self, 
                             dataloader, 
                             threshold: float = 0.1) -> Dict:
        """
        挖掘边界样本
        
        寻找模型预测置信度低的样本，这些样本更可能位于决策边界附近
        
        Args:
            dataloader: 数据加载器
            threshold: 置信度阈值，低于此值的样本被视为边界样本
            
        Returns:
            包含边界样本信息的字典
        """
        self.model.eval()
        boundary_samples = []
        boundary_labels = []
        boundary_confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 提取输入和标签
                inputs = {
                    'text': batch['text'].to(self.model.device),
                    'audio': batch['audioframes'].to(self.model.device),
                    'video': batch['frames'].to(self.model.device)
                }
                labels = batch['label'].to(self.model.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 计算置信度
                confidences = torch.max(F.softmax(outputs, dim=1), dim=1)[0]
                
                # 找出置信度低的样本
                for i, conf in enumerate(confidences):
                    if conf < threshold:
                        # 收集边界样本
                        sample_inputs = {k: v[i:i+1].cpu() for k, v in inputs.items()}
                        boundary_samples.append(sample_inputs)
                        boundary_labels.append(labels[i].cpu())
                        boundary_confidences.append(conf.item())
                        
                        # 达到最大样本数时停止
                        if len(boundary_samples) >= self.max_boundary_samples:
                            break
                
                if len(boundary_samples) >= self.max_boundary_samples:
                    break
        
        # 更新边界样本池
        self.boundary_samples_pool['samples'] = boundary_samples
        self.boundary_samples_pool['labels'] = boundary_labels
        self.boundary_samples_pool['confidence'] = boundary_confidences
        
        return {
            'num_boundary_samples': len(boundary_samples),
            'avg_confidence': np.mean(boundary_confidences) if boundary_confidences else 0.0
        }
    
    def evaluate_adversarial_robustness(self, 
                                       dataloader, 
                                       attack_types: List[str] = ['untargeted'],
                                       num_samples: int = 100) -> Dict[str, float]:
        """
        评估模型对抗鲁棒性
        
        Args:
            dataloader: 数据加载器
            attack_types: 攻击类型列表，可包含 'untargeted' 和 'targeted'
            num_samples: 评估样本数量
            
        Returns:
            包含鲁棒性评分的字典
        """
        self.model.eval()
        
        results = {}
        processed_samples = 0
        
        # 对抗准确率
        adv_correct = {attack_type: 0 for attack_type in attack_types}
        
        for batch in dataloader:
            if processed_samples >= num_samples:
                break
            
            # 提取输入和标签
            inputs = {
                'text': batch['text'].to(self.model.device),
                'audio': batch['audioframes'].to(self.model.device),
                'video': batch['frames'].to(self.model.device)
            }
            labels = batch['label'].to(self.model.device)
            
            batch_size = labels.size(0)
            remaining = num_samples - processed_samples
            if batch_size > remaining:
                # 调整批次大小，避免超过要求的样本数
                inputs = {k: v[:remaining] for k, v in inputs.items()}
                labels = labels[:remaining]
                batch_size = remaining
            
            # 原始准确率
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs, dim=1)
                original_correct = (predictions == labels).sum().item()
            
            # 对抗样本准确率
            for attack_type in attack_types:
                if attack_type == 'untargeted':
                    # 非目标攻击
                    adv_inputs = self.generate_adversarial_samples(inputs, labels, targeted=False)
                elif attack_type == 'targeted':
                    # 目标攻击（默认为翻转标签）
                    target_labels = 1 - labels
                    adv_inputs = self.generate_adversarial_samples(
                        inputs, labels, targeted=True, target_labels=target_labels
                    )
                
                # 对抗准确率
                with torch.no_grad():
                    adv_outputs = self.model(**adv_inputs)
                    adv_predictions = torch.argmax(adv_outputs, dim=1)
                    adv_correct[attack_type] += (adv_predictions == labels).sum().item()
            
            processed_samples += batch_size
        
        # 计算鲁棒性评分
        results['clean_acc'] = original_correct / processed_samples
        for attack_type in attack_types:
            results[f'{attack_type}_robust_acc'] = adv_correct[attack_type] / processed_samples
            # 鲁棒性评分：对抗准确率 / 原始准确率
            if results['clean_acc'] > 0:
                results[f'{attack_type}_robustness'] = results[f'{attack_type}_robust_acc'] / results['clean_acc']
            else:
                results[f'{attack_type}_robustness'] = 0.0
        
        return results 