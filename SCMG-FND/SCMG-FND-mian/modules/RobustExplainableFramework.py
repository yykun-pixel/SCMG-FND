import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from modules.MultiGranularityContrast import MultiGranularityContrast
from modules.AdversarialVerifier import AdversarialVerifier
from Diffusion.ExplainableDetection import ExplainableDetection

class RobustExplainableFramework:
    """
    鲁棒可解释框架
    
    将多粒度对比学习和对抗性验证框架集成到现有的可解释模型中，
    增强模型的鲁棒性和可解释性。
    """
    
    def __init__(self, 
                model: nn.Module,
                feature_dim: int = 768,
                enable_contrast: bool = True,
                enable_adv_verify: bool = True,
                contrast_granularity: str = 'all',  # 'spatial', 'temporal', 'modal', 'all'
                contrast_weight: float = 0.1,
                adv_eps: float = 0.01,
                adv_steps: int = 3,
                device: torch.device = None):
        """
        初始化鲁棒可解释框架
        
        Args:
            model: 主模型
            feature_dim: 特征维度
            enable_contrast: 是否启用多粒度对比学习
            enable_adv_verify: 是否启用对抗性验证
            contrast_granularity: 对比学习粒度
            contrast_weight: 对比损失权重
            adv_eps: 对抗扰动大小
            adv_steps: 对抗攻击步数
            device: 计算设备
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_contrast = enable_contrast
        self.enable_adv_verify = enable_adv_verify
        self.contrast_granularity = contrast_granularity
        self.contrast_weight = contrast_weight
        
        # 获取模型的可解释模块
        self.explainable_module = None
        for module in self.model.modules():
            if isinstance(module, ExplainableDetection):
                self.explainable_module = module
                break
        
        if self.explainable_module is None:
            print("Warning: No ExplainableDetection module found in the model.")
        
        # 初始化多粒度对比学习模块
        if self.enable_contrast:
            self.contrast_module = MultiGranularityContrast(
                feature_dim=feature_dim,
                temperature=0.07,
                spatial_regions=8,   # 空间区域数量
                temporal_segments=4,  # 时间段数量
                modal_components=3    # 模态数量（文本、音频、视频）
            ).to(self.device)
        
        # 初始化对抗性验证框架
        if self.enable_adv_verify:
            self.adv_verifier = AdversarialVerifier(
                model=self.model,
                eps=adv_eps,
                steps=adv_steps
            ).to(self.device)
    
    def compute_loss(self, 
                    inputs: Dict[str, torch.Tensor], 
                    labels: torch.Tensor,
                    outputs: torch.Tensor,
                    original_loss: torch.Tensor,
                    explanations: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算增强损失
        
        Args:
            inputs: 输入数据
            labels: 标签
            outputs: 模型输出
            original_loss: 原始损失
            explanations: 解释结果
            
        Returns:
            增强后的总损失和损失详情
        """
        loss_details = {'original_loss': original_loss.item()}
        total_loss = original_loss
        
        # 对比损失
        if self.enable_contrast:
            # 提取特征
            features = {}
            
            # 从模型中获取各模态特征
            # 假设模型中存储了中间特征
            if hasattr(self.model, 'text_features'):
                features['text'] = self.model.text_features
            if hasattr(self.model, 'audio_features'):
                features['audio'] = self.model.audio_features
            if hasattr(self.model, 'video_features'):
                features['video'] = self.model.video_features
            
            # 如果模型没有存储中间特征，可以从解释模块获取
            if not features and self.explainable_module and hasattr(self.explainable_module, 'last_features'):
                features = self.explainable_module.last_features
            
            # 计算对比损失
            if features:
                contrast_loss = self.contrast_module(
                    features=features,
                    labels=labels,
                    granularity=self.contrast_granularity
                )
                
                loss_details['contrast_loss'] = contrast_loss.item()
                total_loss = total_loss + self.contrast_weight * contrast_loss
        
        # 对抗验证
        if self.enable_adv_verify and self.training:
            # 生成对抗样本
            with torch.no_grad():
                adv_inputs = self.adv_verifier.generate_adversarial_samples(
                    inputs=inputs,
                    labels=labels,
                    targeted=False  # 非目标攻击
                )
            
            # 使用对抗样本前向传播
            adv_outputs = self.model(**adv_inputs)
            
            # 计算对抗损失
            adv_loss = F.cross_entropy(adv_outputs, labels)
            
            # 如果启用了解释，验证一致性
            if explanations is not None:
                # 生成对抗解释
                with torch.no_grad():
                    adv_explanations = self.model.generate_explanation(**adv_inputs)
                
                # 验证预测与解释的一致性
                consistency_scores = self.adv_verifier.verify_consistency(
                    inputs=inputs, 
                    explanations=explanations
                )
                
                # 计算一致性损失：鼓励高一致性
                consistency_loss = 1.0 - consistency_scores.get('overall_consistency', 0.0)
                consistency_loss = torch.tensor(consistency_loss, device=self.device)
                
                loss_details['consistency_loss'] = consistency_loss.item()
                
                # 添加一致性损失
                adv_loss = adv_loss + 0.1 * consistency_loss
            
            loss_details['adv_loss'] = adv_loss.item()
            total_loss = total_loss + 0.5 * adv_loss  # 对抗损失权重0.5
        
        return total_loss, loss_details
    
    def analyze_sample(self, 
                      inputs: Dict[str, torch.Tensor], 
                      labels: torch.Tensor,
                      predictions: torch.Tensor,
                      explanations: Dict[str, torch.Tensor]) -> Dict:
        """
        分析样本，提供鲁棒性和可解释性指标
        
        Args:
            inputs: 输入数据
            labels: 真实标签
            predictions: 模型预测
            explanations: 解释结果
            
        Returns:
            分析结果字典
        """
        results = {}
        
        # 预测正确性
        correct = (predictions == labels).float()
        results['accuracy'] = correct.mean().item()
        
        # 一致性验证
        if self.enable_adv_verify:
            consistency_scores = self.adv_verifier.verify_consistency(
                inputs=inputs,
                explanations=explanations
            )
            results.update(consistency_scores)
            
            # 评估对抗鲁棒性（对于小批次）
            # 创建简单的dataloader包装当前批次
            from torch.utils.data import TensorDataset, DataLoader
            
            # 将当前批次转换为临时dataloader
            temp_dataset = TensorDataset(
                inputs['text'], inputs['audio'], inputs['video'], labels
            )
            temp_loader = DataLoader(temp_dataset, batch_size=len(labels))
            
            # 评估对抗鲁棒性
            robustness_scores = self.adv_verifier.evaluate_adversarial_robustness(
                temp_loader,
                attack_types=['untargeted'],
                num_samples=len(labels)
            )
            results.update(robustness_scores)
        
        # 可解释性指标
        # 1. 模态贡献分布
        if 'modality_weights' in explanations:
            modality_weights = explanations['modality_weights']
            results['modality_diversity'] = torch.std(modality_weights, dim=1).mean().item()
        
        # 2. 特征稀疏性
        feature_importances = {
            'text': explanations.get('text_importance', None),
            'audio': explanations.get('audio_importance', None),
            'video': explanations.get('video_importance', None)
        }
        
        sparsity_scores = {}
        for modality, importance in feature_importances.items():
            if importance is not None:
                # 计算特征稀疏性：零元素比例
                sparsity = (importance.abs() < 0.01).float().mean().item()
                sparsity_scores[f'{modality}_sparsity'] = sparsity
        
        if sparsity_scores:
            results.update(sparsity_scores)
            results['avg_sparsity'] = sum(sparsity_scores.values()) / len(sparsity_scores)
        
        return results
    
    def mine_boundary_samples(self, dataloader, threshold: float = 0.1) -> Dict:
        """
        挖掘边界样本
        
        Args:
            dataloader: 数据加载器
            threshold: 置信度阈值
            
        Returns:
            边界样本统计
        """
        if self.enable_adv_verify:
            return self.adv_verifier.mine_boundary_samples(dataloader, threshold)
        return {'num_boundary_samples': 0, 'avg_confidence': 0.0}
    
    def generate_adversarial_examples(self, 
                                     inputs: Dict[str, torch.Tensor], 
                                     labels: torch.Tensor,
                                     targeted: bool = False) -> Dict[str, torch.Tensor]:
        """
        生成对抗样本
        
        Args:
            inputs: 输入数据
            labels: 标签
            targeted: 是否为目标攻击
            
        Returns:
            对抗样本
        """
        if self.enable_adv_verify:
            return self.adv_verifier.generate_adversarial_samples(inputs, labels, targeted)
        return inputs 