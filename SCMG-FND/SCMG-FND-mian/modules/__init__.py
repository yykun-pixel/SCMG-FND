"""
模块化接口 - 隐藏核心实现细节

本模块提供抽象化的接口，核心实现细节已被保护。
"""

from typing import Protocol, Dict, Any, Optional
import torch

class IRuleEngine(Protocol):
    """规则引擎接口 - 抽象化实现细节"""
    
    def apply_rules(self, 
                   features: torch.Tensor,
                   prediction: torch.Tensor,
                   context: Optional[Dict[str, Any]] = None) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        应用规则调整
        
        Args:
            features: 输入特征
            prediction: 模型预测
            context: 上下文信息（可选）
            
        Returns:
            调整后的特征、预测和规则信息
        """
        ...

class IContrastLearner(Protocol):
    """对比学习接口 - 抽象化实现细节"""
    
    def compute_contrast_loss(self,
                            text_features: torch.Tensor,
                            audio_features: torch.Tensor,
                            video_features: torch.Tensor,
                            global_features: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            text_features: 文本特征
            audio_features: 音频特征
            video_features: 视频特征
            global_features: 全局特征
            labels: 标签
            
        Returns:
            对比损失
        """
        ...

class IAdversarialVerifier(Protocol):
    """对抗验证接口 - 抽象化实现细节"""
    
    def verify_features(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        验证特征
        
        Args:
            features: 输入特征
            
        Returns:
            验证结果
        """
        ...

# 延迟导入，隐藏实现细节
_rule_engine_impl = None
_contrast_learner_impl = None
_adversarial_verifier_impl = None

def get_rule_engine(config: Optional[Dict[str, Any]] = None) -> IRuleEngine:
    """
    获取规则引擎实例（工厂方法）
    
    注意：实际实现细节已被隐藏，仅返回接口对象
    """
    global _rule_engine_impl
    if _rule_engine_impl is None:
        # 延迟导入，隐藏实现
        from ._rule_engine_impl import RuleEngineImpl
        _rule_engine_impl = RuleEngineImpl(config)
    return _rule_engine_impl

def get_contrast_learner(config: Optional[Dict[str, Any]] = None) -> IContrastLearner:
    """
    获取对比学习器实例（工厂方法）
    
    注意：实际实现细节已被隐藏，仅返回接口对象
    """
    global _contrast_learner_impl
    if _contrast_learner_impl is None:
        # 延迟导入，隐藏实现
        from ._contrast_learner_impl import ContrastLearnerImpl
        _contrast_learner_impl = ContrastLearnerImpl(config)
    return _contrast_learner_impl

def get_adversarial_verifier(config: Optional[Dict[str, Any]] = None) -> IAdversarialVerifier:
    """
    获取对抗验证器实例（工厂方法）
    
    注意：实际实现细节已被隐藏，仅返回接口对象
    """
    global _adversarial_verifier_impl
    if _adversarial_verifier_impl is None:
        # 延迟导入，隐藏实现
        from ._adversarial_verifier_impl import AdversarialVerifierImpl
        _adversarial_verifier_impl = AdversarialVerifierImpl(config)
    return _adversarial_verifier_impl

__all__ = [
    'IRuleEngine',
    'IContrastLearner', 
    'IAdversarialVerifier',
    'get_rule_engine',
    'get_contrast_learner',
    'get_adversarial_verifier'
]
