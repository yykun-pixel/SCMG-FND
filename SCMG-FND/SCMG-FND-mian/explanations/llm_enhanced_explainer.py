#!/usr/bin/env python3
"""
LLM增强可解释性模块
结合传统可解释性方法和大语言模型的语义推理能力
提供更全面、更易理解的决策解释
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import time
from abc import ABC, abstractmethod

@dataclass
class ExplanationContext:
    """解释上下文数据结构"""
    video_id: str
    model_prediction: torch.Tensor
    confidence_score: float
    
    # 传统可解释性信息
    modality_weights: Dict[str, float]
    feature_importance: Dict[str, np.ndarray]
    fake_regions: Optional[np.ndarray]
    attention_maps: Dict[str, np.ndarray]
    
    # 神经符号规则信息
    neural_symbolic_info: Dict[str, Any]
    matched_rules: List[Dict[str, Any]]
    rule_application_history: List[Dict[str, Any]]
    
    # 隐式意见分析
    implicit_opinion_analysis: Dict[str, Any]
    feature_analysis: Dict[str, Any]
    
    # 元数据
    video_metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

class LLMProvider(ABC):
    """LLM提供者抽象基类"""
    
    @abstractmethod
    def generate_explanation(self, prompt: str, context: ExplanationContext) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT提供者"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_explanation(self, prompt: str, context: ExplanationContext) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的虚假视频检测解释专家，能够基于技术分析结果提供清晰、准确和有洞察力的解释。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"LLM解释生成失败: {str(e)}"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class LocalLLMProvider(LLMProvider):
    """本地LLM提供者（支持Ollama等）"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    def generate_explanation(self, prompt: str, context: ExplanationContext) -> str:
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 800}
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=data, timeout=60)
            response.raise_for_status()
            return response.json().get('response', '本地LLM响应解析失败')
        except Exception as e:
            return f"本地LLM解释生成失败: {str(e)}"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class HybridExplainer:
    """混合可解释性系统 - 结合传统方法和LLM"""
    
    def __init__(self, 
                 llm_provider: LLMProvider,
                 fallback_to_traditional: bool = True,
                 cache_explanations: bool = True,
                 explanation_templates: Optional[Dict[str, str]] = None):
        """
        初始化混合解释器
        
        Args:
            llm_provider: LLM提供者
            fallback_to_traditional: 当LLM不可用时是否回退到传统方法
            cache_explanations: 是否缓存解释结果
            explanation_templates: 自定义解释模板
        """
        self.llm_provider = llm_provider
        self.fallback_to_traditional = fallback_to_traditional
        self.cache_explanations = cache_explanations
        self.explanation_cache = {}
        
        # 默认解释模板
        self.templates = explanation_templates or {
            "decision_summary": self._get_decision_summary_template(),
            "rule_reasoning": self._get_rule_reasoning_template(),
            "confidence_analysis": self._get_confidence_analysis_template(),
            "risk_assessment": self._get_risk_assessment_template()
        }
    
    def generate_comprehensive_explanation(self, context: ExplanationContext) -> Dict[str, Any]:
        """
        生成综合解释报告
        
        Args:
            context: 解释上下文
            
        Returns:
            包含多层次解释的字典
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._generate_cache_key(context)
        if self.cache_explanations and cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        explanation_result = {
            "video_id": context.video_id,
            "timestamp": time.time(),
            "model_prediction": {
                "class": torch.argmax(context.model_prediction).item(),
                "confidence": context.confidence_score,
                "probabilities": context.model_prediction.tolist()
            }
        }
        
        # 1. 生成决策摘要
        explanation_result["decision_summary"] = self._generate_decision_summary(context)
        
        # 2. 神经符号规则推理
        explanation_result["rule_reasoning"] = self._generate_rule_reasoning(context)
        
        # 3. 置信度分析
        explanation_result["confidence_analysis"] = self._generate_confidence_analysis(context)
        
        # 4. 风险评估
        explanation_result["risk_assessment"] = self._generate_risk_assessment(context)
        
        # 5. 传统可解释性补充
        explanation_result["technical_details"] = self._generate_technical_details(context)
        
        # 6. 用户友好总结
        explanation_result["user_friendly_summary"] = self._generate_user_summary(context, explanation_result)
        
        explanation_result["processing_time"] = time.time() - start_time
        
        # 缓存结果
        if self.cache_explanations:
            self.explanation_cache[cache_key] = explanation_result
        
        return explanation_result
    
    def _generate_decision_summary(self, context: ExplanationContext) -> Dict[str, Any]:
        """生成决策摘要"""
        prompt = self.templates["decision_summary"].format(
            video_id=context.video_id,
            prediction=torch.argmax(context.model_prediction).item(),
            confidence=context.confidence_score,
            modality_weights=json.dumps(context.modality_weights, ensure_ascii=False, indent=2),
            neural_symbolic_info=json.dumps(context.neural_symbolic_info, ensure_ascii=False, indent=2)
        )
        
        if self.llm_provider.is_available():
            llm_explanation = self.llm_provider.generate_explanation(prompt, context)
            return {
                "source": "llm",
                "explanation": llm_explanation,
                "raw_data": {
                    "prediction_class": torch.argmax(context.model_prediction).item(),
                    "confidence": context.confidence_score
                }
            }
        elif self.fallback_to_traditional:
            return self._traditional_decision_summary(context)
        else:
            return {"source": "unavailable", "explanation": "LLM服务不可用"}
    
    def _generate_rule_reasoning(self, context: ExplanationContext) -> Dict[str, Any]:
        """生成规则推理解释"""
        if not context.neural_symbolic_info or not context.matched_rules:
            return {"source": "none", "explanation": "未应用神经符号规则"}
        
        prompt = self.templates["rule_reasoning"].format(
            matched_rules=json.dumps(context.matched_rules, ensure_ascii=False, indent=2),
            rule_application=json.dumps(context.neural_symbolic_info, ensure_ascii=False, indent=2),
            implicit_analysis=json.dumps(context.implicit_opinion_analysis, ensure_ascii=False, indent=2)
        )
        
        if self.llm_provider.is_available():
            llm_explanation = self.llm_provider.generate_explanation(prompt, context)
            return {
                "source": "llm",
                "explanation": llm_explanation,
                "applied_rules": len(context.matched_rules),
                "rule_confidence": context.neural_symbolic_info.get("confidence_boost", 0)
            }
        elif self.fallback_to_traditional:
            return self._traditional_rule_reasoning(context)
        else:
            return {"source": "unavailable", "explanation": "LLM服务不可用"}
    
    def _generate_confidence_analysis(self, context: ExplanationContext) -> Dict[str, Any]:
        """生成置信度分析"""
        prompt = self.templates["confidence_analysis"].format(
            confidence=context.confidence_score,
            feature_analysis=json.dumps(context.feature_analysis, ensure_ascii=False, indent=2),
            modality_weights=json.dumps(context.modality_weights, ensure_ascii=False, indent=2)
        )
        
        if self.llm_provider.is_available():
            llm_explanation = self.llm_provider.generate_explanation(prompt, context)
            return {
                "source": "llm", 
                "explanation": llm_explanation,
                "confidence_level": self._categorize_confidence(context.confidence_score)
            }
        elif self.fallback_to_traditional:
            return self._traditional_confidence_analysis(context)
        else:
            return {"source": "unavailable", "explanation": "LLM服务不可用"}
    
    def _generate_risk_assessment(self, context: ExplanationContext) -> Dict[str, Any]:
        """生成风险评估"""
        prompt = self.templates["risk_assessment"].format(
            prediction=torch.argmax(context.model_prediction).item(),
            confidence=context.confidence_score,
            implicit_analysis=json.dumps(context.implicit_opinion_analysis, ensure_ascii=False, indent=2)
        )
        
        if self.llm_provider.is_available():
            llm_explanation = self.llm_provider.generate_explanation(prompt, context)
            return {
                "source": "llm",
                "explanation": llm_explanation,
                "risk_level": self._calculate_risk_level(context)
            }
        elif self.fallback_to_traditional:
            return self._traditional_risk_assessment(context)
        else:
            return {"source": "unavailable", "explanation": "LLM服务不可用"}
    
    def _generate_technical_details(self, context: ExplanationContext) -> Dict[str, Any]:
        """生成技术细节（传统可解释性）"""
        return {
            "modality_contributions": context.modality_weights,
            "feature_importance_stats": {
                modality: {
                    "max": float(np.max(importance)),
                    "mean": float(np.mean(importance)),
                    "std": float(np.std(importance))
                }
                for modality, importance in context.feature_importance.items()
            },
            "attention_statistics": {
                key: {
                    "shape": attention.shape,
                    "max_attention": float(np.max(attention)),
                    "attention_entropy": float(-np.sum(attention * np.log(attention + 1e-8)))
                }
                for key, attention in context.attention_maps.items()
            },
            "neural_symbolic_metrics": context.neural_symbolic_info
        }
    
    def _generate_user_summary(self, context: ExplanationContext, full_explanation: Dict[str, Any]) -> str:
        """生成用户友好的总结"""
        prediction_class = torch.argmax(context.model_prediction).item()
        prediction_text = "虚假视频" if prediction_class == 1 else "真实视频"
        confidence_text = f"{context.confidence_score*100:.1f}%"
        
        summary_parts = [
            f"🎯 **检测结果**: {prediction_text} (置信度: {confidence_text})",
        ]
        
        # 添加主要依据
        if context.modality_weights:
            dominant_modality = max(context.modality_weights.items(), key=lambda x: x[1])
            summary_parts.append(f"📊 **主要依据**: {dominant_modality[0]}模态 ({dominant_modality[1]*100:.1f}%贡献度)")
        
        # 添加规则应用情况
        if context.neural_symbolic_info and context.neural_symbolic_info.get("matched_rules_count", 0) > 0:
            rule_count = context.neural_symbolic_info["matched_rules_count"]
            summary_parts.append(f"⚖️ **规则匹配**: 应用了{rule_count}条神经符号规则")
        
        # 添加风险等级
        risk_level = full_explanation.get("risk_assessment", {}).get("risk_level", "未知")
        summary_parts.append(f"⚠️ **风险等级**: {risk_level}")
        
        return "\n".join(summary_parts)
    
    def _get_decision_summary_template(self) -> str:
        return """
基于以下技术分析结果，请生成一个清晰的决策摘要解释：

视频ID: {video_id}
预测结果: {prediction} (0=真实, 1=虚假)
置信度: {confidence:.3f}

模态权重分析:
{modality_weights}

神经符号规则信息:
{neural_symbolic_info}

请用专业但易懂的语言解释：
1. 模型为什么做出这个判断？
2. 哪些因素是决定性的？
3. 这个判断的可靠性如何？

请用中文回答，保持客观和专业。
"""
    
    def _get_rule_reasoning_template(self) -> str:
        return """
基于以下神经符号规则应用情况，请解释规则推理过程：

匹配的规则:
{matched_rules}

规则应用结果:
{rule_application}

隐式意见分析:
{implicit_analysis}

请解释：
1. 哪些规则被触发了？为什么？
2. 这些规则如何影响最终判断？
3. 规则应用的合理性如何？

请用中文回答，重点解释规则逻辑。
"""
    
    def _get_confidence_analysis_template(self) -> str:
        return """
基于以下信息分析模型置信度：

当前置信度: {confidence:.3f}

特征分析:
{feature_analysis}

模态权重:
{modality_weights}

请分析：
1. 这个置信度水平意味着什么？
2. 哪些因素影响了置信度？
3. 是否存在不确定性？

请用中文回答，帮助用户理解置信度的含义。
"""
    
    def _get_risk_assessment_template(self) -> str:
        return """
基于检测结果进行风险评估：

预测结果: {prediction}
置信度: {confidence:.3f}

详细分析:
{implicit_analysis}

请评估：
1. 如果判断错误的潜在风险是什么？
2. 建议采取什么后续行动？
3. 需要人工核查吗？

请提供实用的风险评估和建议。
"""
    
    # 传统方法回退函数
    def _traditional_decision_summary(self, context: ExplanationContext) -> Dict[str, Any]:
        prediction = torch.argmax(context.model_prediction).item()
        prediction_text = "虚假视频" if prediction == 1 else "真实视频"
        
        explanation = f"模型预测这是{prediction_text}，置信度为{context.confidence_score:.3f}。"
        
        if context.modality_weights:
            dominant_modality = max(context.modality_weights.items(), key=lambda x: x[1])
            explanation += f" 主要基于{dominant_modality[0]}模态的证据（贡献度{dominant_modality[1]:.2f}）。"
        
        return {
            "source": "traditional",
            "explanation": explanation,
            "raw_data": {"prediction_class": prediction, "confidence": context.confidence_score}
        }
    
    def _traditional_rule_reasoning(self, context: ExplanationContext) -> Dict[str, Any]:
        rule_count = len(context.matched_rules) if context.matched_rules else 0
        explanation = f"应用了{rule_count}条神经符号规则。"
        
        if context.neural_symbolic_info:
            bias_adj = context.neural_symbolic_info.get("bias_adjustment", 0)
            if abs(bias_adj) > 0.01:
                explanation += f" 规则调整了预测偏置{bias_adj:.3f}。"
        
        return {
            "source": "traditional",
            "explanation": explanation,
            "applied_rules": rule_count
        }
    
    def _traditional_confidence_analysis(self, context: ExplanationContext) -> Dict[str, Any]:
        confidence_level = self._categorize_confidence(context.confidence_score)
        explanation = f"置信度为{context.confidence_score:.3f}，属于{confidence_level}水平。"
        
        return {
            "source": "traditional",
            "explanation": explanation,
            "confidence_level": confidence_level
        }
    
    def _traditional_risk_assessment(self, context: ExplanationContext) -> Dict[str, Any]:
        risk_level = self._calculate_risk_level(context)
        prediction = torch.argmax(context.model_prediction).item()
        
        if prediction == 1 and context.confidence_score > 0.8:
            explanation = "高置信度检测到虚假视频，建议进一步审查。"
        elif prediction == 1 and context.confidence_score < 0.6:
            explanation = "检测到可能的虚假视频，但置信度不高，建议人工核查。"
        else:
            explanation = "预测为真实视频，风险较低。"
        
        return {
            "source": "traditional",
            "explanation": explanation,
            "risk_level": risk_level
        }
    
    # 辅助函数
    def _categorize_confidence(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "高"
        elif confidence >= 0.7:
            return "中等"
        elif confidence >= 0.5:
            return "较低"
        else:
            return "很低"
    
    def _calculate_risk_level(self, context: ExplanationContext) -> str:
        prediction = torch.argmax(context.model_prediction).item()
        confidence = context.confidence_score
        
        if prediction == 1:  # 虚假视频
            if confidence >= 0.8:
                return "高风险"
            elif confidence >= 0.6:
                return "中等风险"
            else:
                return "低风险"
        else:  # 真实视频
            if confidence >= 0.8:
                return "低风险"
            else:
                return "需要关注"
    
    def _generate_cache_key(self, context: ExplanationContext) -> str:
        """生成缓存键"""
        return f"{context.video_id}_{hash(str(context.model_prediction.tolist()))}"

# 工厂函数
def create_llm_explainer(provider_type: str = "local", **kwargs) -> HybridExplainer:
    """
    创建LLM解释器的工厂函数
    
    Args:
        provider_type: "openai" 或 "local"
        **kwargs: 提供者特定的参数
    """
    if provider_type == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI provider requires api_key")
        provider = OpenAIProvider(api_key, kwargs.get("model", "gpt-4"))
    elif provider_type == "local":
        provider = LocalLLMProvider(
            kwargs.get("base_url", "http://localhost:11434"),
            kwargs.get("model", "llama2")
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    return HybridExplainer(
        llm_provider=provider,
        fallback_to_traditional=kwargs.get("fallback_to_traditional", True),
        cache_explanations=kwargs.get("cache_explanations", True)
    )

if __name__ == "__main__":
    # 示例用法
    print("LLM增强可解释性模块已创建")
    print("支持的功能：")
    print("- 基于LLM的语义解释")
    print("- 传统可解释性方法回退")
    print("- 多层次解释（决策摘要、规则推理、置信度分析、风险评估）")
    print("- 解释结果缓存")
    print("- 支持OpenAI和本地LLM") 