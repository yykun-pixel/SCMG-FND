"""
神经符号规则引擎实现（内部模块）

注意：此模块包含实现细节，但已移除敏感参数。
实际权重和阈值需要通过配置文件或环境变量提供。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os

class RuleEngineImpl(nn.Module):
    """
    神经符号规则引擎实现
    
    注意：敏感参数（权重、阈值）已从代码中移除，
    需要通过外部配置提供。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 从配置或环境变量加载参数（不硬编码）
        self.config = config or {}
        
        # 参数通过配置加载，不直接暴露
        self._load_parameters()
        
        # 规则库结构（关键词列表保留，但权重移除）
        self.rule_categories = self._initialize_rule_categories()
        
        # 可学习的自适应参数
        num_rule_types = len(self.rule_categories)
        self.adaptive_weights = nn.Parameter(torch.ones(num_rule_types))
        self.threshold_adjustment = nn.Parameter(torch.tensor(0.5))
    
    def _load_parameters(self):
        """从配置或环境变量加载参数"""
        # 权重和阈值不硬编码，从配置读取
        self.rule_weights = self.config.get('rule_weights', {})
        self.rule_threshold = float(os.getenv('RULE_THRESHOLD', 
                                             self.config.get('rule_threshold', '0.05')))
        
        # 如果没有提供配置，使用默认结构（但无具体数值）
        if not self.rule_weights:
            self.rule_weights = self._get_default_structure()
    
    def _get_default_structure(self) -> Dict:
        """
        返回默认规则结构（不含具体权重值）
        
        注意：实际权重值需要通过配置文件提供
        """
        return {
            "strong_negative": {"keywords": [], "weight": None, "bias": None},
            "moderate_negative": {"keywords": [], "weight": None, "bias": None},
            "strong_positive": {"keywords": [], "weight": None, "bias": None},
            "moderate_positive": {"keywords": [], "weight": None, "bias": None}
        }
    
    def _initialize_rule_categories(self) -> Dict:
        """
        初始化规则类别（仅结构，不含权重）
        
        注意：具体关键词和权重需要通过配置文件提供
        """
        # 从配置文件加载规则库
        rule_file = self.config.get('rule_library_path', 'config/rules.json')
        
        if os.path.exists(rule_file):
            import json
            with open(rule_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回空结构，提示需要配置文件
            return {
                "strong_negative": {"keywords": []},
                "moderate_negative": {"keywords": []},
                "strong_positive": {"keywords": []},
                "moderate_positive": {"keywords": []}
            }
    
    def extract_implicit_opinions(self, text_analysis: Dict) -> Dict:
        """
        从隐式意见分析中提取关键信息
        
        注意：权重计算逻辑保留，但具体权重值从配置读取
        """
        matched_rules = {category: [] for category in self.rule_categories.keys()}
        dimension_matches = {}
        total_weight = 0.0
        total_bias = 0.0
        
        # 遍历分析结果
        for dimension, analysis in text_analysis.get("虚假新闻特征检测", {}).items():
            if isinstance(analysis, dict):
                keyword = analysis.get("关键词", "")
                analysis_text = analysis.get("分析", "")
                
                # 匹配规则（权重从配置读取）
                full_text = f"{keyword} {analysis_text}"
                for rule_type, rule_config in self.rule_categories.items():
                    for rule_keyword in rule_config.get("keywords", []):
                        if rule_keyword in full_text:
                            # 从配置读取权重
                            weight = self.rule_weights.get(rule_type, {}).get("weight", 0.0)
                            bias = self.rule_weights.get(rule_type, {}).get("bias", 0.0)
                            
                            matched_rules[rule_type].append({
                                "dimension": dimension,
                                "keyword": rule_keyword,
                                "context": analysis_text[:100]
                            })
                            
                            total_weight += weight
                            total_bias += bias
        
        return {
            "matched_rules": matched_rules,
            "dimension_matches": dimension_matches,
            "total_weight": total_weight,
            "total_bias": total_bias
        }
    
    def compute_rule_adjustment(self, rule_matches: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于规则匹配结果计算调整参数
        
        注意：调整公式保留，但具体系数从配置读取
        """
        base_weight_adj = rule_matches["total_weight"]
        base_bias_adj = rule_matches["total_bias"]
        
        # 应用可学习的自适应权重
        adaptive_factor = torch.sigmoid(self.adaptive_weights).mean()
        
        # 调整系数从配置读取
        adjustment_factor = self.config.get('adjustment_factor', 0.5)
        
        weight_adjustment = torch.tensor(
            base_weight_adj * adaptive_factor * adjustment_factor,
            dtype=torch.float32
        )
        
        prediction_bias = torch.tensor(
            base_bias_adj * adaptive_factor + self.threshold_adjustment,
            dtype=torch.float32
        )
        
        return weight_adjustment, prediction_bias
    
    def apply_rules(self, 
                   features: torch.Tensor,
                   prediction: torch.Tensor,
                   context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        应用规则调整（公共接口）
        """
        if context is None:
            context = {}
        
        # 提取隐式意见
        implicit_analysis = context.get('implicit_opinion_analysis')
        if implicit_analysis is None:
            # 无规则应用，返回原始结果
            return features, prediction, {"matched_rules_count": 0}
        
        # 提取规则匹配
        rule_matches = self.extract_implicit_opinions(implicit_analysis)
        
        # 计算调整
        weight_adj, bias_adj = self.compute_rule_adjustment(rule_matches)
        
        # 应用调整（仅当超过阈值时）
        if abs(bias_adj.item()) > self.rule_threshold:
            feature_multiplier = 1.0 + torch.tanh(weight_adj)
            adjusted_features = features * feature_multiplier.expand_as(features)
            
            prediction_logits = torch.log(prediction + 1e-8)
            adjusted_logits = prediction_logits + bias_adj
            adjusted_prediction = torch.softmax(adjusted_logits, dim=-1)
        else:
            adjusted_features = features
            adjusted_prediction = prediction
        
        rule_info = {
            "matched_rules_count": sum(len(rules) for rules in rule_matches["matched_rules"].values()),
            "weight_adjustment": weight_adj.item(),
            "bias_adjustment": bias_adj.item()
        }
        
        return adjusted_features, adjusted_prediction, rule_info

