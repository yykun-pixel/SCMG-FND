import torch
import torch.nn as nn
import re
import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

class NeuralSymbolicRuleEngine(nn.Module):
    """
    神经符号规则引擎
    
    基于隐式意见中的关键词和逻辑规则，动态调整模型的判断倾向
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 保存配置
        self.config = config or {}
        
        # 规则库结构（权重和阈值从配置加载，不硬编码）
        # 注意：敏感参数已移除，需要通过配置文件或环境变量提供
        self.rule_database = self._load_rule_database()
        self.dimension_rules = self._load_dimension_rules()
    
    def _load_rule_database(self) -> Dict:
        """
        从配置文件加载规则库
        
        注意：权重、偏置等敏感参数不硬编码，需通过配置文件提供
        """
        import os
        import json
        
        rule_config_path = os.getenv('RULE_DATABASE_CONFIG', 'config/rule_database.json')
        
        if os.path.exists(rule_config_path):
            with open(rule_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回结构模板（不含具体权重值）
            return {
                "strong_negative": {
                    "keywords": [],  # 关键词需通过配置提供
                    "weight": None,  # 权重需通过配置提供
                    "bias": None,
                    "confidence_boost": None
                },
                "moderate_negative": {
                    "keywords": [],
                    "weight": None,
                    "bias": None,
                    "confidence_boost": None
                },
                "strong_positive": {
                    "keywords": [],
                    "weight": None,
                    "bias": None,
                    "confidence_boost": None
                },
                "moderate_positive": {
                    "keywords": [],
                    "weight": None,
                    "bias": None,
                    "confidence_boost": None
                }
            }
    
    def _load_dimension_rules(self) -> Dict:
        """
        从配置文件加载维度特定规则
        
        注意：权重和分数值不硬编码
        """
        import os
        import json
        
        dimension_config_path = os.getenv('DIMENSION_RULES_CONFIG', 'config/dimension_rules.json')
        
        if os.path.exists(dimension_config_path):
            with open(dimension_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回结构模板
            return {
                "专业性评估": {},
                "客观事实符合度": {},
                "物理常识符合度": {},
                "情感导向性": {},
                "误导性评估": {}
            }
        
        # 可学习的权重调整参数（数量从配置读取）
        num_rule_types = len(self.rule_database)
        self.adaptive_weights = nn.Parameter(torch.ones(num_rule_types))
        
        # 阈值调整参数（初始值从配置读取，不硬编码）
        initial_threshold = float(os.getenv('INITIAL_THRESHOLD', '0.5'))
        self.threshold_adjustment = nn.Parameter(torch.tensor(initial_threshold))
        
    def extract_implicit_opinions(self, text_analysis: Dict) -> Dict:
        """
        从隐式意见分析中提取关键信息
        
        Args:
            text_analysis: 类似enhanced_results.json中的分析结果
            
        Returns:
            提取的规则匹配结果
        """
        matched_rules = {
            "strong_negative": [],
            "moderate_negative": [],
            "strong_positive": [],
            "moderate_positive": []
        }
        
        dimension_matches = {}
        total_weight = 0.0
        total_bias = 0.0
        confidence_adjustment = 0.0
        
        # 遍历所有维度的分析
        for dimension, analysis in text_analysis.get("虚假新闻特征检测", {}).items():
            if isinstance(analysis, dict):
                keyword = analysis.get("关键词", "")
                analysis_text = analysis.get("分析", "")
                score = float(analysis.get("分数", "2.5"))
                
                # 匹配维度特定规则（权重从配置读取）
                if dimension in self.dimension_rules:
                    for rule_keyword, rule_config in self.dimension_rules[dimension].items():
                        if rule_keyword in keyword or rule_keyword in analysis_text:
                            # 从配置读取，如果不存在则使用默认值
                            fake_score = rule_config.get("fake_score", 0.5) if rule_config.get("fake_score") is not None else 0.5
                            weight = rule_config.get("weight", 0.0) if rule_config.get("weight") is not None else 0.0
                            
                            dimension_matches[dimension] = {
                                "keyword": rule_keyword,
                                "fake_score": fake_score,
                                "weight": weight,
                                "original_score": score
                            }
                
                # 匹配通用规则
                full_text = f"{keyword} {analysis_text}"
                for rule_type, rule_config in self.rule_database.items():
                    for rule_keyword in rule_config["keywords"]:
                        if rule_keyword in full_text:
                            matched_rules[rule_type].append({
                                "dimension": dimension,
                                "keyword": rule_keyword,
                                "context": analysis_text[:100],  # 前100字符作为上下文
                                "score": score
                            })
                            
                            # 累积权重和偏置（从配置读取，可能为None）
                            weight = rule_config.get("weight", 0.0) if rule_config.get("weight") is not None else 0.0
                            bias = rule_config.get("bias", 0.0) if rule_config.get("bias") is not None else 0.0
                            confidence = rule_config.get("confidence_boost", 0.0) if rule_config.get("confidence_boost") is not None else 0.0
                            
                            total_weight += weight
                            total_bias += bias
                            confidence_adjustment += confidence
        
        return {
            "matched_rules": matched_rules,
            "dimension_matches": dimension_matches,
            "total_weight": total_weight,
            "total_bias": total_bias,
            "confidence_adjustment": confidence_adjustment,
            "comprehensive_score": text_analysis.get("综合评分", "3.0")
        }
    
    def compute_rule_adjustment(self, rule_matches: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于规则匹配结果计算调整参数
        
        Returns:
            weight_adjustment: 文本特征权重调整
            prediction_bias: 预测偏置调整
        """
        # 基础调整
        base_weight_adj = rule_matches["total_weight"]
        base_bias_adj = rule_matches["total_bias"]
        
        # 考虑维度匹配的影响
        dimension_weight_adj = 0.0
        for dim, match in rule_matches["dimension_matches"].items():
            dimension_weight_adj += match["weight"] * (1.0 - match["original_score"] / 5.0)
        
        # 应用可学习的权重
        adaptive_factor = torch.sigmoid(self.adaptive_weights).mean()
        
        weight_adjustment = torch.tensor(
            (base_weight_adj + dimension_weight_adj * 0.5) * adaptive_factor,
            dtype=torch.float32
        )
        
        prediction_bias = torch.tensor(
            base_bias_adj * adaptive_factor + self.threshold_adjustment,
            dtype=torch.float32
        )
        
        return weight_adjustment, prediction_bias
    
    def forward(self, 
                text_features: torch.Tensor,
                model_prediction: torch.Tensor,
                implicit_opinion_analysis) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播：应用神经符号规则调整
        
        Args:
            text_features: 原始文本特征
            model_prediction: 模型原始预测
            implicit_opinion_analysis: 隐式意见分析结果（可以是字典或列表）
            
        Returns:
            adjusted_features: 调整后的文本特征
            adjusted_prediction: 调整后的预测
            rule_info: 规则应用信息
        """
        batch_size = text_features.size(0)
        
        # 处理不同类型的隐式意见数据
        if isinstance(implicit_opinion_analysis, list):
            # 批次数据：列表格式
            print(f"🔍 处理批次数据，样本数: {len(implicit_opinion_analysis)}")
            all_rule_matches = []
            valid_samples = 0
            
            for i, sample_analysis in enumerate(implicit_opinion_analysis):
                if sample_analysis is not None:
                    rule_matches = self.extract_implicit_opinions(sample_analysis)
                    all_rule_matches.append(rule_matches)
                    valid_samples += 1
                else:
                    # 对于None样本，创建空的规则匹配
                    all_rule_matches.append({
                        "matched_rules": {
                            "strong_negative": [], "moderate_negative": [],
                            "strong_positive": [], "moderate_positive": []
                        },
                        "dimension_matches": {},
                        "total_weight": 0.0,
                        "total_bias": 0.0,
                        "confidence_adjustment": 0.0,
                        "comprehensive_score": "3.0"
                    })
            
            print(f"🔍 有效样本数: {valid_samples}")
            
            # 汇总批次中所有样本的规则匹配
            combined_matched_rules = {
                "strong_negative": [],
                "moderate_negative": [],
                "strong_positive": [],
                "moderate_positive": []
            }
            
            combined_dimension_matches = {}
            total_weight = 0.0
            total_bias = 0.0
            total_confidence = 0.0
            comprehensive_scores = []
            
            for rule_match in all_rule_matches:
                if "matched_rules" in rule_match:
                    for category, matches in rule_match["matched_rules"].items():
                        combined_matched_rules[category].extend(matches)
                
                # 汇总其他字段
                if "dimension_matches" in rule_match:
                    combined_dimension_matches.update(rule_match["dimension_matches"])
                
                total_weight += rule_match.get("total_weight", 0.0)
                total_bias += rule_match.get("total_bias", 0.0)
                total_confidence += rule_match.get("confidence_adjustment", 0.0)
                
                comp_score = rule_match.get("comprehensive_score", "3.0")
                if comp_score not in comprehensive_scores:
                    comprehensive_scores.append(comp_score)
            
            rule_matches = {
                "matched_rules": combined_matched_rules,
                "dimension_matches": combined_dimension_matches,
                "total_weight": total_weight,
                "total_bias": total_bias,
                "confidence_adjustment": total_confidence,
                "comprehensive_score": "; ".join(comprehensive_scores) if comprehensive_scores else "3.0"
            }
            
            print(f"🔍 汇总结果: 权重={total_weight:.3f}, 偏置={total_bias:.3f}, 匹配规则数={sum(len(rules) for rules in combined_matched_rules.values())}")
            
        elif isinstance(implicit_opinion_analysis, dict):
            # 单个样本：字典格式
            print(f"🔍 处理单个字典样本")
            rule_matches = self.extract_implicit_opinions(implicit_opinion_analysis)
        else:
            # 不支持的格式，创建空规则匹配
            print(f"⚠️ 不支持的数据类型: {type(implicit_opinion_analysis)}")
            rule_matches = {
                "matched_rules": {
                    "strong_negative": [], "moderate_negative": [],
                    "strong_positive": [], "moderate_positive": []
                },
                "dimension_matches": {},
                "total_weight": 0.0,
                "total_bias": 0.0,
                "confidence_adjustment": 0.0,
                "comprehensive_score": "3.0"
            }
        
        # 计算调整参数
        weight_adj, bias_adj = self.compute_rule_adjustment(rule_matches)
        
        # 调整文本特征权重
        feature_multiplier = 1.0 + torch.tanh(weight_adj)  # 1.0附近的调整因子
        adjusted_features = text_features * feature_multiplier.expand_as(text_features)
        
        # 调整预测概率
        prediction_logits = torch.log(model_prediction + 1e-8)  # 转换为logits
        adjusted_logits = prediction_logits + bias_adj
        adjusted_prediction = torch.softmax(adjusted_logits, dim=-1)
        
        # 构建规则应用信息
        rule_info = {
            "matched_rules_count": sum(len(rules) for rules in rule_matches["matched_rules"].values()),
            "weight_adjustment": weight_adj.item(),
            "bias_adjustment": bias_adj.item(),
            "confidence_boost": rule_matches["confidence_adjustment"],
            "key_signals": []
        }
        
        # 提取关键信号用于解释
        for rule_type, matches in rule_matches["matched_rules"].items():
            if matches:
                rule_info["key_signals"].extend([
                    f"{rule_type}: {match['keyword']}" for match in matches[:2]  # 每类最多2个
                ])
        
        return adjusted_features, adjusted_prediction, rule_info

class ImplicitOpinionAnalyzer:
    """
    隐式意见分析器 - 模拟enhanced_results.json的分析过程
    """
    
    def __init__(self, llm_model_name="THUDM/chatglm-6b"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def analyze_implicit_opinion(self, text: str, title: str = "", comments: str = "") -> Dict:
        """
        分析文本的隐式意见，生成类似enhanced_results.json的结果
        """
        
        prompt = f"""请分析以下短视频内容的虚假新闻特征，按照以下维度给出评分和关键词：

标题：{title}
内容：{text}
评论：{comments}

请从以下维度分析（每项给出1-5分，5分为最佳）：

1. 专业性评估：
   关键词：
   分析：
   分数：

2. 客观事实符合度：
   关键词：
   分析：
   分数：

3. 物理常识符合度：
   关键词：
   分析：
   分数：

4. 情感导向性：
   关键词：
   分析：
   分数：

5. 误导性评估：
   关键词：
   分析：
   分数：

6. 信源可靠性：
   关键词：
   分析：
   分数：

综合评分：
虚假新闻可能性：（高/中/低）
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析响应为结构化数据
        parsed_result = self._parse_analysis_response(response)
        
        return parsed_result
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """
        解析LLM的分析响应为结构化数据
        """
        # 简化的解析逻辑，实际应该更加复杂
        result = {
            "虚假新闻特征检测": {},
            "综合评分": "3.0",
            "虚假新闻可能性": "中"
        }
        
        dimensions = [
            "专业性评估", "客观事实符合度", "物理常识符合度", 
            "情感导向性", "误导性评估", "信源可靠性"
        ]
        
        for dim in dimensions:
            # 这里应该有更复杂的正则表达式解析
            # 简化处理
            result["虚假新闻特征检测"][dim] = {
                "关键词": "待解析",
                "分析": "待解析",
                "分数": "3.0"
            }
        
        return result 