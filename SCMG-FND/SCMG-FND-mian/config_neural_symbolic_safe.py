"""
神经符号规则配置文件（安全版本）

注意：本配置文件不包含具体的权重和阈值数值。
实际参数需要通过环境变量或外部配置文件提供。
"""

import os
from typing import Dict, Any

def get_config_from_env() -> Dict[str, Any]:
    """
    从环境变量加载配置（不硬编码敏感参数）
    """
    return {
        "enable_neural_symbolic": os.getenv('ENABLE_NEURAL_SYMBOLIC', 'True').lower() == 'true',
        "neural_symbolic_weight": float(os.getenv('NEURAL_SYMBOLIC_WEIGHT', '0.0')),  # 需通过环境变量设置
        "rule_threshold": float(os.getenv('RULE_THRESHOLD', '0.0')),  # 需通过环境变量设置
        "debug_mode": os.getenv('DEBUG_MODE', 'False').lower() == 'true',
        
        # 实时分析配置
        "enable_realtime_analysis": os.getenv('ENABLE_REALTIME_ANALYSIS', 'False').lower() == 'true',
        "llm_model_path": os.getenv('LLM_MODEL_PATH', ''),
        "max_analysis_length": int(os.getenv('MAX_ANALYSIS_LENGTH', '512')),
        
        # 规则库配置
        "opinion_data_path": os.getenv('OPINION_DATA_PATH', 'enhanced_results.json'),
        "rule_confidence_threshold": float(os.getenv('RULE_CONFIDENCE_THRESHOLD', '0.0')),  # 需通过环境变量设置
        
        # 性能配置
        "batch_processing": os.getenv('BATCH_PROCESSING', 'True').lower() == 'true',
        "enable_caching": os.getenv('ENABLE_CACHING', 'True').lower() == 'true',
    }

def get_rule_structure_template() -> Dict[str, Any]:
    """
    返回规则结构模板（不含具体权重值）
    
    注意：实际权重值需要通过配置文件或环境变量提供
    """
    return {
        "rule_categories": {
            "strong_negative": {
                "description": "强负面信号规则",
                "keywords": [],  # 需通过配置文件提供
                "weight": None,  # 需通过配置文件提供
                "bias": None,
                "confidence_boost": None
            },
            "moderate_negative": {
                "description": "中等负面信号规则",
                "keywords": [],
                "weight": None,
                "bias": None,
                "confidence_boost": None
            },
            "strong_positive": {
                "description": "强正面信号规则",
                "keywords": [],
                "weight": None,
                "bias": None,
                "confidence_boost": None
            },
            "moderate_positive": {
                "description": "中等正面信号规则",
                "keywords": [],
                "weight": None,
                "bias": None,
                "confidence_boost": None
            }
        },
        "dimension_rules": {
            "专业性评估": {},
            "客观事实符合度": {},
            "物理常识符合度": {},
            "情感导向性": {},
            "误导性评估": {},
            "信源可靠性": {}
        }
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    注意：不验证具体数值，只验证结构
    """
    required_keys = [
        "enable_neural_symbolic",
        "neural_symbolic_weight",
        "rule_threshold"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # 不验证数值范围（因为可能为0或None，表示未配置）
    print("配置结构验证通过（注意：权重和阈值需通过外部配置提供）")
    return True

def get_config():
    """
    获取完整的配置（从环境变量）
    """
    config = get_config_from_env()
    config["rule_structure"] = get_rule_structure_template()
    
    return config

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print("神经符号规则配置（安全版本）:")
    print(f"  启用状态: {config['enable_neural_symbolic']}")
    print(f"  整体权重: {config['neural_symbolic_weight']} (需通过环境变量NEURAL_SYMBOLIC_WEIGHT设置)")
    print(f"  规则阈值: {config['rule_threshold']} (需通过环境变量RULE_THRESHOLD设置)")
    print("\n⚠️  注意：具体权重和阈值值需要通过环境变量或配置文件提供，不在代码中硬编码。")

