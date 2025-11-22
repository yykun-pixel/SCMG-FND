"""
神经符号规则配置文件

⚠️ 注意：本文件包含示例配置，实际敏感参数（权重、阈值）已移除。
请使用 config_neural_symbolic_safe.py 并通过环境变量或配置文件提供实际参数。

安全版本：请使用 config_neural_symbolic_safe.py
"""

# 神经符号规则系统配置
# ⚠️ 注意：敏感参数已移除，实际值需通过环境变量或配置文件提供
NEURAL_SYMBOLIC_CONFIG = {
    # 规则引擎基础配置
    "enable_neural_symbolic": True,
    "neural_symbolic_weight": None,  # 需通过环境变量 NEURAL_SYMBOLIC_WEIGHT 提供
    "rule_threshold": None,          # 需通过环境变量 RULE_THRESHOLD 提供
    "debug_mode": True,
    
    # 参数调优指导（仅供参考，实际值需通过配置提供）:
    # 
    # neural_symbolic_weight (神经符号权重):
    # - 0.1-0.3: 保守，主要依赖深度学习，规则仅作微调
    # - 0.4-0.6: 平衡，深度学习与规则并重 [推荐范围]
    # - 0.7-0.9: 激进，主要依赖规则，深度学习作补充
    # 
    # rule_threshold (规则激活阈值):
    # - 0.01-0.03: 敏感，微弱信号也会影响预测
    # - 0.05-0.1:  平衡，中等强度信号才影响预测 [推荐范围]
    # - 0.1-0.2:   保守，只有强烈信号才影响预测
    # 
    # ⚠️ 注意：实际参数值不在此文件中硬编码，需通过环境变量或配置文件提供
    
    # 实时分析配置
    "enable_realtime_analysis": False,
    "llm_model_path": "chatglm-6b",
    "max_analysis_length": 512,
    
    # 规则库配置
    "opinion_data_path": "enhanced_results.json",
    "rule_confidence_threshold": None,  # 需通过环境变量 RULE_CONFIDENCE_THRESHOLD 提供
    
    # 性能配置
    "batch_processing": True,
    "enable_caching": True,
}

# 规则权重配置 (影响调整强度)
# ⚠️ 注意：具体权重值已移除，需通过配置文件 rule_database.json 提供
RULE_WEIGHTS = {
    # 负面规则权重结构（实际值需通过配置文件提供）
    "strong_negative": {
        # 关键词列表保留，但权重值需通过配置文件提供
        "keywords": [],  # 关键词列表
        "weight": None,  # 权重值需配置
        "bias": None,    # 偏置值需配置
    },
    "moderate_negative": {
        "keywords": [],
        "weight": None,
        "bias": None,
    },
    
    # 正面规则权重结构（实际值需通过配置文件提供）
    "strong_positive": {
        "keywords": [],
        "weight": None,
        "bias": None,
    },
    "moderate_positive": {
        "keywords": [],
        "weight": None,
        "bias": None,
    },
}

# 专门的规则库配置
RULE_LIBRARY_CONFIG = {
    # 负面信号关键词扩展
    "negative_keywords": {
        "科学性": [
            "明显伪科学", "违背科学原理", "无科学依据", "伪科学表述",
            "科学常识错误", "理论依据不足", "学术造假嫌疑"
        ],
        "事实性": [
            "明显虚假", "缺乏客观事实", "无法验证", "信息失实",
            "事实歪曲", "数据造假", "虚构事件"
        ],
        "逻辑性": [
            "因果倒置", "断章取义", "逻辑混乱", "推理错误",
            "归因错误", "以偏概全", "逻辑跳跃"
        ],
        "常识性": [
            "违背常识", "违背物理定律", "不符合自然规律", "常识性错误",
            "基本逻辑错误", "违背生活常识"
        ],
        "表达性": [
            "说话过于绝对", "夸大其词", "言辞激烈", "极端表述",
            "煽动性语言", "情绪化表达", "绝对化用词"
        ],
        "意图性": [
            "制造恐慌", "情绪操控", "恶意传播", "煽动情绪",
            "制造焦虑", "引发争议", "流量操控"
        ]
    },
    
    # 正面信号关键词扩展
    "positive_keywords": {
        "可靠性": [
            "官方有报道", "权威机构证实", "联网搜索确实有这件事",
            "可靠消息来源", "官方确认", "权威媒体报道"
        ],
        "科学性": [
            "符合事实", "科学依据充分", "理论支撑完整", "数据可靠",
            "学术支持", "专家认可", "研究证实"
        ],
        "客观性": [
            "客观中性", "事实陈述", "中性表达", "客观描述",
            "不带情感色彩", "理性分析", "冷静客观"
        ],
        "逻辑性": [
            "逻辑合理", "推理正确", "论证充分", "逻辑清晰",
            "因果关系明确", "逻辑链完整"
        ],
        "专业性": [
            "术语专业", "专业表述", "学术规范", "专业准确",
            "技术术语正确", "专业知识准确"
        ]
    }
}

# 规则权重映射
# ⚠️ 注意：具体权重值已移除，需通过配置文件提供
RULE_WEIGHT_MAPPING = {
    # 负面规则权重结构（实际值需通过配置文件提供）
    "科学性": {"weight": None, "bias": None},
    "事实性": {"weight": None, "bias": None},
    "逻辑性": {"weight": None, "bias": None},
    "常识性": {"weight": None, "bias": None},
    "表达性": {"weight": None, "bias": None},
    "意图性": {"weight": None, "bias": None},
    
    # 正面规则权重结构（实际值需通过配置文件提供）
    "可靠性": {"weight": None, "bias": None},
    "科学性_正": {"weight": None, "bias": None},
    "客观性": {"weight": None, "bias": None},
    "逻辑性_正": {"weight": None, "bias": None},
    "专业性": {"weight": None, "bias": None}
}

# 维度特定规则配置
# ⚠️ 注意：权重乘数已移除，需通过配置文件提供
DIMENSION_SPECIFIC_RULES = {
    "专业性评估": {
        "high_impact_keywords": [],  # 关键词需通过配置提供
        "weight_multiplier": None   # 权重乘数需配置
    },
    "客观事实符合度": {
        "high_impact_keywords": [],
        "weight_multiplier": None
    },
    "物理常识符合度": {
        "high_impact_keywords": [],
        "weight_multiplier": None
    },
    "情感导向性": {
        "high_impact_keywords": [],
        "weight_multiplier": None
    },
    "误导性评估": {
        "high_impact_keywords": [],
        "weight_multiplier": None
    },
    "信源可靠性": {
        "high_impact_keywords": [],
        "weight_multiplier": None
    }
}

# 训练配置
# ⚠️ 注意：敏感参数已移除，需通过环境变量或配置文件提供
TRAINING_CONFIG = {
    "neural_symbolic_start_epoch": 2,
    "rule_loss_weight": None,                 # 需通过环境变量 RULE_LOSS_WEIGHT 提供
    "adaptive_weight_lr": None,               # 需通过环境变量 ADAPTIVE_WEIGHT_LR 提供
    "rule_regularization": None,              # 需通过环境变量 RULE_REGULARIZATION 提供
    
    # 规则验证
    "validate_rules": True,
    "rule_validation_interval": 5,
    "rule_performance_threshold": None,      # 需通过环境变量 RULE_PERFORMANCE_THRESHOLD 提供
}

# 实验配置
EXPERIMENT_CONFIG = {
    # 对比实验
    "baseline_without_rules": False,          # 是否运行无规则的基线实验
    "ablation_study": {
        "test_individual_rule_types": False,  # 测试单独的规则类型
        "test_rule_combinations": False,      # 测试规则组合
        "test_weight_sensitivity": False,     # 测试权重敏感性
    },
    
    # 数据分析
    "analyze_rule_coverage": True,           # 分析规则覆盖率
    "analyze_rule_conflicts": True,          # 分析规则冲突
    "generate_rule_statistics": True,        # 生成规则统计报告
}

def get_config():
    """
    获取完整的配置
    """
    config = NEURAL_SYMBOLIC_CONFIG.copy()
    config["rule_library"] = RULE_LIBRARY_CONFIG
    config["rule_weights"] = RULE_WEIGHTS
    config["dimension_rules"] = DIMENSION_SPECIFIC_RULES
    config["training"] = TRAINING_CONFIG
    config["experiment"] = EXPERIMENT_CONFIG
    
    return config

def validate_config(config):
    """
    验证配置的有效性
    """
    required_keys = [
        "enable_neural_symbolic", 
        "neural_symbolic_weight",
        "rule_threshold"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # 验证权重范围
    if not (0.0 <= config["neural_symbolic_weight"] <= 1.0):
        raise ValueError("neural_symbolic_weight must be between 0 and 1")
    
    # 验证规则阈值
    if not (0.0 <= config["rule_threshold"] <= 1.0):
        raise ValueError("rule_threshold must be between 0 and 1")
    
    print("配置验证通过")
    return True

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    validate_config(config)
    
    print("神经符号规则配置:")
    print(f"  启用状态: {config['enable_neural_symbolic']}")
    print(f"  整体权重: {config['neural_symbolic_weight']}")
    print(f"  规则阈值: {config['rule_threshold']}")
    print(f"  负面规则关键词数量: {sum(len(v) for v in config['rule_library']['negative_keywords'].values())}")
    print(f"  正面规则关键词数量: {sum(len(v) for v in config['rule_library']['positive_keywords'].values())}") 