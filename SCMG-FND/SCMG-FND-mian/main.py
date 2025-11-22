"""
多模态虚假内容检测系统 - 主训练脚本

⚠️ 安全说明：
- 敏感参数（权重、阈值、dropout等）不硬编码在代码中
- 所有敏感参数需通过环境变量或命令行参数提供
- 详细配置说明请参考 CONFIGURATION_GUIDE.md
"""

import os
# 设置CUDA内存分配器参数，减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import numpy as np
import sys
import random
import torch.backends.cudnn as cudnn
import torchmetrics
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import time
import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Diffusion.Multimodal_Diffusion import GaussianDiffusionTrainer
from Diffusion.ExplainableDetection import ExplainableDetection
# from Dataset.dataset import FakeAVCeleb  # 注释掉找不到的模块导入
from modules.MultiGranularityContrast import MultiGranularityContrast
from modules.AdversarialVerification import AdversarialVerification
from modules.NeuralSymbolicRules import NeuralSymbolicRuleEngine
from train import train, valid, calculate_f1, calculate_auc
from dataloader_fakesv import get_dataloader
from eval_metrics import eval_FakeSV

from tensorboardX import SummaryWriter
writer = SummaryWriter("logs")

# 创建保存训练结果的目录
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 将训练结果保存为JSON
def save_training_results(epoch_results, directory="results", filename="training_results.json"):
    """
    将训练结果保存到单个JSON文件中
    
    Args:
        epoch_results: 当前epoch的结果字典
        directory: 保存结果的目录
        filename: 保存结果的文件名
    """
    ensure_dir(directory)
    filepath = f"{directory}/{filename}"
    
    # 将tensor转换为Python原生类型
    for key, value in epoch_results.items():
        if isinstance(value, torch.Tensor):
            epoch_results[key] = value.item() if value.numel() == 1 else value.tolist()
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    epoch_results[key][k] = v.item() if v.numel() == 1 else v.tolist()
    
    # 检查文件是否存在
    all_results = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {filepath}文件格式错误，将创建新文件")
    
    # 添加当前epoch的结果
    all_results.append(epoch_results)
    
    # 保存所有结果
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Epoch {epoch_results['epoch']} 结果已添加到 {filepath}")

# 保存可解释性结果
def save_explanation_results(explanations, directory="explanations", filename="explanation_results.pkl"):
    """
    保存可解释性结果
    
    Args:
        explanations: 可解释性结果列表
        directory: 保存目录
        filename: 文件名
    """
    ensure_dir(directory)
    filepath = os.path.join(directory, filename)
    
    # 使用pickle保存结果（包含NumPy数组）
    with open(filepath, 'wb') as f:
        pickle.dump(explanations, f)
    
    print(f"已保存{len(explanations)}个可解释性结果到{filepath}")
    
    # 可视化一部分结果
    visualize_explanation_samples(explanations, os.path.join(directory, "visualization"))

# 可视化部分可解释性结果
def visualize_explanation_samples(explanations, save_dir, num_samples=5):
    """
    可视化部分可解释性结果
    
    Args:
        explanations: 可解释性结果列表
        save_dir: 保存可视化图像的目录
        num_samples: 要可视化的样本数量
    """
    ensure_dir(save_dir)
    
    # 随机选择一部分样本进行可视化
    if len(explanations) > num_samples:
        indices = np.random.choice(len(explanations), num_samples, replace=False)
        samples = [explanations[i] for i in indices]
    else:
        samples = explanations
    
    for i, sample in enumerate(samples):
        # 创建每个样本的单独目录
        sample_dir = os.path.join(save_dir, f"sample_{i}")
        ensure_dir(sample_dir)
        
        # 获取真实标签和预测标签
        true_label = sample.get('label', -1)
        pred_label = int(sample.get('predicted_class', -1)) if 'predicted_class' in sample else -1
        
        # 可视化模态权重
        if 'modality_weights' in sample:
            plt.figure(figsize=(8, 5))
            modality_weights = sample['modality_weights']
            modal_names = ["Text", "Audio", "Video"]
            plt.bar(modal_names, modality_weights)
            plt.title(f"Modality Contribution Weights (True: {('Real' if true_label == 0 else 'Fake')}, Pred: {('Real' if pred_label == 0 else 'Fake')})")
            plt.ylim(0, 1)
            plt.savefig(os.path.join(sample_dir, "modality_weights.png"))
            plt.close()
        
        # 可视化文本重要性
        if 'text_importance' in sample:
            plt.figure(figsize=(10, 2))
            text_importance = sample['text_importance']
            plt.bar(range(len(text_importance)), text_importance)
            plt.title("Text Feature Importance")
            plt.savefig(os.path.join(sample_dir, "text_importance.png"))
            plt.close()
        
        # 可视化音频重要性
        if 'audio_importance' in sample:
            plt.figure(figsize=(10, 2))
            audio_importance = sample['audio_importance']
            plt.bar(range(len(audio_importance)), audio_importance)
            plt.title("Audio Feature Importance")
            plt.savefig(os.path.join(sample_dir, "audio_importance.png"))
            plt.close()
        
        # 可视化视频重要性和热图
        if 'video_importance' in sample:
            plt.figure(figsize=(10, 2))
            video_importance = sample['video_importance']
            plt.bar(range(len(video_importance)), video_importance)
            plt.title("Video Feature Importance")
            plt.savefig(os.path.join(sample_dir, "video_importance.png"))
            plt.close()
        
        # 可视化虚假区域热图
        if 'fake_region_heatmap' in sample:
            plt.figure(figsize=(8, 3))
            heatmap = sample['fake_region_heatmap']
            plt.imshow(heatmap.reshape(1, -1), cmap='hot', aspect='auto')
            plt.colorbar(label='Fake Level')
            plt.title("Fake Region Heatmap")
            plt.savefig(os.path.join(sample_dir, "fake_region_heatmap.png"))
            
            # 保存热图数据
            np.save(os.path.join(sample_dir, "heatmap.npy"), heatmap)
        
        # 保存基本信息
        info = {
            'sample_idx': sample.get('sample_idx', i),
            'batch_idx': sample.get('batch_idx', -1),
            'true_label': true_label,
            'predicted_label': pred_label,
            'correct_prediction': true_label == pred_label
        }
        
        with open(os.path.join(sample_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)
    
    print(f"已生成{len(samples)}个样本的可视化结果，保存在{save_dir}")

def safe_model_save(trainer, optimizer, valid_acc, epoch, modelConfig, filename):
    """安全地保存模型，包含错误处理和重试机制"""
    save_dir = "model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    temp_filepath = f"{filepath}.tmp"
    
    save_dict = {
        'model_state_dict': trainer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_acc': valid_acc,
        'epoch': epoch,
        'modelConfig': modelConfig
    }
    
    # 尝试保存到临时文件，成功后再重命名
    try:
        torch.save(save_dict, temp_filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        print(f"模型成功保存到 {filepath}")
        return True
    except Exception as e:
        print(f"保存模型失败: {e}")
        # 尝试仅保存模型参数
        try:
            print("尝试仅保存模型参数...")
            torch.save(trainer.state_dict(), f"{filepath}_params_only.pt")
            print(f"模型参数已保存到 {filepath}_params_only.pt")
            return True
        except Exception as e2:
            print(f"保存模型参数也失败: {e2}")
            return False

def load_sensitive_params_from_env():
    """
    从环境变量加载敏感参数
    
    注意：敏感参数（权重、阈值等）不硬编码，需通过环境变量提供
    """
    return {
        # 模块权重参数（需通过环境变量设置）
        "contrast_weight": float(os.getenv('CONTRAST_WEIGHT', '0.0')),  # 需设置
        "adv_weight": float(os.getenv('ADV_WEIGHT', '0.0')),  # 需设置
        "neural_symbolic_weight": float(os.getenv('NEURAL_SYMBOLIC_WEIGHT', '0.0')),  # 需设置
        "explain_weight": float(os.getenv('EXPLAIN_WEIGHT', '0.0')),  # 需设置
        
        # 神经符号规则参数
        "rule_threshold": float(os.getenv('RULE_THRESHOLD', '0.0')),  # 需设置
        
        # 扩散模型参数
        "beta_1": float(os.getenv('BETA_1', '0.0')),  # 需设置
        "beta_T": float(os.getenv('BETA_T', '0.0')),  # 需设置
        "diffusion_loss_weight": float(os.getenv('DIFFUSION_LOSS_WEIGHT', '0.0')),  # 需设置
        
        # 正则化参数
        "domain_lambda": float(os.getenv('DOMAIN_LAMBDA', '0.0')),  # 需设置
        "adv_eps": float(os.getenv('ADV_EPS', '0.0')),  # 需设置
        "weight_decay": float(os.getenv('WEIGHT_DECAY', '0.0')),  # 需设置
        "label_smoothing": float(os.getenv('LABEL_SMOOTHING', '0.0')),  # 需设置
        
        # Dropout参数
        "mult_dropout": float(os.getenv('MULT_DROPOUT', '0.0')),  # 需设置
        "Text_Pre_dropout": float(os.getenv('TEXT_PRE_DROPOUT', '0.0')),  # 需设置
        "Img_Pre_dropout": float(os.getenv('IMG_PRE_DROPOUT', '0.0')),  # 需设置
        "comments_dropout": float(os.getenv('COMMENTS_DROPOUT', '0.0')),  # 需设置
        
        # 对比学习参数
        "contrast_temperature": float(os.getenv('CONTRAST_TEMPERATURE', '0.0')),  # 需设置
        "contrast_projection_dim": int(os.getenv('CONTRAST_PROJECTION_DIM', '0')),  # 需设置
        
        # 对抗验证参数
        "adv_dropout": float(os.getenv('ADV_DROPOUT', '0.0')),  # 需设置
        "adv_hidden_dim": int(os.getenv('ADV_HIDDEN_DIM', '0')),  # 需设置
    }

def parse_arguments():
    """
    解析命令行参数
    
    注意：敏感参数（权重、阈值）不在此处硬编码，需通过环境变量提供
    """
    parser = argparse.ArgumentParser(description='多模态虚假视频检测系统 - 模块化运行')
    
    # 基础训练参数
    parser.add_argument('--epoch', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=7e-5, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (auto/cuda:0/cpu)')
    
    # 功能模块开关
    parser.add_argument('--use_explain', type=str, default='True', 
                       help='是否启用可解释性模块 (True/False)')
    parser.add_argument('--use_multi_granularity_contrast', type=str, default='True',
                       help='是否启用多粒度对比学习 (True/False)')
    parser.add_argument('--use_adversarial_verification', type=str, default='True',
                       help='是否启用对抗性验证框架 (True/False)')
    parser.add_argument('--use_neural_symbolic', type=str, default='True',
                       help='是否启用神经符号规则系统 (True/False)')
    
    # 模块权重参数（从环境变量读取，不硬编码默认值）
    parser.add_argument('--contrast_weight', type=float, default=None, 
                       help='对比学习损失权重（需通过环境变量CONTRAST_WEIGHT设置）')
    parser.add_argument('--adv_weight', type=float, default=None, 
                       help='对抗验证损失权重（需通过环境变量ADV_WEIGHT设置）')
    parser.add_argument('--neural_symbolic_weight', type=float, default=None, 
                       help='神经符号规则权重（需通过环境变量NEURAL_SYMBOLIC_WEIGHT设置）')
    parser.add_argument('--explain_weight', type=float, default=None, 
                       help='可解释性损失权重（需通过环境变量EXPLAIN_WEIGHT设置）')
    
    # 神经符号规则特定参数
    parser.add_argument('--rule_threshold', type=float, default=None, 
                       help='规则激活阈值（需通过环境变量RULE_THRESHOLD设置）')
    parser.add_argument('--enable_implicit_analysis', type=str, default='False',
                       help='是否启用实时隐式意见分析 (True/False)')
    parser.add_argument('--opinion_data_path', type=str, default='enhanced_results.json',
                       help='隐式意见分析数据文件路径')
    
    # 调试和日志参数
    parser.add_argument('--debug_neural_symbolic', type=str, default='True',
                       help='是否启用神经符号规则调试输出 (True/False)')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10, help='日志输出间隔')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='SVFEND', help='数据集名称')
    parser.add_argument('--datamode', type=str, default='title+ocr', help='数据模式')
    
    return parser.parse_args()

def str_to_bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_model_config(args):
    """
    根据命令行参数创建模型配置
    
    注意：敏感参数（权重、阈值、dropout等）从环境变量读取，不硬编码
    """
    # 从环境变量加载敏感参数
    sensitive_params = load_sensitive_params_from_env()
    
    # 设备配置
    if args.device == 'auto':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 权重参数：优先使用命令行参数，其次环境变量，最后报错
    contrast_weight = args.contrast_weight if args.contrast_weight is not None else sensitive_params["contrast_weight"]
    adv_weight = args.adv_weight if args.adv_weight is not None else sensitive_params["adv_weight"]
    neural_symbolic_weight = args.neural_symbolic_weight if args.neural_symbolic_weight is not None else sensitive_params["neural_symbolic_weight"]
    explain_weight = args.explain_weight if args.explain_weight is not None else sensitive_params["explain_weight"]
    rule_threshold = args.rule_threshold if args.rule_threshold is not None else sensitive_params["rule_threshold"]
    
    # 验证敏感参数是否已设置
    if contrast_weight == 0.0 and str_to_bool(args.use_multi_granularity_contrast):
        print("⚠️ 警告: CONTRAST_WEIGHT未设置，对比学习模块可能无法正常工作")
    if adv_weight == 0.0 and str_to_bool(args.use_adversarial_verification):
        print("⚠️ 警告: ADV_WEIGHT未设置，对抗验证模块可能无法正常工作")
    if neural_symbolic_weight == 0.0 and str_to_bool(args.use_neural_symbolic):
        print("⚠️ 警告: NEURAL_SYMBOLIC_WEIGHT未设置，神经符号规则模块可能无法正常工作")
    if rule_threshold == 0.0 and str_to_bool(args.use_neural_symbolic):
        print("⚠️ 警告: RULE_THRESHOLD未设置，神经符号规则模块可能无法正常工作")
    
    modelConfig = {
        "state": "train",
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "T": 100,  # 扩散步数（非敏感参数）
        # Dropout参数（从环境变量读取）
        "mult_dropout": sensitive_params["mult_dropout"] if sensitive_params["mult_dropout"] > 0 else 0.4,
        "Text_Pre_dropout": sensitive_params["Text_Pre_dropout"] if sensitive_params["Text_Pre_dropout"] > 0 else 0.3,
        "Img_Pre_dropout": sensitive_params["Img_Pre_dropout"] if sensitive_params["Img_Pre_dropout"] > 0 else 0.3,
        "comments_dropout": sensitive_params["comments_dropout"] if sensitive_params["comments_dropout"] > 0 else 0.3,
        "lr": args.lr,
        # 扩散模型参数（从环境变量读取）
        "beta_1": sensitive_params["beta_1"] if sensitive_params["beta_1"] > 0 else 1e-4,
        "beta_T": sensitive_params["beta_T"] if sensitive_params["beta_T"] > 0 else 0.02,
        "device": device,
        # 特征维度（非敏感参数）
        "t_in": 768,
        "i_in": 2048,
        "a_in": 128,
        "v_in": 4096,
        "c3d_in": 4096,
        "t_in_pre": 100,
        "a_in_pre": 128,
        "v_in_pre": 1000,
        "c3d_in_pre": 128,
        "label_dim": 2,
        "d_m": 128,
        "unified_size": 128,
        "vertex_num": 32,
        "routing": 2,
        "T_t": 2,
        "T_a": 2,
        "T_v": 2,
        # 正则化参数（从环境变量读取）
        "weight_decay": sensitive_params["weight_decay"] if sensitive_params["weight_decay"] > 0 else 0.05,
        "num_workers": 4,
        "save_freq": 5,
        "early_stop": 15,
        "use_lr_scheduler": False,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        
        # 功能模块配置 - 从命令行参数读取
        "use_explain": str_to_bool(args.use_explain),
        "use_multi_granularity_contrast": str_to_bool(args.use_multi_granularity_contrast),
        "use_adversarial_verification": str_to_bool(args.use_adversarial_verification),
        "use_neural_symbolic": str_to_bool(args.use_neural_symbolic),
        
        # 权重配置（从环境变量或命令行参数读取）
        "contrast_weight": contrast_weight,
        "adv_weight": adv_weight,
        "neural_symbolic_weight": neural_symbolic_weight,
        "explain_weight": explain_weight,
        "domain_lambda": sensitive_params["domain_lambda"] if sensitive_params["domain_lambda"] > 0 else 0.05,
        "adv_eps": sensitive_params["adv_eps"] if sensitive_params["adv_eps"] > 0 else 0.05,
        
        # 神经符号规则配置
        "enable_neural_symbolic": str_to_bool(args.use_neural_symbolic),
        "rule_threshold": rule_threshold,
        "enable_implicit_analysis": str_to_bool(args.enable_implicit_analysis),
        "opinion_data_path": args.opinion_data_path,
        "debug_neural_symbolic": str_to_bool(args.debug_neural_symbolic),
        "log_rule_applications": True,
        "save_rule_statistics": True,
        
        # 数据加载与处理参数
        "datamode": args.datamode,
        "dataset": args.dataset,
        "drop_last_batch": False,  
        "skip_error_batches": True,
        "pin_memory": True,
        
        # 学习率调度参数
        "use_lr_scheduler": True,
        "lr_patience": 5,
        "lr_factor": 0.7,
        "lr_threshold": 1e-4,
        "lr_min": 1e-7,
        
        # 梯度裁剪与优化器参数
        "clip_grad_norm": 1.0,
        "weight_init": "xavier_normal",
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        
        # 正则化与训练稳定性参数（从环境变量读取）
        "label_smoothing": sensitive_params["label_smoothing"] if sensitive_params["label_smoothing"] > 0 else 0.1,
        "use_amp": True,
        "warmup_steps": 1000,
        "use_warmup": True,
        
        # 扩散模型特定参数（从环境变量读取）
        "diffusion_loss_weight": sensitive_params["diffusion_loss_weight"] if sensitive_params["diffusion_loss_weight"] > 0 else 0.008,
        
        # 对比学习参数（从环境变量读取）
        "contrast_temperature": sensitive_params["contrast_temperature"] if sensitive_params["contrast_temperature"] > 0 else 0.1,
        "contrast_projection_dim": sensitive_params["contrast_projection_dim"] if sensitive_params["contrast_projection_dim"] > 0 else 64,
        "contrast_spatial_regions": 4,
        "contrast_temporal_segments": 8,
        
        # 对抗验证参数（从环境变量读取）
        "adv_dropout": sensitive_params["adv_dropout"] if sensitive_params["adv_dropout"] > 0 else 0.3,
        "adv_hidden_dim": sensitive_params["adv_hidden_dim"] if sensitive_params["adv_hidden_dim"] > 0 else 256,
        "adv_z_dim": 64,
        "adv_num_layers": 2,
        
        # 保存目录配置
        "save_dir": args.save_dir,
        "log_interval": args.log_interval,
    }
    
    return modelConfig

def print_module_status(config):
    """
    打印当前启用的模块状态
    """
    print("="*60)
    print("🔧 模块配置状态")
    print("="*60)
    print(f"🧠 可解释性模块: {'✅ 启用' if config['use_explain'] else '❌ 关闭'}")
    print(f"🔍 多粒度对比学习: {'✅ 启用' if config['use_multi_granularity_contrast'] else '❌ 关闭'}")
    print(f"🛡️ 对抗性验证框架: {'✅ 启用' if config['use_adversarial_verification'] else '❌ 关闭'}")
    print(f"⚖️ 神经符号规则: {'✅ 启用' if config['use_neural_symbolic'] else '❌ 关闭'}")
    print()
    
    if config['use_neural_symbolic']:
        print("🎯 神经符号规则详细配置:")
        print(f"   - 规则权重: {config['neural_symbolic_weight']}")
        print(f"   - 激活阈值: {config['rule_threshold']}")
        print(f"   - 实时分析: {'启用' if config['enable_implicit_analysis'] else '关闭'}")
        print(f"   - 数据路径: {config['opinion_data_path']}")
        print(f"   - 调试输出: {'启用' if config['debug_neural_symbolic'] else '关闭'}")
        print()
    
    print("⚙️ 权重配置:")
    if config['use_explain']:
        print(f"   - 可解释性权重: {config['explain_weight']}")
    if config['use_multi_granularity_contrast']:
        print(f"   - 对比学习权重: {config['contrast_weight']}")
    if config['use_adversarial_verification']:
        print(f"   - 对抗验证权重: {config['adv_weight']}")
    if config['use_neural_symbolic']:
        print(f"   - 神经符号权重: {config['neural_symbolic_weight']}")
    print("="*60)

def main(external_config=None):
    """
    主函数：解析参数并启动训练
    
    Args:
        external_config: 外部传入的配置（可选），如果提供则使用该配置而不是解析命令行参数
    """
    if external_config is not None:
        # 使用外部传入的配置
        modelConfig = external_config
        print("✅ 使用外部传入的配置")
    else:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建模型配置
        modelConfig = create_model_config(args)
        
        # 打印模块状态
        print_module_status(modelConfig)
    
    # 如果用户选择了模块，确保相关依赖可用
    if modelConfig['use_neural_symbolic']:
        try:
            from modules.NeuralSymbolicRules import NeuralSymbolicRuleEngine
            print("✅ 神经符号规则模块导入成功")
        except ImportError as e:
            print(f"❌ 无法导入神经符号规则模块: {e}")
            print("💡 请确保已正确安装相关依赖")
            return -1
    
    # 开始训练流程
    print("🚀 开始训练流程...")
    
    # 数据加载
    device = torch.device(modelConfig["device"])
    print(f"🖥️ 使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"   可用GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU内存: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")

    print("📊 开始加载数据...")
    dataloader = get_dataloader(modelConfig=modelConfig, data_type='SVFEND')
    print("✅ 数据加载完成")

    # 模型初始化
    print("🔧 初始化模型...")
    trainer = GaussianDiffusionTrainer(
        modelConfig, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        modelConfig["t_in"], modelConfig["a_in"], modelConfig["v_in"], modelConfig["d_m"], modelConfig["mult_dropout"],
        modelConfig["label_dim"],
        modelConfig["unified_size"], modelConfig["vertex_num"], modelConfig["routing"], modelConfig["T_t"],
        modelConfig["T_a"],  modelConfig["T_v"], modelConfig["batch_size"]).to(device)
    print("✅ 主模型初始化完成")

    # 应用权重初始化
    if modelConfig["weight_init"] == "xavier_normal":
        print("应用Xavier Normal权重初始化...")
        for p in trainer.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    elif modelConfig["weight_init"] == "kaiming_normal":
        print("应用Kaiming Normal权重初始化...")
        for p in trainer.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
    
    # 初始化多粒度对比学习模块（如果启用）
    contrast_module = None
    if modelConfig.get("use_multi_granularity_contrast", False):
        print("初始化多粒度对比学习模块...")
        # 参数从配置读取（已从环境变量加载）
        contrast_module = MultiGranularityContrast(
            feature_dim=modelConfig["unified_size"],  # 使用统一特征维度
            projection_dim=modelConfig.get("contrast_projection_dim", 64),
            temperature=modelConfig.get("contrast_temperature", 0.1),
            spatial_regions=modelConfig.get("contrast_spatial_regions", 4),
            temporal_segments=modelConfig.get("contrast_temporal_segments", 8),
            modal_components=3  # 文本、音频、视频三种模态
        ).to(device)
        print(f"多粒度对比学习配置: 特征维度={modelConfig['unified_size']}, "
              f"温度={modelConfig.get('contrast_temperature', 0.1)} "
              f"(从环境变量CONTRAST_TEMPERATURE读取)")
    
    # 初始化对抗性验证框架（如果启用）
    adv_framework = None
    if modelConfig.get("use_adversarial_verification", False):
        print("初始化对抗性验证框架...")
        from modules.AdversarialVerification import AdversarialVerification
        # 参数从配置读取（已从环境变量加载）
        adv_framework = AdversarialVerification(
            feature_dim=modelConfig["unified_size"],
            hidden_dim=modelConfig.get("adv_hidden_dim", 256),
            z_dim=modelConfig.get("adv_z_dim", 64),
            num_layers=modelConfig.get("adv_num_layers", 2),
            dropout=modelConfig.get("adv_dropout", 0.3)
        ).to(device)
        print(f"对抗性验证框架配置: 特征维度={modelConfig['unified_size']}, "
              f"隐藏维度={modelConfig.get('adv_hidden_dim', 256)} "
              f"(从环境变量ADV_HIDDEN_DIM读取), "
              f"Dropout={modelConfig.get('adv_dropout', 0.3)} "
              f"(从环境变量ADV_DROPOUT读取)")

    optimizer = torch.optim.AdamW(
        trainer.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=modelConfig["weight_decay"],
        betas=modelConfig.get("betas", (0.9, 0.999)),
        eps=modelConfig.get("eps", 1e-8)
    )
    
    # 添加学习率调度器
    if modelConfig.get("use_lr_scheduler", False):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 监控验证准确率
            factor=modelConfig.get("lr_factor", 0.7),
            patience=modelConfig.get("lr_patience", 5),
            verbose=True,
            threshold=modelConfig.get("lr_threshold", 1e-4),
            min_lr=modelConfig.get("lr_min", 1e-7)
        )
        print(f"已启用学习率调度器 - 参数: patience={modelConfig.get('lr_patience', 5)}, factor={modelConfig.get('lr_factor', 0.7)}")
    else:
        scheduler = None
    
    # 添加Warmup调度器
    if modelConfig.get("use_warmup", False):
        print(f"启用学习率预热（Warmup）- 预热步数: {modelConfig.get('warmup_steps', 1000)}")
        # 这里只是记录warmup状态，实际实现在训练循环中
        
    # 损失函数 - 添加标签平滑
    if modelConfig.get("label_smoothing", 0) > 0:
        print(f"使用标签平滑，平滑系数: {modelConfig['label_smoothing']}")
        criterion = nn.CrossEntropyLoss(label_smoothing=modelConfig["label_smoothing"]).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if modelConfig["dataset"] in ['WEIBO']:
        best_valid_acc = -1
        epoch, best_epoch = 0, 0
        global_step = 0  # 全局步数计数器，用于warmup
    else:
        # 对于其他数据集，也初始化相关变量
        best_valid_acc = -1  # 初始化为一个较小的值
        epoch, best_epoch = 0, 0
        global_step = 0

    # 创建结果保存目录与文件名
    results_dir = f"results_{modelConfig['dataset']}"
    results_filename = f"{modelConfig['dataset']}_training_results.json"
    ensure_dir(results_dir)
    
    # 创建可解释性结果保存目录
    explanation_dir = modelConfig.get("explanation_dir", "explanations")
    ensure_dir(explanation_dir)
    
    # 每个epoch保存最佳的可解释性结果
    best_explanation_results = None

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if modelConfig.get("use_amp", False) and torch.cuda.is_available() else None
    if scaler:
        print("已启用混合精度训练")
        
    # 输出是否启用可解释性
    if modelConfig.get("enable_explanation", False):
        print(f"已启用可解释性模块 - 可视化结果将保存在 {explanation_dir} 目录中")

    # 设置随机种子
    setup_seed(42)
    
    # 修改训练函数以支持多粒度对比学习和对抗性验证框架
    from train import train
    
    # 修改为支持新模块的训练函数调用
    best_val_acc = -1
    best_model_path = ''
    patience_counter = 0
    
    for epoch in range(modelConfig["epoch"]):
        # 训练一个epoch
        train_loss, train_acc, valid_loss, valid_acc, explanations = train(
            trainer, device, dataloader["train"], dataloader["val"], optimizer, epoch,
            modelConfig, criterion=criterion, contrast_module=contrast_module, adv_framework=adv_framework
        )
        
        # 评估测试集性能
        print("评估测试集性能...")
        trainer.eval()
        with torch.no_grad():
            test_loss, test_results, test_truths, _ = valid(dataloader["test"], trainer, criterion, modelConfig)
            test_acc = 0.0
            if len(test_results) > 0 and len(test_truths) > 0:
                test_acc = (test_results == test_truths).float().mean().item()
            
            # 计算其他测试指标
            test_predictions = test_results.cpu().numpy()
            test_labels = test_truths.cpu().numpy()
            try:
                test_f1 = calculate_f1(test_predictions, test_labels)
                test_auc = calculate_auc(test_predictions, test_labels)
            except:
                test_f1 = 0.0
                test_auc = 0.0
                
            print(f"测试集评估: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        # 保存结果到epoch_results列表
        epoch_results = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train': {
                'loss': train_loss,
                'accuracy': train_acc,
            },
            'validation': {
                'loss': valid_loss,
                'accuracy': valid_acc,
            },
            'test': {  # 添加测试集的结果
                'loss': test_loss,
                'accuracy': test_acc,
                'f1_score': test_f1,
                'auc': test_auc
            },
            'hyperparameters': {
                'learning_rate': optimizer.param_groups[0]['lr'],
                'batch_size': modelConfig['batch_size'],
                'dropout': modelConfig['mult_dropout'],
                'weight_decay': modelConfig['weight_decay'],
                'diffusion_loss_weight': modelConfig.get('diffusion_loss_weight', 0.008),
                'contrast_weight': modelConfig.get('contrast_weight', 0.1) if modelConfig.get('use_multi_granularity_contrast', False) else 0,
                'adv_weight': modelConfig.get('adv_weight', 0.1) if modelConfig.get('use_adversarial_verification', False) else 0
            },
            'best_so_far': valid_acc >= best_val_acc
        }
        
        # 保存训练结果到单个JSON文件
        save_training_results(epoch_results, results_dir, results_filename)
        
        # 保存可解释性结果
        if modelConfig.get("enable_explanation", False) and modelConfig.get("save_explanations", False):
            # 使用测试集的可解释性结果
            epoch_explanation_dir = os.path.join(explanation_dir, f"epoch_{epoch}")
            ensure_dir(epoch_explanation_dir)
            
            # 保存本轮的可解释性结果
            explanation_filename = f"{modelConfig['dataset']}_explanations_epoch_{epoch}.pkl"
            save_explanation_results(explanations, epoch_explanation_dir, explanation_filename)
            
            # 可视化部分结果
            num_vis_samples = min(len(explanations), modelConfig.get("visualization_samples", 10))
            visualize_explanation_samples(
                explanations, 
                os.path.join(epoch_explanation_dir, "visualization"),
                num_samples=num_vis_samples
            )
            
            # 如果是最佳epoch，保存为最佳可解释性结果
            if valid_acc >= best_valid_acc:
                best_explanation_results = explanations
        
        # 调用学习率调度器，根据验证准确率调整学习率
        if scheduler is not None and modelConfig.get("use_lr_scheduler", False):
            scheduler.step(valid_acc)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
        
        # 打印进度
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " +
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # 保存最好的模型
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            patience_counter = 0
            
            # 保存模型
            best_model_path = os.path.join(results_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
            }, best_model_path)
            print(f"保存最佳模型，验证准确率: {valid_acc:.4f}")
            
            # 可视化最佳模型的解释结果
            if modelConfig.get("use_explain", True) and explanations:
                visualize_explanation_samples(explanations, os.path.join(explanation_dir, f"epoch_{epoch}"))
                print(f"保存了解释结果到 {explanation_dir}/epoch_{epoch}")
        else:
            patience_counter += 1
            print(f"验证准确率未提高, 耐心计数: {patience_counter}/{modelConfig['early_stop']}")
        
        # 每save_freq个epoch保存一次检查点
        if (epoch + 1) % modelConfig["save_freq"] == 0:
            checkpoint_path = os.path.join(results_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
        
        # 早停
        if patience_counter >= modelConfig["early_stop"]:
            print(f"连续 {modelConfig['early_stop']} 个epoch验证准确率未提高，早停")
            break
        
        writer.close()
    
    # 训练完成后，使用最佳模型进行最终测试集评估
    print("训练完成，使用最佳模型进行最终测试集评估...")
    if os.path.exists(best_model_path):
        # 加载最佳模型
        checkpoint = torch.load(best_model_path)
        trainer.load_state_dict(checkpoint['model_state_dict'])
        
        # 在测试集上评估
        trainer.eval()
        with torch.no_grad():
            test_loss, test_results, test_truths, _ = valid(dataloader["test"], trainer, criterion, modelConfig)
            test_acc = 0.0
            if len(test_results) > 0 and len(test_truths) > 0:
                test_acc = (test_results == test_truths).float().mean().item()
            
            # 计算其他评估指标
            test_predictions = test_results.cpu().numpy()
            test_labels = test_truths.cpu().numpy()
            try:
                test_f1 = calculate_f1(test_predictions, test_labels)
                test_auc = calculate_auc(test_predictions, test_labels)
            except:
                test_f1 = 0.0
                test_auc = 0.0
            
            # 保存最终测试结果
            final_test_results = {
                'final_test': {
                    'loss': test_loss,
                    'accuracy': test_acc,
                    'f1_score': test_f1,
                    'auc': test_auc
                },
                'best_model_path': best_model_path,
                'training_complete': True
            }
            
            # 添加到已有结果文件
            results_filepath = f"{results_dir}/{results_filename}"
            if os.path.exists(results_filepath):
                try:
                    with open(results_filepath, 'r', encoding='utf-8') as f:
                        all_results = json.load(f)
                    
                    # 添加最终测试结果作为额外条目
                    all_results.append(final_test_results)
                    
                    # 保存更新后的结果
                    with open(results_filepath, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=4)
                    
                    print(f"最终测试结果已添加到 {results_filepath}")
                except Exception as e:
                    print(f"保存最终测试结果时出错: {e}")
            
            print(f"最终测试评估: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    else:
        print(f"警告: 无法找到最佳模型文件 {best_model_path}，跳过最终测试评估")
    
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}, 模型保存在: {best_model_path}")
    return best_val_acc

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# log
class Logger(object):
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    setup_seed(2021)
    sys.stdout = Logger('result.txt', sys.stdout)
    sys.stderr = Logger('error.txt', sys.stderr)
    main()
