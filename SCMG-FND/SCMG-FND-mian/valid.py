import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

def validate(modelConfig, loader, model, criterion):
    """
    验证模型的主要函数
    
    Args:
        modelConfig: 模型配置字典
        loader: 数据加载器
        model: 模型实例
        criterion: 损失函数
    
    Returns:
        valid_loss: 平均验证损失
        results: 预测结果
        truths: 真实标签
        explanations (可选): 如果启用可解释性，返回解释结果
    """
    # 参数初始化
    model.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    
    # 获取配置参数
    device = torch.device(modelConfig.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    diffusion_loss_weight = modelConfig.get("diffusion_loss_weight", 0.008)
    
    # 如果启用可解释性，初始化存储解释结果的列表
    enable_explanation = modelConfig.get("enable_explanation", False)
    explanations = [] if enable_explanation else None
    
    with torch.no_grad():
        for batch_idx, (text_batch, audio_batch, video_batch, y_batch, idx, file_names) in enumerate(loader):
            # 确保批次大小与模型配置一致
            batch_size = text_batch.size(0)
            if batch_size != modelConfig["batch_size"] and not modelConfig.get("drop_last_batch", False):
                print(f"警告: 验证批次大小 {batch_size} 小于配置的 {modelConfig['batch_size']}，可能影响性能")
            
            # 移动数据到设备
            text_batch = text_batch.to(device)
            audio_batch = audio_batch.to(device)
            video_batch = video_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 确保数据类型正确
            if text_batch.dtype != torch.float32:
                text_batch = text_batch.float()
            if audio_batch.dtype != torch.float32:
                audio_batch = audio_batch.float()
            if video_batch.dtype != torch.float32:
                video_batch = video_batch.float()
            
            try:
                # 根据是否启用可解释性选择不同的前向传播方式
                if enable_explanation:
                    loss, y_pred, explanation_dict = model(text_batch, audio_batch, video_batch, y_batch, return_explanations=True)
                else:
                    loss, y_pred = model(text_batch, audio_batch, video_batch, y_batch)
                
                # 使用权重因子调整损失
                loss = loss * diffusion_loss_weight
                
                # 累计损失和批次大小
                total_loss += loss.item() * batch_size
                total_batch_size += batch_size
                
                # 收集结果和真实标签
                results.append(y_pred.detach().cpu())
                truths.append(y_batch.detach().cpu())
                
                # 如果启用可解释性，保存解释结果
                if enable_explanation:
                    # 为每个样本添加元数据并添加到解释列表中
                    for i in range(batch_size):
                        sample_explanation = {
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'file_name': file_names[i] if file_names is not None else f"sample_{batch_idx}_{i}",
                            'label': y_batch[i].item(),
                            'predicted_class': torch.argmax(y_pred[i]).item(),
                            'prediction_confidence': torch.softmax(y_pred[i], dim=-1).max().item()
                        }
                        
                        # 添加可解释性结果
                        for key, value in explanation_dict.items():
                            if isinstance(value, torch.Tensor):
                                # 处理每个样本的张量数据
                                if value.dim() > 1 and i < value.size(0):
                                    sample_explanation[key] = value[i].detach().cpu().numpy()
                            elif isinstance(value, list) and i < len(value):
                                # 处理列表类型数据
                                sample_explanation[key] = value[i]
                            elif isinstance(value, dict):
                                # 处理字典类型数据
                                sample_dict = {}
                                for k, v in value.items():
                                    if isinstance(v, torch.Tensor) and v.dim() > 1 and i < v.size(0):
                                        sample_dict[k] = v[i].detach().cpu().numpy()
                                    elif isinstance(v, list) and i < len(v):
                                        sample_dict[k] = v[i]
                                sample_explanation[key] = sample_dict
                        
                        explanations.append(sample_explanation)
                
                # 可选：打印进度
                if batch_idx % 10 == 0:
                    current_loss = loss.item() * diffusion_loss_weight
                    print(f'验证批次 [{batch_idx}/{len(loader)}], 损失: {current_loss:.4f}')
                    
            except RuntimeError as e:
                if modelConfig.get("skip_error_batches", True):
                    print(f"警告: 处理验证批次 {batch_idx} 时出错, 跳过这个批次. 错误: {str(e)}")
                    continue
                else:
                    raise e
    
    # 计算平均损失
    valid_loss = total_loss / total_batch_size if total_batch_size > 0 else float('inf')
    
    # 合并所有批次的结果
    results = torch.cat(results, dim=0) if results else torch.tensor([])
    truths = torch.cat(truths, dim=0) if truths else torch.tensor([])
    
    if enable_explanation:
        return valid_loss, results, truths, explanations
    else:
        return valid_loss, results, truths 