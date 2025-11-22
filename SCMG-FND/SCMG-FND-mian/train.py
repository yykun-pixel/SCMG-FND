from typing import Dict
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import os
import numpy as np
import torch.optim as optim

# 导入新增的多粒度对比学习和对抗性验证模块
from modules.MultiGranularityContrast import MultiGranularityContrast
from modules.AdversarialVerification import AdversarialVerification

def train(trainer, device, train_loader, val_loader, optimizer, epoch, 
        modelConfig, criterion=None, contrast_module=None, adv_framework=None):
    """
    训练一个epoch
    
    Args:
        trainer: 模型
        device: 计算设备
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        epoch: 当前epoch
        modelConfig: 模型配置字典
        criterion: 损失函数
        contrast_module: 多粒度对比学习模块
        adv_framework: 对抗性验证框架
        
    Returns:
        train_loss: 训练损失
        train_acc: 训练准确率
        valid_loss: 验证损失
        valid_acc: 验证准确率
        explanations: 可解释性结果（如果启用）
    """
    # 设置模型为训练模式
    trainer.train()
    len_train = len(train_loader)
    
    # 梯度累积设置
    gradient_accumulation_steps = modelConfig.get('gradient_accumulation_steps', 1)
    effective_batch_size = modelConfig.get('batch_size', 32) * gradient_accumulation_steps
    print(f"使用梯度累积: 步数={gradient_accumulation_steps}, 有效批量大小={effective_batch_size}")
    
    # 初始化指标
    total_loss = 0
    total_acc = 0
    bsz_sum = 0
    diffusion_loss_sum = 0
    classification_loss_sum = 0
    explain_loss_sum = 0
    contrast_loss_sum = 0   # 多粒度对比损失总和
    adv_loss_sum = 0        # 对抗性验证损失总和
    
    # 如果没有传入对比学习模块但配置中启用了，则创建一个新的模块
    if contrast_module is None and modelConfig.get("use_multi_granularity_contrast", False):
        print("创建新的多粒度对比学习模块...")
        contrast_module = MultiGranularityContrast(
            feature_dim=modelConfig["unified_size"],
            projection_dim=modelConfig.get("contrast_projection_dim", 64),
            temperature=modelConfig.get("contrast_temperature", 0.1),
            spatial_regions=modelConfig.get("contrast_spatial_regions", 4),
            temporal_segments=modelConfig.get("contrast_temporal_segments", 8),
            modal_components=3  # 文本、音频、视频三种模态
        ).to(device)
    
    # 如果没有传入对抗性验证框架但配置中启用了，则创建一个新的框架
    if adv_framework is None and modelConfig.get("use_adversarial_verification", False):
        print("创建新的对抗性验证框架...")
        adv_framework = AdversarialVerification(
            feature_dim=modelConfig["unified_size"],
            hidden_dim=modelConfig.get("adv_hidden_dim", 256),
            z_dim=modelConfig.get("adv_z_dim", 64),
            num_layers=modelConfig.get("adv_num_layers", 2),
            dropout=modelConfig.get("adv_dropout", 0.3)
        ).to(device)
    
    # 定义对抗验证的优化器
    adv_optimizer = None
    if adv_framework is not None:
        adv_optimizer = torch.optim.Adam([
            {'params': adv_framework.encoder.parameters()},
            {'params': adv_framework.generator.parameters()},
            {'params': adv_framework.discriminator.parameters(), 'lr': 1e-4}  # 判别器使用较小的学习率
        ], lr=2e-4, betas=(0.5, 0.999))

    # 在TQDM中显示进度条
    with tqdm(total=len_train, desc=f'Epoch {epoch + 1}/{modelConfig["epoch"]}', unit='batch', ncols=100) as pbar:
        for i, batch_data in enumerate(train_loader):
            # 解析批次数据
            try:
                # 根据数据加载器的返回格式解析批次数据
                if isinstance(batch_data, dict):
                    # 如果是字典格式
                    texts = batch_data.get("text").float().to(device)
                    audios = batch_data.get("audioframes").float().to(device)
                    videos = batch_data.get("frames").float().to(device)
                    labels = batch_data.get("label").long().to(device)
                    comments = batch_data.get("comments", torch.zeros(1)).to(device)
                    c3d = batch_data.get("c3d", torch.zeros(1)).to(device)
                    user_intro = batch_data.get("user_intro", torch.zeros(1)).to(device)
                    gpt_description = batch_data.get("gpt_description", torch.zeros(1)).to(device)
                    implicit_opinion_data = batch_data.get('implicit_opinion_data')  # 新增：获取隐式意见数据
                    print(f"🔍 字典批次中的隐式意见数据: 存在={implicit_opinion_data is not None}, 非空个数={sum(1 for x in implicit_opinion_data if x is not None) if implicit_opinion_data else 0}")
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 4:
                    # 如果是列表或元组格式
                    texts = batch_data[0].float().to(device)
                    audios = batch_data[1].float().to(device)
                    videos = batch_data[2].float().to(device)
                    labels = batch_data[3].long().to(device)
                    # 如果有更多元素，继续解包
                    comments = batch_data[4].to(device) if len(batch_data) > 4 else torch.zeros(1).to(device)
                    c3d = batch_data[5].to(device) if len(batch_data) > 5 else torch.zeros(1).to(device)
                    user_intro = batch_data[6].to(device) if len(batch_data) > 6 else torch.zeros(1).to(device)
                    gpt_description = batch_data[7].to(device) if len(batch_data) > 7 else torch.zeros(1).to(device)
                    implicit_opinion_data = batch_data[8] if len(batch_data) > 8 else None # 新增：获取隐式意见数据（保持字典格式）
                else:
                    raise ValueError(f"无法解析批次数据，格式: {type(batch_data)}")
                
                # 检查批次大小
                batch_size = labels.size(0)
            except Exception as e:
                print(f"解析批次数据时出错: {e}")
                print(f"批次数据类型: {type(batch_data)}")
                continue
            
            # 仅在梯度累积的第一步或不使用梯度累积时清零梯度
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # 关闭解释模块的梯度计算以提高训练速度
            if hasattr(trainer, 'explainer') and epoch < modelConfig.get('explain_start_epoch', 5):
                for param in trainer.explainer.parameters():
                    param.requires_grad = False
            elif hasattr(trainer, 'explainer'):
                for param in trainer.explainer.parameters():
                    param.requires_grad = True
            
            # 混合精度训练
            use_amp = modelConfig.get('mixed_precision', False) and torch.cuda.is_available()
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 判断是否需要特征用于对比学习或对抗验证
                need_features_for_contrast = contrast_module is not None and epoch >= modelConfig.get('contrast_start_epoch', 3)
                need_features_for_adv = adv_framework is not None and adv_optimizer is not None and epoch >= modelConfig.get('adv_start_epoch', 5)
                
                # 当需要解释性分析或对比学习或对抗验证时，启用return_explanations
                if (modelConfig.get("enable_explanation", False) and epoch >= modelConfig.get('explain_start_epoch', 5)) or need_features_for_contrast or need_features_for_adv:
                    # 带解释的前向传播
                    loss, outputs, explanation_results = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=True, implicit_opinion_data=implicit_opinion_data)
                    # 提取特征用于对比学习和对抗验证
                    unified_features = explanation_results.get('unified_features', None)
                    text_features = explanation_results.get('text_features', None)
                    audio_features = explanation_results.get('audio_features', None)
                    video_features = explanation_results.get('video_features', None)
                else:
                    # 不需要解释性结果和特征
                    loss, outputs, *_ = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=False, implicit_opinion_data=implicit_opinion_data)
                    unified_features, text_features, audio_features, video_features = None, None, None, None
                
                # 计算主要扩散损失
                diffusion_loss = loss * modelConfig.get('diffusion_loss_weight', 0.008)
                
                # 计算分类损失
                classification_loss = torch.tensor(0.0, device=device)
                if criterion is not None:
                    classification_loss = criterion(outputs, labels)
                else:
                    classification_loss = F.cross_entropy(outputs, labels)
                
                # 计算解释损失
                explain_loss = torch.tensor(0.0, device=device)
                if modelConfig.get("enable_explanation", False) and epoch >= modelConfig.get('explain_start_epoch', 5):
                    explain_loss = explanation_results.get('explain_loss', torch.tensor(0.0, device=device))
                
                # 计算多粒度对比学习损失
                contrast_loss = torch.tensor(0.0, device=device)
                if contrast_module is not None and epoch >= modelConfig.get('contrast_start_epoch', 3):
                    # 确保有必要的特征
                    if text_features is not None and audio_features is not None and video_features is not None and unified_features is not None:
                        contrast_loss = contrast_module(
                            text_features=text_features,
                            audio_features=audio_features,
                            video_features=video_features,
                            global_features=unified_features,
                            labels=labels
                        )
                    else:
                        print("警告: 缺少对比学习所需的特征")
                
                # 计算对抗性验证损失
                adv_loss = torch.tensor(0.0, device=device)
                if adv_framework is not None and adv_optimizer is not None and epoch >= modelConfig.get('adv_start_epoch', 5):
                    if unified_features is not None:
                        # 训练判别器
                        adv_optimizer.zero_grad()
                        real_score, fake_score, _ = adv_framework.forward_discriminator(unified_features.detach())
                        d_loss = adv_framework.compute_discriminator_loss(real_score, fake_score)
                        d_loss.backward()
                        adv_optimizer.step()
                        
                        # 训练生成器
                        adv_optimizer.zero_grad()
                        gen_score, _, _ = adv_framework.forward_generator(unified_features.detach())
                        g_loss = adv_framework.compute_generator_loss(gen_score)
                        
                        # 计算对抗验证损失
                        adv_loss = g_loss * modelConfig.get('adv_weight', 0.1)
                    else:
                        print("警告: 缺少对抗验证所需的特征")
                
                # 组合所有损失
                total_loss_batch = classification_loss + diffusion_loss + \
                                   modelConfig.get('explain_weight', 0.1) * explain_loss + \
                                   modelConfig.get('contrast_weight', 0.1) * contrast_loss + \
                                   modelConfig.get('adv_weight', 0.1) * adv_loss
                
                # 如果使用梯度累积，则将损失除以累积步数
                if gradient_accumulation_steps > 1:
                    total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # 反向传播
            total_loss_batch.backward()
            
            # 只在累积完成后更新权重
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len_train:
                # 梯度裁剪，防止梯度爆炸
                if modelConfig.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), modelConfig.get('grad_clip', 1.0))
                
                optimizer.step()
                optimizer.zero_grad()  # 确保梯度清零
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            
            # 累加指标
            total_loss += total_loss_batch.item() * labels.size(0) * (1 if gradient_accumulation_steps <= 1 else gradient_accumulation_steps)
            total_acc += accuracy * labels.size(0)
            bsz_sum += labels.size(0)
            diffusion_loss_sum += diffusion_loss.item() * labels.size(0)
            classification_loss_sum += classification_loss.item() * labels.size(0)
            explain_loss_sum += explain_loss.item() * labels.size(0)
            contrast_loss_sum += contrast_loss.item() * labels.size(0)
            adv_loss_sum += adv_loss.item() * labels.size(0)
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': total_loss / max(1, bsz_sum),
                'acc': total_acc / max(1, bsz_sum),
                'clf_loss': classification_loss_sum / max(1, bsz_sum),
                'diff_loss': diffusion_loss_sum / max(1, bsz_sum),
                'expl_loss': explain_loss_sum / max(1, bsz_sum),
                'cont_loss': contrast_loss_sum / max(1, bsz_sum),
                'adv_loss': adv_loss_sum / max(1, bsz_sum)
            })
            pbar.update(1)
    
    # 计算平均指标
    train_loss = total_loss / max(1, bsz_sum)
    train_acc = total_acc / max(1, bsz_sum)
    
    # 验证
    print("开始验证...")
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    # 使用valid函数进行验证
    if modelConfig.get("enable_explanation", False):
        valid_loss, valid_results, valid_truths, explanations = valid(val_loader, trainer, criterion, modelConfig)
    else:
        valid_loss, valid_results, valid_truths, explanations = valid(val_loader, trainer, criterion, modelConfig)
    
    # 计算验证准确率
    valid_acc = 0.0
    if len(valid_results) > 0 and len(valid_truths) > 0:
        valid_acc = (valid_results == valid_truths).float().mean().item()
    
    # 打印结果
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
    
    # 返回训练和验证结果 - 始终返回5个值，包括explanations（即使是None）
    return train_loss, train_acc, valid_loss, valid_acc, explanations

def calculate_f1(y_pred, y_true):
    """计算F1分数"""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')

def calculate_auc(y_pred, y_true):
    """计算AUC分数"""
    from sklearn.metrics import roc_auc_score
    try:
        # 对于二分类问题，转换预测值为概率再计算AUC
        return roc_auc_score(y_true, y_pred)
    except:
        # 对于多分类问题或格式不匹配的情况，返回0
        return 0.0

def save_explanations(explanations, checkpoint_path, epoch):
    """保存解释结果"""
    import json
    
    explanation_dir = os.path.join(checkpoint_path, 'explanations')
    os.makedirs(explanation_dir, exist_ok=True)
    
    # 保存为JSON文件
    with open(os.path.join(explanation_dir, f'explanations_epoch_{epoch+1}.json'), 'w') as f:
        json.dump(explanations, f, indent=2)
    
    print(f"解释结果已保存: explanations_epoch_{epoch+1}.json")

def valid(loader, trainer, criterion, modelConfig: Dict):
    trainer.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    
    # 用于收集可解释性结果
    explanations = [] if modelConfig.get("enable_explanation", False) else None
    
    # 获取混合精度训练的设置
    use_amp = modelConfig.get("use_amp", False) and torch.cuda.is_available()
    diffusion_loss_weight = modelConfig.get("diffusion_loss_weight", 0.008)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch_size = batch["label"].size(0)
            
            # 特殊处理最后一个批次（通常是不完整的）
            is_last_batch = batch_idx == len(loader) - 1
            if is_last_batch and batch_size == 1:
                print(f"处理验证集中的最后一个批次(批次大小={batch_size})，可能需要特殊处理")
            
            # 确保batch_size不超过模型配置
            if batch_size != modelConfig["batch_size"]:
                print(f"警告: 验证批次大小 {batch_size} 与模型配置 {modelConfig['batch_size']} 不匹配")
                
                # 尝试更新trainer中的batch_size参数
                if hasattr(trainer, 'batch_size'):
                    trainer.batch_size = batch_size
                    print(f"已更新trainer.batch_size为 {batch_size}")
                
            total_batch_size += batch_size
            
            # 提取各模态数据
            texts = batch["text"]
            audios = batch["audioframes"]
            videos = batch["frames"]
            comments = batch["comments"]
            labels = batch["label"]
            c3d = batch["c3d"]
            user_intro = batch["user_intro"]
            gpt_description = batch["gpt_description"]
            implicit_opinion_data = batch.get('implicit_opinion_data')  # 新增：获取隐式意见数据
            
            # 检查特殊情况：videos是二维的（可能是最后一个批次）
            if len(videos.shape) == 2:
                print(f"检测到videos是2D张量: {videos.shape}，尝试调整形状")
                try:
                    # 如果是[seq_len, features]，扩展为[1, seq_len, features]
                    videos = videos.unsqueeze(0)
                    print(f"调整后videos形状: {videos.shape}")
                except Exception as e:
                    print(f"调整videos形状失败: {e}")
            
            # 检查特殊情况：如果是单样本，确保所有特征维度都正确
            if batch_size == 1:
                print(f"批次大小为1，确保所有特征形状正确")
                
                # 确保texts至少是2D
                if len(texts.shape) == 1:
                    texts = texts.unsqueeze(0)
                    print(f"调整后texts形状: {texts.shape}")
                
                # 同样处理其他特征
                if len(audios.shape) == 2:  # 如果是[seq_len, features]
                    audios = audios.unsqueeze(0)
                    print(f"调整后audios形状: {audios.shape}")
            
            # 移动到GPU
            if torch.cuda.is_available():
                audios = audios.cuda()
                texts = texts.cuda()
                videos = videos.cuda()
                comments = comments.cuda()
                labels = labels.cuda()
                c3d = c3d.cuda()
                user_intro = user_intro.cuda()
                gpt_description = gpt_description.cuda()

            try:
                # 使用混合精度验证
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # 根据是否启用可解释性选择不同的前向传播方式
                        if modelConfig.get("enable_explanation", False):
                            loss, pred, explanation = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=True, implicit_opinion_data=implicit_opinion_data)
                            # 收集可解释性结果
                            if explanations is not None:
                                # 为每个样本添加批次索引
                                for i in range(batch_size):
                                    batch_explanation = {
                                        'batch_idx': batch_idx,
                                        'sample_idx': i,
                                        'label': labels[i].item(),
                                    }
                                    # 将解释字典中的张量转换为CPU上的NumPy数组
                                    for key, tensor in explanation.items():
                                        if isinstance(tensor, torch.Tensor):
                                            if tensor.dim() > 1 and i < tensor.shape[0]:
                                                batch_explanation[key] = tensor[i].detach().cpu().numpy()
                                    explanations.append(batch_explanation)
                        else:
                            loss, pred, *_ = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=False, implicit_opinion_data=implicit_opinion_data)
                        
                        # 获取预测类别
                        _, y = torch.max(pred, 1)
                        
                        # 计算损失
                        diffusion_loss = loss * diffusion_loss_weight
                else:
                    # 常规验证（无混合精度）
                    # 根据是否启用可解释性选择不同的前向传播方式
                    if modelConfig.get("enable_explanation", False):
                        loss, pred, explanation = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=True, implicit_opinion_data=implicit_opinion_data)
                        # 收集可解释性结果
                        if explanations is not None:
                            # 为每个样本添加批次索引
                            for i in range(batch_size):
                                batch_explanation = {
                                    'batch_idx': batch_idx,
                                    'sample_idx': i,
                                    'label': labels[i].item(),
                                }
                                # 将解释字典中的张量转换为CPU上的NumPy数组
                                for key, tensor in explanation.items():
                                    if isinstance(tensor, torch.Tensor):
                                        if tensor.dim() > 1 and i < tensor.shape[0]:
                                            batch_explanation[key] = tensor[i].detach().cpu().numpy()
                                explanations.append(batch_explanation)
                    else:
                        loss, pred, *_ = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=False, implicit_opinion_data=implicit_opinion_data)
                    
                    # 获取预测类别
                    _, y = torch.max(pred, 1)
                    
                    # 计算损失
                    diffusion_loss = loss * diffusion_loss_weight
                
                # 收集结果
                results.append(y)
                truths.append(labels)
                total_loss += diffusion_loss
                
            except RuntimeError as e:
                print(f"验证中出现运行时错误: {e}")
                print(f"错误发生时的批次大小: {batch_size}")
                try:
                    print(f"形状信息: texts={texts.shape}, audios={audios.shape}, videos={videos.shape}, c3d={c3d.shape}")
                except:
                    print("无法打印形状信息")
                
                if batch_size == 1 and is_last_batch:
                    print("这是批次大小为1的最后一个批次，跳过而不中断验证过程")
                    continue
                else:
                    # 对于大批次，我们需要处理这个批次或终止验证
                    if modelConfig.get("skip_error_batches", True):
                        print("根据配置，跳过这个错误的批次")
                        continue
                    else:
                        print("根据配置，中断验证过程")
                        break
            
            except Exception as e:
                print(f"验证中出现其他错误: {str(e)}")
                # 通常跳过这个批次
                if modelConfig.get("skip_error_batches", True):
                    continue
                else:
                    break

        # 确保有结果才进行连接
        if results:
            try:
                # 尝试连接所有结果
                results = torch.cat(results)
                truths = torch.cat(truths)
                # 始终返回4个值，包括explanations（即使是None）
                return total_loss, results, truths, explanations
            except RuntimeError as e:
                print(f"连接验证结果时出错: {e}")
                # 尝试处理不同大小的结果张量
                print("尝试处理不同大小的结果张量...")
                
                # 查找第一个非空张量的形状和设备
                first_result = None
                first_truth = None
                for r in results:
                    if r.numel() > 0:
                        first_result = r
                        break
                for t in truths:
                    if t.numel() > 0:
                        first_truth = t
                        break
                
                if first_result is not None and first_truth is not None:
                    # 调整所有张量的形状
                    adjusted_results = []
                    adjusted_truths = []
                    
                    for r in results:
                        if r.numel() > 0:
                            adjusted_results.append(r)
                    
                    for t in truths:
                        if t.numel() > 0:
                            adjusted_truths.append(t)
                    
                    # 再次尝试连接
                    try:
                        results = torch.cat(adjusted_results)
                        truths = torch.cat(adjusted_truths)
                        return total_loss, results, truths, explanations
                    except:
                        print("处理后仍然无法连接结果，返回空结果")
        
        print("警告: 验证过程中没有生成任何结果")
        # 返回空结果，但仍然保持4个返回值
        return 0.0, torch.tensor([]), torch.tensor([]), explanations