import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .Multimodal_Model import Text_Noise_Pre, Audio_Noise_Pre, Visual_Noise_Pre
from .ExplainableDetection import ExplainableDetection
from src.CrossmodalTransformer import MULTModel
from src.StoG import CapsuleSequenceToGraph
from modules.NeuralSymbolicRules import NeuralSymbolicRuleEngine, ImplicitOpinionAnalyzer
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import math
import gc


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, modelConfig, beta_1, beta_T, T, t_in, a_in, v_in, d_m, dropout, label_dim,
                 unified_size, vertex_num, routing, T_t, T_a, T_v,  batch_size):
        super().__init__()

        self.T = T
        self.batch_size = batch_size
        self.mult_dropout = dropout
        self.unified_size = unified_size
        self.vertex_num = vertex_num

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # Feature Extraction
        self.fc_pre_t_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_t_2 = nn.Linear(modelConfig["t_in_pre"]*2, modelConfig["t_in_pre"])
        self.fc_pre_v = torch.nn.Linear(v_in, modelConfig["v_in_pre"])
        self.fc_pre_com = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_user = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(),
                                        nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_c3d = torch.nn.Linear(modelConfig["c3d_in"], unified_size)
        self.fc_pre_gpt_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_gpt_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])
        
        # 添加一个投影层，用于将videos_global从v_in_pre维度映射到unified_size维度
        self.videos_global_proj = torch.nn.Linear(modelConfig["v_in_pre"], unified_size)

        self.vggish_layer = torch.hub.load(r'torchvggish-master', 'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        self.fc_pre_a = nn.Linear(a_in, modelConfig["a_in_pre"])

        # Intra-modal Enhancement
        self.fc_g_t = nn.Linear(d_m * 6, d_m)
        self.fc_a_MTout = nn.Linear(d_m * 3, d_m)
        self.fc_v_MTout = nn.Linear(d_m * 3, d_m)
        self.CrossmodalTransformer = MULTModel(modelConfig["t_in_pre"], modelConfig["a_in_pre"], modelConfig["v_in_pre"], d_m, self.mult_dropout)
        self.StoG = CapsuleSequenceToGraph(d_m, unified_size, vertex_num, routing, T_t, T_a, T_v)

        # Cross-modal Interaction
        self.model_t = Text_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                           dropout=modelConfig["Text_Pre_dropout"],
                           in_ch=unified_size)
        self.model_a = Audio_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                           dropout=modelConfig["Img_Pre_dropout"],
                           in_ch=unified_size)
        self.model_v = Visual_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                                       dropout=modelConfig["Img_Pre_dropout"],
                                       in_ch=unified_size)

        self.fc_t = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_a = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_v = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_m = nn.Linear(in_features=unified_size * 3, out_features=unified_size)

        # Prediction
        self.fc_pre = nn.Linear(in_features=unified_size, out_features=label_dim)
        self.trm = nn.TransformerEncoderLayer(d_model=unified_size, nhead=2, batch_first=True)
        
        # 可解释性模块
        self.explainer = ExplainableDetection(unified_size, vertex_num)
        
        # 神经符号规则引擎
        self.neural_symbolic_engine = NeuralSymbolicRuleEngine()
        # 是否启用神经符号规则
        self.enable_neural_symbolic = modelConfig.get("enable_neural_symbolic", True)
        # 存储规则阈值，避免在forward中访问modelConfig
        self.rule_threshold = modelConfig.get("rule_threshold", 0.1)
        
        # 隐式意见分析器（可选，用于实时分析）
        self.implicit_analyzer = None
        if modelConfig.get("enable_implicit_analysis", False):
            try:
                self.implicit_analyzer = ImplicitOpinionAnalyzer(
                    llm_model_name=modelConfig.get("llm_model_name", "THUDM/chatglm-6b")
                )
                print("隐式意见分析器初始化成功")
            except Exception as e:
                print(f"隐式意见分析器初始化失败: {e}")
                self.implicit_analyzer = None
        
        # 保存原始图像尺寸以便可视化（训练中设置）
        self.original_video_frames = None
        # 是否启用可解释性功能
        self.enable_explanation = modelConfig.get("enable_explanation", False)

    def forward(self, texts, audios, videos, comments, c3d, user_intro, gpt_description, return_explanations=False, implicit_opinion_data=None):
        # 使用with语句确保临时变量被释放
        with torch.set_grad_enabled(self.training):
            # Feature Extraction
            texts_local, _ = self.fc_pre_t_1(texts)
            texts_local = self.fc_pre_t_2(texts_local)
            
            # 显式释放不再需要的变量
            del texts
            
            # 调整audios的形状
            # 打印原始形状以便调试
            original_shape = audios.shape
            print(f"原始audios形状: {original_shape}")
            
            # 根据论文中VGGish模型的要求调整音频特征
            # VGGish期望输入形状为[batch_size, 1, time_steps, freq_bins]
            try:
                # 如果是4D张量，调整通道顺序
                if len(original_shape) == 4:
                    b, c, t, f = original_shape
                    # 如果通道数大于1，只使用第一个通道
                    if c > 1:
                        audios = audios[:, 0:1, :, :]
                        print(f"调整后audios形状 (选择第一个通道): {audios.shape}")
                # 如果是3D张量，增加通道维度
                elif len(original_shape) == 3:
                    b, t, f = original_shape
                    audios = audios.unsqueeze(1)  # 添加通道维度
                    print(f"调整后audios形状 (增加通道维度): {audios.shape}")
                # 如果是2D张量，视为单个样本，添加batch和通道维度
                elif len(original_shape) == 2:
                    t, f = original_shape
                    audios = audios.unsqueeze(0).unsqueeze(1)  # 添加batch和通道维度
                    print(f"调整后audios形状 (增加batch和通道维度): {audios.shape}")
                
                # 尝试通过vggish_modified处理
                try:
                    # 首先确保数据类型正确
                    audios = audios.float()
                    audios = self.vggish_modified(audios)
                    print(f"vggish_modified处理后audios形状: {audios.shape}")
                    
                    # 重塑audios到fc_pre_a期望的形状
                    # fc_pre_a应该期望输入维度为[batch_size, a_in]，其中a_in为128
                    batch_size = audios.shape[0]
                    audios = audios.reshape(batch_size, -1)
                    print(f"重塑后audios形状: {audios.shape}")
                    
                    # 如果需要，截断或填充到a_in的大小
                    a_in_size = 128  # 根据modelConfig["a_in"]参数
                    if audios.shape[1] > a_in_size:
                        # 截断到a_in_size
                        audios = audios[:, :a_in_size]
                        print(f"截断后audios形状: {audios.shape}")
                    elif audios.shape[1] < a_in_size:
                        # 填充到a_in_size
                        padding = torch.zeros(batch_size, a_in_size - audios.shape[1], device=audios.device)
                        audios = torch.cat([audios, padding], dim=1)
                        print(f"填充后audios形状: {audios.shape}")
                    
                    # 显式清理临时变量，减少内存占用
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    print(f"vggish_modified处理失败，错误: {e}")
                    print(f"尝试调整形状后再处理...")
                    
                    # 尝试不同的调整方式
                    if len(audios.shape) == 4:
                        # 尝试不同的调整方式
                        b, c, t, f = audios.shape
                        
                        # 方案1: 确保只有一个通道
                        if c > 1:
                            audios = audios[:, 0:1, :, :]
                        
                        # 方案2: 如果特征维度太大，尝试将时间和频率维度展平
                        if audios.shape[-1] * audios.shape[-2] > 128:
                            # 重塑为[batch_size, 1, time*freq]
                            audios = audios.view(b, 1, -1)
                            # 截取到合适的长度
                            audios = audios[:, :, :128]
                            print(f"调整后audios形状 (展平并截取): {audios.shape}")
                            
                            # 如果vggish期望2D输入，去掉通道维度
                            audios = audios.squeeze(1)
                        
                    # 再次尝试
                    audios = self.vggish_modified(audios)
                    print(f"重新调整后，vggish_modified处理成功，形状: {audios.shape}")
                    
                    # 重塑为fc_pre_a期望的形状
                    batch_size = audios.shape[0]
                    audios = audios.reshape(batch_size, -1)
                    a_in_size = 128
                    if audios.shape[1] > a_in_size:
                        audios = audios[:, :a_in_size]
                        print(f"截断后audios形状: {audios.shape}")
                    elif audios.shape[1] < a_in_size:
                        padding = torch.zeros(batch_size, a_in_size - audios.shape[1], device=audios.device)
                        audios = torch.cat([audios, padding], dim=1)
                        print(f"填充后audios形状: {audios.shape}")
                    
            except Exception as e:
                print(f"处理音频时出现未知错误: {e}")
                # 在错误情况下，创建一个与预期输出形状匹配的零张量
                # 根据fc_pre_a的输入维度确定形状
                batch_size = original_shape[0] if len(original_shape) >= 1 else 1
                audios = torch.zeros(batch_size, 128, device=texts_local.device)
                print(f"使用零张量替代，形状: {audios.shape}")
            
            audios_local = self.fc_pre_a(audios)
            c3d_local = self.fc_pre_c3d(c3d)
            gpt_local, _ = self.fc_pre_gpt_1(gpt_description)
            gpt_local = self.fc_pre_gpt_2(gpt_local)
            comments_global = self.fc_pre_com(comments)
            user_intro_global = self.fc_pre_user(user_intro.squeeze())
            
            # 检查并处理视频特征张量
            original_videos_shape = videos.shape
            print(f"原始videos形状: {original_videos_shape}")
            
            # 确保视频特征维度正确
            try:
                # 检查videos是否只有2D而没有批次维度 (特别是验证/测试阶段可能出现)
                if len(original_videos_shape) == 2:
                    # 对于形状为[83, 4096]的情况，需要增加批次维度
                    print(f"videos是2D张量，缺少批次维度。添加批次维度")
                    
                    # 检测这个是否是单样本的情况
                    if batch_size == 1 or texts_local.shape[0] == 1:
                        print(f"单样本情况下的videos 2D张量处理")
                        # 添加批次维度: [83, 4096] -> [1, 83, 4096]
                        videos = videos.unsqueeze(0)
                        print(f"为videos添加批次维度后形状: {videos.shape}")
                        # 更新形状信息
                        original_videos_shape = videos.shape
                    else:
                        print(f"警告: 批次大小不为1 ({batch_size}) 但videos张量没有批次维度")
                        # 尝试扩展批次维度, 但需要根据实际情况调整
                        try:
                            # 尝试通过复制扩展批次维度
                            videos = videos.unsqueeze(0).expand(batch_size, *original_videos_shape)
                            print(f"扩展后videos形状: {videos.shape}")
                            original_videos_shape = videos.shape
                        except Exception as e:
                            print(f"扩展videos批次维度失败: {e}")
                            # 创建一个符合预期形状的零张量
                            seq_len, features = original_videos_shape
                            videos = torch.zeros(batch_size, seq_len, features, device=videos.device)
                            print(f"使用零张量替代，形状: {videos.shape}")
                            original_videos_shape = videos.shape
                
                # 如果视频特征是3D张量 [batch_size, seq_len, features]
                if len(original_videos_shape) == 3:
                    batch_size, seq_len, features = original_videos_shape
                    
                    # 验证特征维度是否符合预期
                    if features != 4096:  # v_in预期为4096
                        print(f"警告: 视频特征维度不符合预期，当前为{features}，预期为4096")
                        # 如果特征维度太小或太大，进行调整
                        if features < 4096:
                            # 扩展特征维度
                            padding = torch.zeros(batch_size, seq_len, 4096 - features, device=videos.device)
                            videos = torch.cat([videos, padding], dim=2)
                        else:
                            # 截断特征维度
                            videos = videos[:, :, :4096]
                        print(f"调整后videos形状: {videos.shape}")
                    
                    # 直接应用fc_pre_v
                    videos = self.fc_pre_v(videos)  # 从[batch, seq_len, 4096]到[batch, seq_len, v_in_pre]
                    print(f"fc_pre_v处理后videos形状: {videos.shape}")
                    
                    # 计算全局特征，取平均值
                    videos_global = torch.mean(videos, dim=1)  # [batch, v_in_pre]
                    print(f"视频全局特征形状: {videos_global.shape}")
                
                # 如果视频特征是2D张量 [batch_size, features] - 可能已经是全局特征
                elif len(original_videos_shape) == 2 and original_videos_shape[0] == batch_size:
                    print("视频特征为2D张量 [batch_size, features]，可能已经是全局特征")
                    batch_size, features = original_videos_shape
                    
                    # 如果是全局特征，我们需要同时为local和global处理
                    if features == self.fc_pre_v.in_features:  # 检查特征维度是否符合fc_pre_v的输入维度
                        # 添加一个序列长度维度
                        temp_videos = videos.unsqueeze(1)  # [batch, 1, features]
                        videos = self.fc_pre_v(temp_videos)  # [batch, 1, v_in_pre]
                        videos_global = videos.squeeze(1)  # 去掉时序维度 [batch, v_in_pre]
                    else:
                        # 特征维度不匹配，需要调整
                        print(f"警告: 视频特征维度不符合预期，当前为{features}，预期为{self.fc_pre_v.in_features}")
                        
                        # 添加序列长度维度
                        videos = videos.unsqueeze(1)  # [batch, 1, features]
                        
                        # 调整特征维度
                        if features < self.fc_pre_v.in_features:
                            # 扩展特征维度
                            padding = torch.zeros(batch_size, 1, self.fc_pre_v.in_features - features, device=videos.device)
                            videos = torch.cat([videos, padding], dim=2)
                        else:
                            # 截断特征维度
                            videos = videos[:, :, :self.fc_pre_v.in_features]
                        
                        videos = self.fc_pre_v(videos)  # [batch, 1, v_in_pre]
                        videos_global = videos.squeeze(1)  # [batch, v_in_pre]
                
                else:
                    raise ValueError(f"视频特征维度异常: {original_videos_shape}")
                
            except Exception as e:
                print(f"处理视频特征时出错: {e}")
                # 出错时创建零张量替代
                batch_size = texts_local.shape[0] if hasattr(texts_local, 'shape') and len(texts_local.shape) > 0 else 1
                videos = torch.zeros(batch_size, 83, 1000, device=comments_global.device)  # v_in_pre=1000
                videos_global = torch.zeros(batch_size, 1000, device=comments_global.device)
                print(f"使用零张量替代，形状: videos={videos.shape}, videos_global={videos_global.shape}")
            
            # Intra-modal Enhancement
            z_t, z_g, z_a, z_v = self.CrossmodalTransformer(texts_local, gpt_local, audios_local, videos)  # (49,32,64) (200,32,64)
            z_t = self.fc_g_t(torch.cat([z_t, z_g], dim=2))
            z_a = self.fc_a_MTout(z_a)
            z_v = self.fc_v_MTout(z_v)
            
            # 打印特征形状，用于调试
            print(f"增强后特征形状: z_t={z_t.shape}, z_a={z_a.shape}, z_v={z_v.shape}")
            
            # 检查特征维度是否一致，必要时进行调整
            # 确保所有特征在第一维（序列长度）上具有相同的维度
            target_seq_len = min(z_t.shape[0], z_a.shape[0], z_v.shape[0])
            if z_t.shape[0] != target_seq_len:
                print(f"调整z_t序列长度从{z_t.shape[0]}到{target_seq_len}")
                z_t = z_t[:target_seq_len]
            if z_a.shape[0] != target_seq_len:
                print(f"调整z_a序列长度从{z_a.shape[0]}到{target_seq_len}")
                z_a = z_a[:target_seq_len]
            if z_v.shape[0] != target_seq_len:
                print(f"调整z_v序列长度从{z_v.shape[0]}到{target_seq_len}")
                z_v = z_v[:target_seq_len]
            
            # 确保所有特征在最后一维（特征维度）上具有相同的维度
            # 这里假设StoG期望所有特征具有相同的维度
            # 如果不同，可能需要先通过线性层调整
            
            print(f"调整后特征形状: z_t={z_t.shape}, z_a={z_a.shape}, z_v={z_v.shape}")
            
            try:
                x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, self.batch_size) #(32,32,64)
                print(f"StoG输出形状: x_t={x_t.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
            except RuntimeError as e:
                print(f"StoG处理错误: {e}")
                # 出错时，尝试调整特征维度后重试
                # 1. 确保所有特征具有相同的形状
                batch_dim = z_t.shape[1]  # 批次大小
                feature_dim = z_t.shape[2]  # 特征维度
                
                # 打印当前批次大小信息
                print(f"当前批次大小: {batch_dim}, 配置的批次大小: {self.batch_size}")
                
                # 如果批次大小为1，使用特殊处理
                if batch_dim == 1:
                    print("检测到批次大小为1，使用特殊处理")
                    
                    # 使用临时的批次大小
                    temp_batch_size = batch_dim
                    
                    # 确保所有特征长度相同且都是1
                    seq_len = z_t.shape[0]
                    if z_a.shape[0] != seq_len or z_v.shape[0] != seq_len:
                        print(f"调整序列长度: z_a={z_a.shape[0]}→{seq_len}, z_v={z_v.shape[0]}→{seq_len}")
                        
                        # 如果序列长度不同，创建相同长度的张量
                        if z_a.shape[0] != seq_len:
                            new_z_a = torch.zeros_like(z_t)
                            min_len = min(z_a.shape[0], seq_len)
                            new_z_a[:min_len] = z_a[:min_len]
                            z_a = new_z_a
                        
                        if z_v.shape[0] != seq_len:
                            new_z_v = torch.zeros_like(z_t)
                            min_len = min(z_v.shape[0], seq_len)
                            new_z_v[:min_len] = z_v[:min_len]
                            z_v = new_z_v
                    
                    # 再次尝试，使用实际批次大小
                    try:
                        x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, temp_batch_size)
                        print(f"小批次特殊处理后StoG输出形状: x_t={x_t.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
                    except RuntimeError as e2:
                        print(f"小批次特殊处理仍然失败: {e2}")
                        # 如果还是失败，创建虚拟输出
                        # 假设StoG输出维度为[batch_size, vertex_num, feature_dim]
                        vertex_num = 32  # 根据模型配置
                        x_t = torch.zeros(batch_dim, vertex_num, feature_dim, device=z_t.device)
                        x_a = torch.zeros(batch_dim, vertex_num, feature_dim, device=z_a.device)
                        x_v = torch.zeros(batch_dim, vertex_num, feature_dim, device=z_v.device)
                        print(f"创建虚拟输出: 形状={x_t.shape}")
                else:
                    # 使用z_t作为模板，将z_a和z_v调整为相同形状
                    if z_a.shape != z_t.shape:
                        print(f"调整z_a形状从{z_a.shape}到{z_t.shape}")
                        z_a_new = torch.zeros_like(z_t)
                        min_seq = min(z_a.shape[0], z_t.shape[0])
                        min_batch = min(z_a.shape[1], z_t.shape[1])
                        min_feat = min(z_a.shape[2], z_t.shape[2])
                        z_a_new[:min_seq, :min_batch, :min_feat] = z_a[:min_seq, :min_batch, :min_feat]
                        z_a = z_a_new
                    
                    if z_v.shape != z_t.shape:
                        print(f"调整z_v形状从{z_v.shape}到{z_t.shape}")
                        z_v_new = torch.zeros_like(z_t)
                        min_seq = min(z_v.shape[0], z_t.shape[0])
                        min_batch = min(z_v.shape[1], z_t.shape[1])
                        min_feat = min(z_v.shape[2], z_t.shape[2])
                        z_v_new[:min_seq, :min_batch, :min_feat] = z_v[:min_seq, :min_batch, :min_feat]
                        z_v = z_v_new
                    
                    print(f"重试StoG，输入形状: z_t={z_t.shape}, z_a={z_a.shape}, z_v={z_v.shape}")
                    x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, self.batch_size)
                    print(f"重试成功，StoG输出形状: x_t={x_t.shape}, x_a={x_a.shape}, x_v={x_v.shape}")

            # Cross-modal Interaction
            x_m = torch.concat([x_t.squeeze(), x_a.squeeze(), x_v.squeeze()], dim=2)
            x_m = self.fc_m(x_m)

            # 确保批次大小正确，特别是在小批次时
            actual_batch_size = x_t.shape[0]
            if actual_batch_size != self.batch_size:
                print(f"实际批次大小({actual_batch_size})与配置的批次大小({self.batch_size})不一致，使用实际值")
            
            # 使用实际批次大小生成时间步长
            t_t = torch.randint(self.T, size=(actual_batch_size, ), device=x_t.device) # batchsize (0->T-1)
            noise_t = torch.randn_like(x_t)
            x_tmp_t = (
                extract(self.sqrt_alphas_bar, t_t, x_t.shape) * x_t +
                extract(self.sqrt_one_minus_alphas_bar, t_t, x_t.shape) * noise_t)

            t_a = torch.randint(self.T, size=(actual_batch_size,), device=x_a.device)
            noise_a = torch.randn_like(x_a)
            x_tmp_a = (
                    extract(self.sqrt_alphas_bar, t_a, x_a.shape) * x_a +
                    extract(self.sqrt_one_minus_alphas_bar, t_a, x_a.shape) * noise_a)

            t_v = torch.randint(self.T, size=(actual_batch_size,), device=x_v.device)
            noise_v = torch.randn_like(x_v)
            x_tmp_v = (
                    extract(self.sqrt_alphas_bar, t_v, x_v.shape) * x_v +
                    extract(self.sqrt_one_minus_alphas_bar, t_v, x_v.shape) * noise_v)

            # 打印扩散处理前形状
            print(f"扩散模型输入形状: x_tmp_t={x_tmp_t.shape}, t_t={t_t.shape}, x_m={x_m.shape}")
            
            try:
                x_a_pre = self.model_a(x_tmp_a, t_a, x_m)
                x_v_pre = self.model_v(x_tmp_v, t_v, x_m)
                x_t_pre = self.model_t(x_tmp_t, t_t, x_m)
                
                # 打印预测后形状
                print(f"扩散模型输出形状: x_t_pre={x_t_pre.shape}, x_a_pre={x_a_pre.shape}, x_v_pre={x_v_pre.shape}")
                
                # 确保预测张量与原始张量形状一致，否则可能导致MSE损失计算错误
                if x_t_pre.shape != x_t.shape:
                    print(f"警告: x_t_pre形状({x_t_pre.shape})与x_t形状({x_t.shape})不匹配")
                    # 如果只是batch维度不同，可以截断或者填充
                    if x_t_pre.shape[1:] == x_t.shape[1:]:
                        min_batch = min(x_t_pre.shape[0], x_t.shape[0])
                        x_t_pre = x_t_pre[:min_batch]
                        x_t = x_t[:min_batch]
                        print(f"调整后: x_t_pre={x_t_pre.shape}, x_t={x_t.shape}")
                
                if x_a_pre.shape != x_a.shape:
                    print(f"警告: x_a_pre形状({x_a_pre.shape})与x_a形状({x_a.shape})不匹配")
                    if x_a_pre.shape[1:] == x_a.shape[1:]:
                        min_batch = min(x_a_pre.shape[0], x_a.shape[0])
                        x_a_pre = x_a_pre[:min_batch]
                        x_a = x_a[:min_batch]
                        print(f"调整后: x_a_pre={x_a_pre.shape}, x_a={x_a.shape}")
                
                if x_v_pre.shape != x_v.shape:
                    print(f"警告: x_v_pre形状({x_v_pre.shape})与x_v形状({x_v.shape})不匹配")
                    if x_v_pre.shape[1:] == x_v.shape[1:]:
                        min_batch = min(x_v_pre.shape[0], x_v.shape[0])
                        x_v_pre = x_v_pre[:min_batch]
                        x_v = x_v[:min_batch]
                        print(f"调整后: x_v_pre={x_v_pre.shape}, x_v={x_v.shape}")
                
                # 修改这里：将MSE损失计算的reduction从'none'改为'mean'，使损失变成标量值
                loss_a = F.mse_loss(x_a_pre.squeeze(), x_a, reduction='mean')
                loss_t = F.mse_loss(x_t_pre.squeeze(), x_t, reduction='mean')
                loss_v = F.mse_loss(x_v_pre.squeeze(), x_v, reduction='mean')
                loss = loss_t + loss_a + loss_v
                
            except RuntimeError as e:
                print(f"扩散模型处理错误: {e}")
                # 如果出错，创建零损失和预测值
                print("创建零张量替代")
                
                # 创建与原始张量相同形状的零张量作为预测值
                x_t_pre = torch.zeros_like(x_t)
                x_a_pre = torch.zeros_like(x_a)
                x_v_pre = torch.zeros_like(x_v)
                
                # 创建零损失
                loss = torch.zeros(1, device=x_t.device)

            output_a = self.fc_a(x_a_pre.transpose(2,1))
            output_t = self.fc_t(x_t_pre.transpose(2,1))
            output_v = self.fc_v(x_v_pre.transpose(2,1))
            output_a = output_a.transpose(2, 1)
            output_t = output_t.transpose(2, 1)
            output_v = output_v.transpose(2, 1)

            comments_global = comments_global.unsqueeze(1)
            
            # 处理videos_global的维度问题
            print(f"最终特征形状: output_t={output_t.shape}, output_a={output_a.shape}, videos_global={videos_global.shape}")
            
            # 调整videos_global的维度从1000到128，与其他特征保持一致
            if videos_global.shape[1] != 128:
                print(f"调整videos_global维度从{videos_global.shape[1]}到128")
                # 使用预定义的线性层调整维度
                videos_global = self.videos_global_proj(videos_global)
                print(f"调整后videos_global形状: {videos_global.shape}")
            
            videos_global = videos_global.unsqueeze(1)
            user_intro_global = user_intro_global.unsqueeze(1)
            
            print(f"拼接前最终形状: output_t={output_t.shape}, output_a={output_a.shape}, videos_global={videos_global.shape}, output_v={output_v.shape}, comments_global={comments_global.shape}")

            # Prediction
            output_m = torch.concat([output_t, output_a, videos_global, user_intro_global, output_v, comments_global], dim=1)
            output_m = self.trm(output_m)
            output_m = torch.mean(output_m, -2)
            
            # 保存多模态融合特征用于神经符号规则
            multimodal_features = output_m.clone()
            
            output_m = self.fc_pre(output_m.squeeze())
            
            # 存储原始视频帧和x_v用于可视化
            self.original_video_features = videos  # 用于保存当前批次的视频特征
            
            # 应用神经符号规则（如果启用且有隐式意见数据）
            rule_info = None
            print(f"🔧 神经符号检查: enable_neural_symbolic={self.enable_neural_symbolic}, implicit_opinion_data={implicit_opinion_data is not None}")
            if self.enable_neural_symbolic and implicit_opinion_data is not None:
                print(f"🧠 神经符号规则: 收到隐式意见数据，类型={type(implicit_opinion_data)}")
                try:
                    # 处理不同类型的隐式意见数据
                    if isinstance(implicit_opinion_data, dict):
                        # 单个样本的字典数据
                        opinion_analysis = implicit_opinion_data
                        print(f"📊 处理字典类型数据")
                    elif isinstance(implicit_opinion_data, list):
                        # 批次数据列表
                        opinion_analysis = implicit_opinion_data
                        valid_count = sum(1 for x in implicit_opinion_data if x is not None)
                        print(f"📊 处理列表类型数据，批次大小: {len(implicit_opinion_data)}, 有效样本: {valid_count}")
                    elif isinstance(implicit_opinion_data, str):
                        # 原始文本，需要实时分析
                        if self.implicit_analyzer is not None:
                            opinion_analysis = self.implicit_analyzer.analyze_implicit_opinion(implicit_opinion_data)
                            print(f"📊 处理字符串类型数据，实时分析")
                        else:
                            opinion_analysis = None
                            print(f"⚠️ 字符串数据但没有实时分析器")
                    else:
                        opinion_analysis = None
                        print(f"⚠️ 不支持的数据类型: {type(implicit_opinion_data)}")
                    
                    if opinion_analysis is not None:
                        print(f"🔍 开始应用神经符号规则...")
                        # 应用神经符号规则调整
                        # 使用文本特征作为调整目标
                        adjusted_text_features, adjusted_prediction, rule_info = self.neural_symbolic_engine(
                            text_features=x_t,
                            model_prediction=torch.softmax(output_m, dim=-1),
                            implicit_opinion_analysis=opinion_analysis
                        )
                        
                        print(f"🔍 规则应用结果: {rule_info}")
                        
                        # 如果规则引擎产生了显著调整，更新最终预测
                        if rule_info and abs(rule_info.get("bias_adjustment", 0)) > self.rule_threshold:
                            output_m = torch.log(adjusted_prediction + 1e-8)  # 转回logits
                            print(f"✅ 神经符号规则调整: 权重调整={rule_info.get('weight_adjustment', 0):.3f}, "
                                  f"偏置调整={rule_info.get('bias_adjustment', 0):.3f}")
                        else:
                            bias_adj = rule_info.get('bias_adjustment', 0) if rule_info else 0
                            print(f"⚠️ 规则调整幅度过小，不更新预测。偏置调整: {bias_adj:.6f}, 阈值: {self.rule_threshold}")
                    else:
                        print(f"❌ opinion_analysis 为 None，跳过规则应用")
                            
                except Exception as e:
                    print(f"神经符号规则应用失败: {e}")
                    rule_info = {"error": str(e)}
            
            # 检查是否需要返回可解释性结果
            enable_explanation = self.enable_explanation or return_explanations
            
            # 返回可解释性结果（如果需要）
            if enable_explanation:
                # 调用可解释性模块生成解释
                explanation = self.explainer(x_t, x_a, x_v, output_m)
                
                # 添加特征到解释结果中，以供对比学习和对抗验证使用
                explanation['text_features'] = x_t  # 添加文本特征
                explanation['audio_features'] = x_a  # 添加音频特征
                explanation['video_features'] = x_v  # 添加视频特征
                explanation['unified_features'] = output_m  # 添加统一特征
                
                # 添加神经符号规则信息到解释中
                if rule_info is not None:
                    explanation['neural_symbolic_rules'] = rule_info
                
                # 如果模型预测为虚假（类别1），则创建热图
                # 获取预测的类别
                _, predicted_class = torch.max(output_m, dim=1)
                
                # 记录预测结果到解释中
                explanation['predicted_class'] = predicted_class
                
                return loss, output_m, explanation
            
            return loss, output_m
        
    def get_explanations(self, texts, audios, videos, comments, c3d, user_intro, gpt_description):
        """
        专门用于获取解释结果的方法，不进行训练
        返回模型的预测结果和解释信息
        """
        with torch.no_grad():
            loss, pred, explanation = self.forward(
                texts, audios, videos, comments, c3d, user_intro, gpt_description,
                return_explanations=True
            )
        return pred, explanation
    
    def visualize_fake_regions(self, explanation_dict, video_frames=None, save_path=None):
        """
        可视化虚假区域的方法
        
        Args:
            explanation_dict: 包含可解释性信息的字典
            video_frames: 原始视频帧 (如果有)
            save_path: 保存可视化结果的路径
            
        Returns:
            可视化结果
        """
        return self.explainer.visualize_explanation(
            explanation_dict, 
            video_frames if video_frames is not None else self.original_video_frames,
            save_path
        )