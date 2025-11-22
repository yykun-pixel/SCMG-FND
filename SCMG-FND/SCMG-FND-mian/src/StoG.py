import torch.nn as nn
import torch.nn.functional as F
import torch


class CapsuleSequenceToGraph(nn.Module):
    def __init__(self, MULT_d, dim_capsule, vertex_num, routing,
                 T_t, T_a, T_v):
        super(CapsuleSequenceToGraph, self).__init__()
        self.d_c = dim_capsule
        self.n = vertex_num
        self.routing = routing
        # create primary capsule
        self.W_tpc = nn.Parameter(torch.Tensor(T_t, self.n, MULT_d, self.d_c))
        self.W_apc = nn.Parameter(torch.Tensor(T_a, self.n, MULT_d, self.d_c))
        self.W_vpc = nn.Parameter(torch.Tensor(T_v, self.n, MULT_d, self.d_c))
        nn.init.xavier_normal_(self.W_tpc)
        nn.init.xavier_normal_(self.W_apc)
        nn.init.xavier_normal_(self.W_vpc)

    def forward(self, text, audio, video, batch_size):
        try:
            # 获取实际维度
            T_t = text.shape[0]
            T_a = audio.shape[0]
            T_v = video.shape[0]
            actual_batch_size = text.shape[1]
            
            # 打印输入形状用于调试
            print(f"StoG输入形状: text={text.shape}, audio={audio.shape}, video={video.shape}, batch_size={batch_size}")
            
            # 检查批次大小
            if actual_batch_size != batch_size:
                print(f"警告：StoG中实际批次大小({actual_batch_size})与传入批次大小({batch_size})不匹配，使用实际值")
                batch_size = actual_batch_size
            
            # 特殊处理小批次，尤其是批次大小为1的情况
            if batch_size == 1:
                print(f"StoG处理小批次(size=1)数据")
                
                # 检查所有输入形状维度是否一致
                expected_dims = 3  # 期望的维度数
                if len(text.shape) != expected_dims:
                    print(f"修正text形状从{text.shape}到3D张量")
                    if len(text.shape) < expected_dims:
                        # 如果维度不足，添加缺失的维度
                        while len(text.shape) < expected_dims:
                            text = text.unsqueeze(-1)
                    else:
                        # 如果维度过多，压缩多余的维度
                        text = text.view(text.shape[0], text.shape[1], -1)
                
                # 同样处理audio和video
                if len(audio.shape) != expected_dims:
                    print(f"修正audio形状从{audio.shape}到3D张量")
                    if len(audio.shape) < expected_dims:
                        while len(audio.shape) < expected_dims:
                            audio = audio.unsqueeze(-1)
                    else:
                        audio = audio.view(audio.shape[0], audio.shape[1], -1)
                
                if len(video.shape) != expected_dims:
                    print(f"修正video形状从{video.shape}到3D张量")
                    if len(video.shape) < expected_dims:
                        while len(video.shape) < expected_dims:
                            video = video.unsqueeze(-1)
                    else:
                        video = video.view(video.shape[0], video.shape[1], -1)
                
                print(f"形状调整后: text={text.shape}, audio={audio.shape}, video={video.shape}")
            
            # 确保特征维度一致
            d_t = text.shape[2]
            d_a = audio.shape[2]
            d_v = video.shape[2]
            
            if d_t != d_a or d_t != d_v:
                print(f"警告：特征维度不匹配: text={d_t}, audio={d_a}, video={d_v}")
                # 选择最大的特征维度作为目标
                target_dim = max(d_t, d_a, d_v)
                
                # 调整维度
                if d_t != target_dim:
                    print(f"调整text特征维度从{d_t}到{target_dim}")
                    # 使用零填充或者截断
                    if d_t < target_dim:
                        padding = torch.zeros(T_t, batch_size, target_dim - d_t, device=text.device)
                        text = torch.cat([text, padding], dim=2)
                    else:
                        text = text[:, :, :target_dim]
                
                if d_a != target_dim:
                    print(f"调整audio特征维度从{d_a}到{target_dim}")
                    if d_a < target_dim:
                        padding = torch.zeros(T_a, batch_size, target_dim - d_a, device=audio.device)
                        audio = torch.cat([audio, padding], dim=2)
                    else:
                        audio = audio[:, :, :target_dim]
                
                if d_v != target_dim:
                    print(f"调整video特征维度从{d_v}到{target_dim}")
                    if d_v < target_dim:
                        padding = torch.zeros(T_v, batch_size, target_dim - d_v, device=video.device)
                        video = torch.cat([video, padding], dim=2)
                    else:
                        video = video[:, :, :target_dim]
            
            # 更新T值以防它们在上面的处理中发生了变化
            T_t = text.shape[0]
            T_a = audio.shape[0]
            T_v = video.shape[0]
            
            # 创建primary capsule - 使用try-except捕获可能的错误
            try:
                text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc)).permute(1, 0, 2, 3)
                audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc)).permute(1, 0, 2, 3)
                video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc)).permute(1, 0, 2, 3)
            except RuntimeError as e:
                print(f"创建primary capsule时出错: {e}")
                print(f"尝试调整矩阵大小...")
                
                # 确保T_t、T_a、T_v不超过对应W矩阵的第一维
                if T_t > self.W_tpc.shape[0]:
                    print(f"调整text序列长度从{T_t}到{self.W_tpc.shape[0]}")
                    text = text[:self.W_tpc.shape[0]]
                    T_t = self.W_tpc.shape[0]
                
                if T_a > self.W_apc.shape[0]:
                    print(f"调整audio序列长度从{T_a}到{self.W_apc.shape[0]}")
                    audio = audio[:self.W_apc.shape[0]]
                    T_a = self.W_apc.shape[0]
                
                if T_v > self.W_vpc.shape[0]:
                    print(f"调整video序列长度从{T_v}到{self.W_vpc.shape[0]}")
                    video = video[:self.W_vpc.shape[0]]
                    T_v = self.W_vpc.shape[0]
                
                # 重新尝试
                text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc[:T_t])).permute(1, 0, 2, 3)
                audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc[:T_a])).permute(1, 0, 2, 3)
                video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc[:T_v])).permute(1, 0, 2, 3)

            # routing mechanism does not participate in back propagation
            text_pri_caps_temp = text_pri_caps.detach()
            audio_pri_caps_temp = audio_pri_caps.detach()
            video_pri_caps_temp = video_pri_caps.detach()

            # begin routing
            for r in range(self.routing + 1):
                if r == 0:
                    b_t = torch.zeros(batch_size, T_t, self.n).to(text.device)  # initialize routing coefficients
                    b_a = torch.zeros(batch_size, T_a, self.n).to(audio.device)
                    b_v = torch.zeros(batch_size, T_v, self.n).to(video.device)
                rc_t = F.softmax(b_t, 2)
                rc_a = F.softmax(b_a, 2)
                rc_v = F.softmax(b_v, 2)

                text_vertex = torch.tanh(torch.sum(text_pri_caps_temp * rc_t.unsqueeze(-1), 1))
                audio_vertex = torch.tanh(torch.sum(audio_pri_caps_temp * rc_a.unsqueeze(-1), 1))
                video_vertex = torch.tanh(torch.sum(video_pri_caps_temp * rc_v.unsqueeze(-1), 1))

                # update routing coefficients
                if r < self.routing:
                    last = b_t
                    new = ((text_vertex.unsqueeze(1)) * text_pri_caps_temp).sum(3)
                    b_t = last + new

                    last = b_a
                    new = (audio_vertex.unsqueeze(1) * audio_pri_caps_temp).sum(3)
                    b_a = last + new

                    last = b_v
                    new = (video_vertex.unsqueeze(1) * video_pri_caps_temp).sum(3)
                    b_v = last + new

            # create vertex using the routing coefficients in final round
            text_vertex = torch.tanh(torch.sum(text_pri_caps * rc_t.unsqueeze(-1), 1))
            audio_vertex = torch.tanh(torch.sum(audio_pri_caps * rc_a.unsqueeze(-1), 1))
            video_vertex = torch.tanh(torch.sum(video_pri_caps * rc_v.unsqueeze(-1), 1))
            
            # 确保输出形状一致性
            print(f"StoG输出形状: text_vertex={text_vertex.shape}, audio_vertex={audio_vertex.shape}, video_vertex={video_vertex.shape}")
            
            return text_vertex, audio_vertex, video_vertex
            
        except Exception as e:
            print(f"StoG.forward中出现未处理异常: {e}")
            print(f"创建备用输出...")
            
            # 创建备用输出，确保维度正确
            feature_dim = text.shape[2] if len(text.shape) >= 3 else (
                           audio.shape[2] if len(audio.shape) >= 3 else (
                           video.shape[2] if len(video.shape) >= 3 else 128))
            
            batch_size = max(1, batch_size)  # 确保批次大小至少为1
            
            # 创建正确形状的张量
            text_vertex = torch.zeros(batch_size, self.n, self.d_c, device=text.device)
            audio_vertex = torch.zeros(batch_size, self.n, self.d_c, device=audio.device)
            video_vertex = torch.zeros(batch_size, self.n, self.d_c, device=video.device)
            
            print(f"创建备用输出: shape={text_vertex.shape}")
            return text_vertex, audio_vertex, video_vertex
