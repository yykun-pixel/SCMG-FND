import torch
import sys
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_a, orig_d_v, MULT_d, mult_dropout):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = orig_d_l, orig_d_a, orig_d_v
        self.d_l, self.d_a, self.d_v = MULT_d, MULT_d, MULT_d
        self.num_heads = 2
        self.layers = 5
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = mult_dropout
        self.embed_dropout = 0.25
        self.attn_mask = True

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_g = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_a = self.get_network(self_type='va')

        self.trans_g_with_l = self.get_network(self_type='l')
        self.trans_g_with_a = self.get_network(self_type='la')
        self.trans_g_with_v = self.get_network(self_type='lv')

        self.trans_l_with_g = self.get_network(self_type='l')
        self.trans_a_with_g = self.get_network(self_type='la')
        self.trans_v_with_g = self.get_network(self_type='lv')
       
        '''# Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)'''

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  position_emb = True)
            
    def forward(self, x_l, x_g, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # 打印输入张量的形状以便调试
        print(f"CrossmodalTransformer输入形状: x_l={x_l.shape}, x_g={x_g.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
        
        # 确保张量至少是3D的 [batch_size, seq_len, features]
        if len(x_l.shape) < 3:
            print(f"x_l维度不足，添加seq_len维度: {x_l.shape}")
            x_l = x_l.unsqueeze(1)  # 添加seq_len维度
            print(f"调整后: {x_l.shape}")
        
        if len(x_g.shape) < 3:
            print(f"x_g维度不足，添加seq_len维度: {x_g.shape}")
            x_g = x_g.unsqueeze(1)  # 添加seq_len维度
            print(f"调整后: {x_g.shape}")
        
        if len(x_a.shape) < 3:
            print(f"x_a维度不足，添加seq_len维度: {x_a.shape}")
            x_a = x_a.unsqueeze(1)  # 添加seq_len维度
            print(f"调整后: {x_a.shape}")
        
        if len(x_v.shape) < 3:
            print(f"x_v维度不足，添加seq_len维度: {x_v.shape}")
            x_v = x_v.unsqueeze(1)  # 添加seq_len维度
            print(f"调整后: {x_v.shape}")
        
        # 调整序列长度为最小1
        if x_l.shape[1] == 0:
            x_l = torch.zeros(x_l.shape[0], 1, x_l.shape[2], device=x_l.device)
        if x_g.shape[1] == 0:
            x_g = torch.zeros(x_g.shape[0], 1, x_g.shape[2], device=x_g.device)
        if x_a.shape[1] == 0:
            x_a = torch.zeros(x_a.shape[0], 1, x_a.shape[2], device=x_a.device)
        if x_v.shape[1] == 0:
            x_v = torch.zeros(x_v.shape[0], 1, x_v.shape[2], device=x_v.device)
        
        try:
            x_l = F.dropout(x_l.transpose(2, 1), p=self.embed_dropout, training=self.training)
            x_g = x_g.transpose(2, 1)
            x_a = x_a.transpose(2, 1)
            
            # 特殊处理视频特征
            print(f"视频特征转置前形状: {x_v.shape}")
            # 确保形状是[batch, features, seq_len]或可以转为这种形状
            if len(x_v.shape) == 3:
                # 检查当前顺序
                batch_size, dim1, dim2 = x_v.shape
                # 如果dim1是seq_len，dim2是features(通常seq_len小于features)
                if dim1 < dim2:
                    # 当前是[batch, seq_len, features]，需要transpose
                    x_v = x_v.transpose(2, 1)
                # 否则可能已经是[batch, features, seq_len]，不需要转置
            
            print(f"视频特征转置后形状: {x_v.shape}")
        except IndexError as e:
            print(f"转置操作失败: {e}")
            print(f"转置前的形状: x_l={x_l.shape}, x_g={x_g.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
            # 尝试不同的处理方式
            if len(x_l.shape) == 2:
                # 如果是[batch_size, features]，添加一个虚拟的seq_len维度
                x_l = x_l.unsqueeze(1)
                x_l = F.dropout(x_l.transpose(2, 1), p=self.embed_dropout, training=self.training)
            else:
                # 如果无法转置，使用原始张量并跳过转置
                x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
            
            # 对其他张量也做类似处理
            if len(x_g.shape) == 2:
                x_g = x_g.unsqueeze(1).transpose(2, 1)
            if len(x_a.shape) == 2:
                x_a = x_a.unsqueeze(1).transpose(2, 1)
            if len(x_v.shape) == 2:
                # 视频特征需要特殊处理
                print(f"异常处理：视频特征为2D张量: {x_v.shape}")
                x_v = x_v.unsqueeze(1).transpose(2, 1)
                print(f"处理后: {x_v.shape}")
            elif len(x_v.shape) == 3:
                # 检查视频特征的当前维度顺序
                batch_size, dim1, dim2 = x_v.shape
                print(f"异常处理：视频特征为3D张量: {x_v.shape}")
                # 根据维度大小判断可能的顺序
                if dim1 < dim2:
                    # 如果第一维小于第二维，可能是[batch, seq_len, features]
                    # 需要转为[batch, features, seq_len]
                    x_v = x_v.transpose(2, 1)
                    print(f"转置后: {x_v.shape}")
                # 否则保持不变，可能已经是正确的顺序
            else:
                # 如果维度异常，创建一个空张量
                print(f"异常处理：视频特征维度异常: {x_v.shape}")
                batch_size = x_v.shape[0] if len(x_v.shape) > 0 else 16
                # 创建一个符合预期的张量[batch, features, seq_len]
                x_v = torch.zeros(batch_size, self.orig_d_v, 1, device=x_l.device)
                print(f"创建新张量: {x_v.shape}")
        
        print(f"转置后的形状: x_l={x_l.shape}, x_g={x_g.shape}, x_a={x_a.shape}, x_v={x_v.shape}")
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        
        # 检查视频特征形状并进行必要的转换
        print(f"投影前视频特征形状: x_v={x_v.shape}, 期望通道数: {self.orig_d_v}")
        if x_v.shape[1] != self.orig_d_v:
            print(f"视频特征通道数不匹配: {x_v.shape[1]} vs 期望的 {self.orig_d_v}")
            # 转换视频特征以匹配预期的输入通道
            if len(x_v.shape) == 3:  # [batch, channels, seq_len]
                # 尝试不同的方法来调整通道维度
                if x_v.shape[1] < self.orig_d_v:
                    # 方法1: 使用线性插值扩展通道维度
                    print("使用线性插值扩展通道")
                    batch_size, curr_channels, seq_len = x_v.shape
                    # 先创建一个目标大小的张量
                    target_x_v = torch.zeros(batch_size, self.orig_d_v, seq_len, device=x_v.device)
                    # 将当前特征映射到目标特征的子集
                    target_x_v[:, :curr_channels, :] = x_v
                    x_v = target_x_v
                    print(f"调整后视频特征形状: {x_v.shape}")
                elif x_v.shape[1] > self.orig_d_v:
                    # 方法2: 截断多余的通道
                    print("截断多余的通道")
                    x_v = x_v[:, :self.orig_d_v, :]
                    print(f"截断后视频特征形状: {x_v.shape}")
        
        # 针对视频特征，将序列维度通过平均池化压缩为1，与其他特征保持一致的形状
        if len(x_v.shape) == 3 and x_v.shape[2] > 1:
            print(f"对视频特征进行平均池化，压缩序列维度: {x_v.shape}")
            x_v = torch.mean(x_v, dim=2, keepdim=True)
            print(f"池化后视频特征形状: {x_v.shape}")
        
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_g = x_g if self.orig_d_l == self.d_l else self.proj_g(x_g)
        
        print(f"投影后特征形状: proj_x_l={proj_x_l.shape}, proj_x_a={proj_x_a.shape}, proj_x_v={proj_x_v.shape}, proj_x_g={proj_x_g.shape}")
        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_g = proj_x_g.permute(2, 0, 1)
        
        print(f"permute后特征形状: proj_x_l={proj_x_l.shape}, proj_x_a={proj_x_a.shape}, proj_x_v={proj_x_v.shape}, proj_x_g={proj_x_g.shape}")

        # (A,V,G) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_l_with_gs = self.trans_l_with_g(proj_x_l, proj_x_g, proj_x_g)
        
        print(f"注意力机制处理后形状: h_l_with_as={h_l_with_as.shape}, h_l_with_vs={h_l_with_vs.shape}, h_l_with_gs={h_l_with_gs.shape}")
        
        # 确保维度一致后再拼接
        target_size = h_l_with_as.shape  # 使用第一个特征的形状作为目标形状
        
        # 如果形状不匹配，调整h_l_with_vs的形状
        if h_l_with_vs.shape != target_size:
            print(f"调整h_l_with_vs形状从{h_l_with_vs.shape}到{target_size}")
            h_l_with_vs = F.adaptive_avg_pool1d(h_l_with_vs.transpose(1, 2), target_size[0]).transpose(1, 2)
            # 或直接复制h_l_with_as到h_l_with_vs
            # h_l_with_vs = h_l_with_as.clone()
        
        # 如果形状不匹配，调整h_l_with_gs的形状
        if h_l_with_gs.shape != target_size:
            print(f"调整h_l_with_gs形状从{h_l_with_gs.shape}到{target_size}")
            h_l_with_gs = F.adaptive_avg_pool1d(h_l_with_gs.transpose(1, 2), target_size[0]).transpose(1, 2)
            # 或直接复制
            # h_l_with_gs = h_l_with_as.clone()
            
        print(f"调整后形状: h_l_with_as={h_l_with_as.shape}, h_l_with_vs={h_l_with_vs.shape}, h_l_with_gs={h_l_with_gs.shape}")
        
        h_ls = F.dropout(torch.cat([h_l_with_as, h_l_with_vs, h_l_with_gs], dim=2), p=self.out_dropout, training=self.training)
        # (L,V,G) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_a_with_gs = self.trans_a_with_g(proj_x_a, proj_x_g, proj_x_g)
        
        print(f"音频注意力机制处理后形状: h_a_with_ls={h_a_with_ls.shape}, h_a_with_vs={h_a_with_vs.shape}, h_a_with_gs={h_a_with_gs.shape}")
        
        # 确保维度一致后再拼接
        target_size = h_a_with_ls.shape  # 使用第一个特征的形状作为目标形状
        
        # 调整h_a_with_vs的形状
        if h_a_with_vs.shape != target_size:
            print(f"调整h_a_with_vs形状从{h_a_with_vs.shape}到{target_size}")
            h_a_with_vs = F.adaptive_avg_pool1d(h_a_with_vs.transpose(1, 2), target_size[0]).transpose(1, 2)
        
        # 调整h_a_with_gs的形状
        if h_a_with_gs.shape != target_size:
            print(f"调整h_a_with_gs形状从{h_a_with_gs.shape}到{target_size}")
            h_a_with_gs = F.adaptive_avg_pool1d(h_a_with_gs.transpose(1, 2), target_size[0]).transpose(1, 2)
            
        print(f"调整后形状: h_a_with_ls={h_a_with_ls.shape}, h_a_with_vs={h_a_with_vs.shape}, h_a_with_gs={h_a_with_gs.shape}")
        
        h_as = F.dropout(torch.cat([h_a_with_ls, h_a_with_vs, h_a_with_gs], dim=2), p=self.out_dropout, training=self.training)
        # (L,A,G) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_v_with_gs = self.trans_v_with_g(proj_x_v, proj_x_g, proj_x_g)
        
        print(f"视频注意力机制处理后形状: h_v_with_ls={h_v_with_ls.shape}, h_v_with_as={h_v_with_as.shape}, h_v_with_gs={h_v_with_gs.shape}")
        
        # 确保维度一致后再拼接
        target_size = h_v_with_ls.shape  # 使用第一个特征的形状作为目标形状
        
        # 特别处理小批次情况，确保维度匹配
        if h_v_with_ls.shape[1] == 1 and (h_v_with_as.shape != target_size or h_v_with_gs.shape != target_size):
            print(f"小批次特殊处理：批次大小为1时进行形状调整")
            try:
                # 尝试使用克隆复制形状
                if h_v_with_as.shape != target_size:
                    h_v_with_as = h_v_with_ls.clone()
                if h_v_with_gs.shape != target_size:
                    h_v_with_gs = h_v_with_ls.clone()
                print(f"使用克隆替换不匹配的张量，调整后: h_v_with_as={h_v_with_as.shape}, h_v_with_gs={h_v_with_gs.shape}")
            except Exception as e:
                print(f"克隆替换失败: {e}")
        
        # 调整h_v_with_as的形状
        if h_v_with_as.shape != target_size:
            print(f"调整h_v_with_as形状从{h_v_with_as.shape}到{target_size}")
            h_v_with_as = F.adaptive_avg_pool1d(h_v_with_as.transpose(1, 2), target_size[0]).transpose(1, 2)
        
        # 调整h_v_with_gs的形状
        if h_v_with_gs.shape != target_size:
            print(f"调整h_v_with_gs形状从{h_v_with_gs.shape}到{target_size}")
            h_v_with_gs = F.adaptive_avg_pool1d(h_v_with_gs.transpose(1, 2), target_size[0]).transpose(1, 2)
            
        print(f"调整后形状: h_v_with_ls={h_v_with_ls.shape}, h_v_with_as={h_v_with_as.shape}, h_v_with_gs={h_v_with_gs.shape}")
        
        h_vs = F.dropout(torch.cat([h_v_with_ls, h_v_with_as, h_v_with_gs], dim=2), p=self.out_dropout, training=self.training)

        # (L,A,V) --> G
        h_g_with_ls = self.trans_g_with_l(proj_x_g, proj_x_l, proj_x_l)
        h_g_with_as = self.trans_g_with_a(proj_x_g, proj_x_a, proj_x_a)
        h_g_with_vs = self.trans_g_with_v(proj_x_g, proj_x_g, proj_x_g)
        
        print(f"GPT注意力机制处理后形状: h_g_with_ls={h_g_with_ls.shape}, h_g_with_as={h_g_with_as.shape}, h_g_with_vs={h_g_with_vs.shape}")
        
        # 确保维度一致后再拼接
        target_size = h_g_with_ls.shape  # 使用第一个特征的形状作为目标形状
        
        # 特别处理小批次情况
        if h_g_with_ls.shape[1] == 1 and (h_g_with_as.shape != target_size or h_g_with_vs.shape != target_size):
            print(f"小批次特殊处理：批次大小为1时进行形状调整")
            try:
                # 尝试使用克隆复制形状
                if h_g_with_as.shape != target_size:
                    h_g_with_as = h_g_with_ls.clone()
                if h_g_with_vs.shape != target_size:
                    h_g_with_vs = h_g_with_ls.clone()
                print(f"使用克隆替换不匹配的张量，调整后: h_g_with_as={h_g_with_as.shape}, h_g_with_vs={h_g_with_vs.shape}")
            except Exception as e:
                print(f"克隆替换失败: {e}")
                
        # 调整h_g_with_as的形状
        if h_g_with_as.shape != target_size:
            print(f"调整h_g_with_as形状从{h_g_with_as.shape}到{target_size}")
            h_g_with_as = F.adaptive_avg_pool1d(h_g_with_as.transpose(1, 2), target_size[0]).transpose(1, 2)
        
        # 调整h_g_with_vs的形状
        if h_g_with_vs.shape != target_size:
            print(f"调整h_g_with_vs形状从{h_g_with_vs.shape}到{target_size}")
            h_g_with_vs = F.adaptive_avg_pool1d(h_g_with_vs.transpose(1, 2), target_size[0]).transpose(1, 2)
            
        print(f"调整后形状: h_g_with_ls={h_g_with_ls.shape}, h_g_with_as={h_g_with_as.shape}, h_g_with_vs={h_g_with_vs.shape}")
        
        h_gs = F.dropout(torch.cat([h_g_with_ls, h_g_with_as, h_g_with_vs], dim=2), p=self.out_dropout, training=self.training)

        return h_ls, h_gs, h_as, h_vs

