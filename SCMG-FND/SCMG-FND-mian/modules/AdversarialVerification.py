import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class AdversarialVerification(nn.Module):
    """
    对抗性验证框架
    
    通过生成器和判别器的对抗训练，增强模型对假视频的鉴别能力
    """
    def __init__(self, 
                 feature_dim=128,
                 hidden_dim=256,
                 z_dim=64,
                 num_layers=2,
                 dropout=0.3):
        """
        初始化对抗性验证框架
        
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            z_dim: 隐藏变量维度
            num_layers: 网络层数
            dropout: Dropout概率
        """
        super(AdversarialVerification, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # 特征编码器（将原始特征编码到隐空间）
        encoder_layers = []
        encoder_layers.append(nn.Linear(feature_dim, hidden_dim))
        encoder_layers.append(nn.LeakyReLU(0.2))
        encoder_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(dropout))
            
        encoder_layers.append(nn.Linear(hidden_dim, z_dim * 2))  # 均值和方差
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 特征生成器（从隐空间生成伪造特征）
        generator_layers = []
        generator_layers.append(nn.Linear(z_dim, hidden_dim))
        generator_layers.append(nn.LeakyReLU(0.2))
        generator_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            generator_layers.append(nn.Linear(hidden_dim, hidden_dim))
            generator_layers.append(nn.LeakyReLU(0.2))
            generator_layers.append(nn.Dropout(dropout))
            
        generator_layers.append(nn.Linear(hidden_dim, feature_dim))
        generator_layers.append(nn.Tanh())  # 特征归一化
        
        self.generator = nn.Sequential(*generator_layers)
        
        # 特征判别器（区分真实特征和生成特征）
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(feature_dim, hidden_dim))
        discriminator_layers.append(nn.LeakyReLU(0.2))
        discriminator_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            discriminator_layers.append(nn.Linear(hidden_dim, hidden_dim))
            discriminator_layers.append(nn.LeakyReLU(0.2))
            discriminator_layers.append(nn.Dropout(dropout))
            
        discriminator_layers.append(nn.Linear(hidden_dim, 1))
        
        self.discriminator = nn.Sequential(*discriminator_layers)
        
    def reparameterize(self, mu, log_var):
        """
        重参数化技巧
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            采样得到的隐变量z
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """
        编码特征到隐空间
        
        Args:
            x: 输入特征
            
        Returns:
            mu: 均值
            log_var: 对数方差
        """
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var
    
    def generate(self, z=None, batch_size=None):
        """
        从隐空间生成特征
        
        Args:
            z: 隐变量，如果为None则随机采样
            batch_size: 批大小，当z为None时使用
            
        Returns:
            生成的特征
        """
        if z is None:
            assert batch_size is not None, "batch_size must be provided when z is None"
            z = torch.randn(batch_size, self.z_dim, device=self.discriminator[0].weight.device)
        
        return self.generator(z)
    
    def discriminate(self, x):
        """
        判别特征真假
        
        Args:
            x: 输入特征
            
        Returns:
            判别得分
        """
        return self.discriminator(x)
    
    def forward_discriminator(self, real_features, fake_features=None, batch_size=None):
        """
        判别器前向传播
        
        Args:
            real_features: 真实特征
            fake_features: 生成的特征，如果为None则自动生成
            batch_size: 批大小，当fake_features为None时使用
            
        Returns:
            real_scores: 真实特征的判别分数
            fake_scores: 生成特征的判别分数
            fake_features: 生成的特征
        """
        # 判别真实特征
        real_scores = self.discriminate(real_features)
        
        # 生成并判别伪造特征
        if fake_features is None:
            # 自动生成伪造特征
            fake_features = self.generate(batch_size=batch_size or real_features.size(0))
        
        fake_scores = self.discriminate(fake_features.detach())  # 分离计算图，避免更新生成器
        
        return real_scores, fake_scores, fake_features
    
    def forward_generator(self, real_features=None, batch_size=None):
        """
        生成器前向传播
        
        Args:
            real_features: 真实特征，如果不为None则使用VAE模式
            batch_size: 批大小，当real_features为None且不使用VAE模式时使用
            
        Returns:
            gen_scores: 生成特征的判别分数
            gen_features: 生成的特征
            kl_loss: KL散度损失（VAE模式下）
        """
        # VAE模式：从真实特征重构
        if real_features is not None:
            mu, log_var = self.encode(real_features)
            z = self.reparameterize(mu, log_var)
            gen_features = self.generator(z)
            gen_scores = self.discriminate(gen_features)
            
            # 计算KL散度损失
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / real_features.size(0)
            
            return gen_scores, gen_features, kl_loss
        
        # 纯生成模式：从随机噪声生成
        else:
            batch_size = batch_size or real_features.size(0)
            gen_features = self.generate(batch_size=batch_size)
            gen_scores = self.discriminate(gen_features)
            
            return gen_scores, gen_features, None
    
    def compute_discriminator_loss(self, real_scores, fake_scores):
        """
        计算判别器损失
        
        Args:
            real_scores: 真实特征的判别分数
            fake_scores: 生成特征的判别分数
            
        Returns:
            判别器损失
        """
        # 真实样本应接近1，生成样本应接近0
        real_loss = F.binary_cross_entropy_with_logits(
            real_scores, 
            torch.ones_like(real_scores)
        )
        
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_scores,
            torch.zeros_like(fake_scores)
        )
        
        # 总判别器损失
        d_loss = real_loss + fake_loss
        
        return d_loss
    
    def compute_generator_loss(self, gen_scores, real_features=None, gen_features=None, kl_loss=None, kl_weight=0.1):
        """
        计算生成器损失
        
        Args:
            gen_scores: 生成特征的判别分数
            real_features: 真实特征
            gen_features: 生成的特征
            kl_loss: KL散度损失（VAE模式下）
            kl_weight: KL散度损失权重
            
        Returns:
            生成器损失
        """
        # 对抗损失：生成样本应接近1（欺骗判别器）
        g_loss = F.binary_cross_entropy_with_logits(
            gen_scores,
            torch.ones_like(gen_scores)
        )
        
        # 如果使用VAE模式，添加重构损失和KL散度损失
        if real_features is not None and gen_features is not None:
            # 重构损失
            recon_loss = F.mse_loss(gen_features, real_features)
            
            # 添加KL散度损失
            if kl_loss is not None:
                g_loss = g_loss + recon_loss + kl_weight * kl_loss
            else:
                g_loss = g_loss + recon_loss
        
        return g_loss
    
    def forward(self, real_features, mode='both'):
        """
        前向传播
        
        Args:
            real_features: 真实特征
            mode: 运行模式，'discriminator'、'generator'或'both'
            
        Returns:
            d_loss: 判别器损失
            g_loss: 生成器损失
            fake_features: 生成的特征
        """
        batch_size = real_features.size(0)
        
        if mode == 'discriminator' or mode == 'both':
            # 判别器前向传播
            real_scores, fake_scores, fake_features = self.forward_discriminator(real_features)
            d_loss = self.compute_discriminator_loss(real_scores, fake_scores)
        else:
            d_loss = None
            fake_features = None
        
        if mode == 'generator' or mode == 'both':
            # 生成器前向传播
            gen_scores, gen_features, kl_loss = self.forward_generator(real_features)
            g_loss = self.compute_generator_loss(gen_scores, real_features, gen_features, kl_loss)
            
            if fake_features is None:
                fake_features = gen_features
        else:
            g_loss = None
        
        return d_loss, g_loss, fake_features

    def perturb_features(self, features):
        """仅生成扰动特征，用于测试阶段
        
        Args:
            features: 输入特征
            
        Returns:
            扰动后的特征
        """
        with torch.no_grad():
            batch_size = features.size(0)
            device = features.device
            
            # 生成随机噪声
            noise = torch.randn(batch_size, self.feature_dim, device=device) * self.noise_scale
            
            # 生成扰动
            perturbation = self.perturbator(features + noise)
            
            # 扰动后的特征
            perturbed_features = features + perturbation * self.noise_scale
            
        return perturbed_features
        
    def discriminator_step(self, optimizer):
        """执行判别器的优化步骤
        
        Args:
            optimizer: 判别器的优化器
            
        Returns:
            None
        """
        # 仅更新判别器参数
        for param in self.perturbator.parameters():
            param.requires_grad = False
            
        for param in self.discriminator.parameters():
            param.requires_grad = True
            
        optimizer.step()
        
    def generator_step(self, optimizer):
        """执行生成器的优化步骤
        
        Args:
            optimizer: 生成器的优化器
            
        Returns:
            None
        """
        # 仅更新生成器参数
        for param in self.perturbator.parameters():
            param.requires_grad = True
            
        for param in self.discriminator.parameters():
            param.requires_grad = False
            
        optimizer.step()
        
    def train_mode(self):
        """设置为训练模式"""
        self.perturbator.train()
        self.discriminator.train()
        
    def eval_mode(self):
        """设置为评估模式"""
        self.perturbator.eval()
        self.discriminator.eval()

    def compute_gradient_penalty(self, real_features, fake_features):
        """
        计算梯度惩罚，用于WGAN-GP风格的训练
        
        Args:
            real_features: 真实样本特征 [batch_size, feature_dim]
            fake_features: 生成样本特征 [batch_size, feature_dim]
            
        Returns:
            gradient_penalty: 梯度惩罚项
        """
        batch_size = real_features.size(0)
        device = real_features.device
        
        # 在真实和生成样本之间采样随机插值点
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = alpha * real_features + ((1 - alpha) * fake_features)
        interpolates.requires_grad_(True)
        
        # 计算判别器在插值点的输出
        disc_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                               grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # 计算梯度的范数
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # 计算梯度惩罚
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def adversarial_attack(self, features, target_label, model, steps=3, step_size=0.01):
        """
        执行多步对抗攻击
        
        Args:
            features: 输入特征 [batch_size, feature_dim]
            target_label: 目标标签（攻击目标） [batch_size]
            model: 目标模型
            steps: 攻击步数
            step_size: 每步攻击强度
            
        Returns:
            perturbed_features: 对抗性扰动后的特征
        """
        perturbed_features = features.clone().detach()
        perturbed_features.requires_grad = True
        
        for _ in range(steps):
            # 前向传播
            outputs = model(perturbed_features)
            
            # 计算损失（针对目标标签的损失）
            loss = F.cross_entropy(outputs, target_label)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, perturbed_features,
                                      retain_graph=False, create_graph=False)[0]
            
            # 更新扰动特征
            perturbed_features = perturbed_features.detach() + step_size * grad.sign()
            
            # 控制扰动范围
            delta = perturbed_features - features
            delta = torch.clamp(delta, -self.noise_scale, self.noise_scale)
            perturbed_features = features + delta
            perturbed_features.requires_grad = True
        
        return perturbed_features.detach()
    
    def verify_robustness(self, features, labels, model, threshold=0.8):
        """
        验证模型对对抗样本的鲁棒性
        
        Args:
            features: 输入特征 [batch_size, feature_dim]
            labels: 原始标签 [batch_size]
            model: 目标模型
            threshold: 鲁棒性阈值
            
        Returns:
            robustness_score: 鲁棒性分数
            vulnerable_samples: 脆弱样本的索引
        """
        device = features.device
        batch_size = features.size(0)
        
        # 获取模型在原始样本上的预测
        with torch.no_grad():
            original_outputs = model(features)
            original_preds = torch.argmax(original_outputs, dim=1)
        
        # 生成对抗样本
        # 使用梯度符号法（FGSM）生成对抗样本
        features.requires_grad = True
        outputs = model(features)
        loss = F.cross_entropy(outputs, original_preds)
        loss.backward()
        
        # 生成对抗扰动
        adv_features = features.detach() + self.noise_scale * features.grad.sign()
        features.requires_grad = False
        
        # 获取模型在对抗样本上的预测
        with torch.no_grad():
            adv_outputs = model(adv_features)
            adv_preds = torch.argmax(adv_outputs, dim=1)
        
        # 计算鲁棒性分数：对抗样本上正确预测的比例
        correct_after_attack = (adv_preds == original_preds).float()
        robustness_score = correct_after_attack.mean().item()
        
        # 找出脆弱样本：在对抗攻击后预测发生变化的样本
        vulnerable_mask = (correct_after_attack == 0)
        vulnerable_samples = torch.nonzero(vulnerable_mask).squeeze().tolist()
        
        # 如果只有一个脆弱样本，确保返回的是列表
        if isinstance(vulnerable_samples, int):
            vulnerable_samples = [vulnerable_samples]
        
        return robustness_score, vulnerable_samples

    def forward(self,
               model: nn.Module,
               features: Dict[str, torch.Tensor],
               labels: torch.Tensor,
               target_modality: str = "all",
               verify_only: bool = False) -> Dict[str, Any]:
        """
        前向传播，生成对抗样本并验证模型
        
        Args:
            model: 待验证的模型
            features: 不同模态的特征字典
            labels: 样本标签
            target_modality: 目标扰动的模态，可选值：'text', 'audio', 'video', 'all'
            verify_only: 是否仅进行验证，不生成对抗样本
            
        Returns:
            包含对抗样本、对抗损失和验证结果的字典
        """
        # 确保模型处于评估模式
        model_training = model.training
        model.eval()
        
        # 储存原始特征和结果
        original_features = {}
        for key, value in features.items():
            original_features[key] = value.clone().detach()
        
        # 获取原始预测结果
        with torch.no_grad():
            original_outputs = model(original_features)
            original_preds = torch.argmax(original_outputs, dim=1)
            original_accuracy = (original_preds == labels).float().mean()
        
        results = {
            'original_features': original_features,
            'original_outputs': original_outputs,
            'original_preds': original_preds,
            'original_accuracy': original_accuracy,
            'adversarial_features': None,
            'adversarial_outputs': None,
            'adversarial_preds': None,
            'adversarial_accuracy': None,
            'verification_score': None
        }
        
        if verify_only:
            results['verification_score'] = original_accuracy
            # 恢复模型训练状态
            model.train(model_training)
            return results
        
        # 创建对抗样本
        adversarial_features, d_loss, adv_loss, verify_loss = self.forward(
            model, 
            original_features, 
            labels
        )
        
        # 验证模型在对抗样本上的表现
        with torch.no_grad():
            adversarial_outputs = model(adversarial_features)
            adversarial_preds = torch.argmax(adversarial_outputs, dim=1)
            adversarial_accuracy = (adversarial_preds == labels).float().mean()
        
        # 计算验证分数 - 衡量模型的鲁棒性
        verification_score = adversarial_accuracy / (original_accuracy + 1e-10)
        
        # 更新结果
        results.update({
            'adversarial_features': adversarial_features,
            'adversarial_outputs': adversarial_outputs,
            'adversarial_preds': adversarial_preds,
            'adversarial_accuracy': adversarial_accuracy,
            'verification_score': verification_score
        })
        
        # 恢复模型训练状态
        model.train(model_training)
        
        return results
    
    def _generate_adversarial_examples(self,
                                      model: nn.Module,
                                      features: Dict[str, torch.Tensor],
                                      labels: torch.Tensor,
                                      target_modality: str = "all") -> Dict[str, torch.Tensor]:
        """
        生成对抗样本
        
        Args:
            model: 目标模型
            features: 原始特征
            labels: 原始标签
            target_modality: 目标模态
            
        Returns:
            对抗性特征
        """
        # 复制特征以避免修改原始数据
        adv_features = {}
        for key, value in features.items():
            adv_features[key] = value.clone().detach()
            
            # 对目标模态或所有模态添加扰动
            if target_modality == "all" or key == target_modality:
                # 确保特征需要梯度
                adv_features[key].requires_grad = True
                
                # 如果启用随机开始，添加随机噪声
                if self.random_start:
                    adv_features[key] = adv_features[key] + torch.FloatTensor(
                        adv_features[key].shape).uniform_(-self.eps, self.eps).to(adv_features[key].device)
                    adv_features[key] = torch.clamp(adv_features[key], self.clip_min, self.clip_max)
        
        # 生成攻击
        for _ in range(self.steps):
            # 前向传播
            outputs = model(adv_features)
            
            # 计算损失
            if self.targeted:
                # 目标攻击：最小化与目标标签的距离
                loss = -self.loss_fn(outputs, labels)
            else:
                # 非目标攻击：最大化与原始标签的距离
                loss = self.loss_fn(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新对抗样本
            for key, value in adv_features.items():
                if target_modality == "all" or key == target_modality:
                    if value.grad is not None:
                        # 正则化梯度，获取梯度方向
                        grad_sign = value.grad.sign()
                        
                        # 使用FGSM更新对抗样本
                        value.data = value.data + (self.alpha * grad_sign)
                        
                        # 投影到eps球内
                        delta = torch.clamp(value.data - features[key], -self.eps, self.eps)
                        value.data = features[key] + delta
                        
                        # 裁剪到合法范围
                        value.data = torch.clamp(value.data, self.clip_min, self.clip_max)
                        
                        # 重置梯度
                        value.grad.zero_()
        
        # 移除梯度
        for key, value in adv_features.items():
            if hasattr(value, 'requires_grad'):
                value.requires_grad = False
        
        return adv_features
    
    def verify_robustness(self,
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         target_modality: str = "all",
                         device: torch.device = None) -> Dict[str, float]:
        """
        验证模型在数据集上的鲁棒性
        
        Args:
            model: 待验证的模型
            dataloader: 数据加载器
            target_modality: 目标模态
            device: 计算设备
            
        Returns:
            包含验证结果的字典
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        original_correct = 0
        adversarial_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            # 解包数据
            features, labels = batch
            
            # 移动数据到设备
            labels = labels.to(device)
            for key in features:
                features[key] = features[key].to(device)
            
            # 获取验证结果
            verification_results = self.forward(
                model, 
                features, 
                labels
            )
            
            # 累加结果
            batch_size = labels.size(0)
            original_correct += (verification_results['original_preds'] == labels).sum().item()
            adversarial_correct += (verification_results['adversarial_preds'] == labels).sum().item()
            total_samples += batch_size
        
        # 计算精度
        original_accuracy = original_correct / total_samples if total_samples > 0 else 0
        adversarial_accuracy = adversarial_correct / total_samples if total_samples > 0 else 0
        robustness_score = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0
        
        return {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_score': robustness_score
        }
    
    def generate_boundary_examples(self,
                                  model: nn.Module,
                                  features: Dict[str, torch.Tensor],
                                  labels: torch.Tensor,
                                  target_modality: str = "all",
                                  binary_search_steps: int = 5) -> Dict[str, torch.Tensor]:
        """
        生成边界样本（位于决策边界附近的样本）
        
        Args:
            model: 目标模型
            features: 原始特征
            labels: 原始标签
            target_modality: 目标模态
            binary_search_steps: 二分查找步数，用于找到最小的有效扰动
            
        Returns:
            边界样本特征
        """
        # 复制特征
        boundary_features = {}
        for key, value in features.items():
            boundary_features[key] = value.clone().detach()
        
        # 生成对抗样本
        adv_features = self._generate_adversarial_examples(model, features, labels, target_modality)
        
        # 检查对抗样本是否成功
        with torch.no_grad():
            outputs = model(adv_features)
            preds = torch.argmax(outputs, dim=1)
            success_indices = (preds != labels)
        
        # 对于成功的对抗样本，使用二分查找寻找边界
        for idx in range(len(labels)):
            if not success_indices[idx]:
                continue
            
            # 对每个目标模态执行二分查找
            for key in boundary_features:
                if target_modality != "all" and key != target_modality:
                    continue
                
                # 设置二分查找的上下界
                alpha_low = 0.0
                alpha_high = 1.0
                
                # 执行二分查找
                for _ in range(binary_search_steps):
                    alpha_mid = (alpha_low + alpha_high) / 2.0
                    
                    # 创建中间样本
                    mid_features = {k: v.clone() for k, v in features.items()}
                    mid_features[key][idx] = features[key][idx] + alpha_mid * (adv_features[key][idx] - features[key][idx])
                    
                    # 检查中间样本的预测结果
                    with torch.no_grad():
                        mid_outputs = model(mid_features)
                        mid_pred = torch.argmax(mid_outputs[idx])
                    
                    # 更新搜索范围
                    if mid_pred == labels[idx]:
                        alpha_low = alpha_mid
                    else:
                        alpha_high = alpha_mid
                
                # 使用最终的alpha生成边界样本
                boundary_features[key][idx] = features[key][idx] + alpha_high * (adv_features[key][idx] - features[key][idx])
        
        return boundary_features
    
    def analyze_modality_robustness(self,
                                   model: nn.Module,
                                   features: Dict[str, torch.Tensor],
                                   labels: torch.Tensor) -> Dict[str, float]:
        """
        分析不同模态的鲁棒性
        
        Args:
            model: 目标模型
            features: 原始特征
            labels: 原始标签
            
        Returns:
            各模态的鲁棒性分数
        """
        robustness_scores = {}
        
        # 获取所有模态
        modalities = list(features.keys())
        
        # 对每个模态进行分析
        for modality in modalities:
            # 为当前模态生成对抗样本
            verification_results = self.forward(
                model, 
                features, 
                labels, 
                target_modality=modality
            )
            
            # 计算该模态的鲁棒性得分
            original_acc = verification_results['original_accuracy'].item()
            adv_acc = verification_results['adversarial_accuracy'].item()
            
            # 鲁棒性得分: 对抗准确率 / 原始准确率
            robustness_score = adv_acc / original_acc if original_acc > 0 else 0
            robustness_scores[modality] = robustness_score
            
        # 计算总体鲁棒性分数
        verification_results = self.forward(
            model, 
            features, 
            labels, 
            target_modality="all"
        )
        
        original_acc = verification_results['original_accuracy'].item()
        adv_acc = verification_results['adversarial_accuracy'].item()
        
        robustness_score = adv_acc / original_acc if original_acc > 0 else 0
        robustness_scores['overall'] = robustness_score
        
        return robustness_scores 