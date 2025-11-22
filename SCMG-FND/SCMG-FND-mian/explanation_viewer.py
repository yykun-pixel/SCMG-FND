#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可解释性结果可视化界面
用于交互式查看和分析模型的解释结果
"""

import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob

st.set_page_config(
    page_title="虚假视频检测可解释性分析",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #004D40;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0277BD;
        margin-top: 1rem;
    }
    .highlight-text {
        background-color: #F9FBE7;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .correct {
        color: #2E7D32;
        font-weight: bold;
    }
    .incorrect {
        color: #C62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 标题
    st.markdown('<div class="main-header">虚假视频检测可解释性分析</div>', unsafe_allow_html=True)
    st.markdown("本工具用于可视化和分析模型的可解释性结果，帮助理解模型如何区分真实和虚假视频。")
    
    # 侧边栏：选择结果目录
    st.sidebar.markdown('<div class="sub-header">设置</div>', unsafe_allow_html=True)
    explanation_dir = st.sidebar.text_input("解释结果目录路径", value="explanation_results")
    
    if not os.path.exists(explanation_dir):
        st.warning(f"目录 '{explanation_dir}' 不存在。请输入有效的解释结果目录路径。")
        return
    
    # 加载汇总信息
    summary_path = os.path.join(explanation_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        # 显示基本信息
        st.markdown('<div class="section-header">基本信息</div>', unsafe_allow_html=True)
        st.write(f"生成时间: {summary['timestamp']}")
        st.write(f"处理批次数: {summary['num_processed_batches']}")
        st.write(f"模型路径: {summary['model_path']}")
        
        # 列出批次目录
        batch_dirs = summary.get('batch_dirs', [])
        if not batch_dirs:
            # 如果汇总文件中没有批次目录，尝试直接获取
            batch_dirs = [d for d in glob.glob(os.path.join(explanation_dir, "batch_*")) if os.path.isdir(d)]
    else:
        # 如果没有汇总文件，尝试直接获取批次目录
        batch_dirs = [d for d in glob.glob(os.path.join(explanation_dir, "batch_*")) if os.path.isdir(d)]
        
    if not batch_dirs:
        st.warning("未找到任何批次数据。请确保目录结构正确。")
        return
    
    # 选择批次
    batch_options = [os.path.basename(d) for d in batch_dirs]
    selected_batch = st.sidebar.selectbox("选择批次", batch_options)
    
    # 加载所选批次的元数据
    batch_dir = os.path.join(explanation_dir, selected_batch)
    batch_metadata_path = os.path.join(batch_dir, "metadata.json")
    
    if os.path.exists(batch_metadata_path):
        with open(batch_metadata_path, "r") as f:
            batch_metadata = json.load(f)
        
        # 显示批次信息
        st.markdown('<div class="section-header">批次信息</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"批次ID: {batch_metadata['batch_idx']}")
        with col2:
            st.write(f"样本数量: {batch_metadata['num_samples']}")
        with col3:
            accuracy = batch_metadata['accuracy'] * 100
            st.write(f"准确率: {accuracy:.2f}%")
        
        # 获取样本目录
        sample_dirs = [d for d in glob.glob(os.path.join(batch_dir, "sample_*")) if os.path.isdir(d)]
        
        if not sample_dirs:
            st.warning("未找到该批次的样本数据。")
            return
        
        # 选择样本
        sample_options = [os.path.basename(d) for d in sample_dirs]
        selected_sample = st.sidebar.selectbox("选择样本", sample_options)
        
        # 加载所选样本的信息
        sample_dir = os.path.join(batch_dir, selected_sample)
        sample_info_path = os.path.join(sample_dir, "sample_info.json")
        
        if os.path.exists(sample_info_path):
            with open(sample_info_path, "r") as f:
                sample_info = json.load(f)
            
            # 显示样本信息
            st.markdown('<div class="section-header">样本信息</div>', unsafe_allow_html=True)
            correct = sample_info['correct']
            prediction = "假" if sample_info['predicted'] == 1 else "真"
            truth = "假" if sample_info['true_label'] == 1 else "真"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"样本ID: {sample_info['sample_idx']}")
            with col2:
                st.write(f"预测: {prediction}")
            with col3:
                st.write(f"真实标签: {truth}")
            
            result_class = "correct" if correct else "incorrect"
            st.markdown(f'<div class="{result_class}">预测结果: {"正确" if correct else "错误"}</div>', unsafe_allow_html=True)
            
            # 创建主要内容区域的选项卡
            tab1, tab2, tab3, tab4 = st.tabs(["模态贡献", "特征重要性", "虚假区域检测", "模态间注意力"])
            
            # 选项卡1: 模态贡献
            with tab1:
                modality_weights_path = os.path.join(sample_dir, "modality_weights.npy")
                if os.path.exists(modality_weights_path):
                    weights = np.load(modality_weights_path)
                    
                    st.markdown('<div class="section-header">模态贡献度</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(['文本', '音频', '视频'], weights, color=['#1976D2', '#D32F2F', '#388E3C'])
                    ax.set_title('各模态对判断结果的贡献度')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('贡献度')
                    
                    # 添加数值标签
                    for i, v in enumerate(weights):
                        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
                    
                    st.pyplot(fig)
                    
                    # 提供分析
                    max_modality = ['文本', '音频', '视频'][np.argmax(weights)]
                    st.markdown(f'<div class="highlight-text">分析: {max_modality}模态在此样本的判断中贡献最大，权重为{weights.max():.3f}</div>', unsafe_allow_html=True)
                else:
                    st.warning("未找到模态贡献度数据。")
            
            # 选项卡2: 特征重要性
            with tab2:
                st.markdown('<div class="section-header">特征重要性</div>', unsafe_allow_html=True)
                
                # 文本特征重要性
                text_imp_path = os.path.join(sample_dir, "text_importance.npy")
                if os.path.exists(text_imp_path):
                    text_imp = np.load(text_imp_path)
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(text_imp, color='#1976D2')
                    ax.set_title('文本特征重要性')
                    ax.set_xlabel('特征索引')
                    ax.set_ylabel('重要性')
                    st.pyplot(fig)
                    
                    # 计算前5个最重要的特征
                    top5_indices = np.argsort(text_imp)[-5:][::-1]
                    top5_values = text_imp[top5_indices]
                    
                    st.write("前5个最重要的文本特征:")
                    for i, (idx, val) in enumerate(zip(top5_indices, top5_values)):
                        st.write(f"{i+1}. 特征 #{idx}: {val:.4f}")
                else:
                    st.info("未找到文本特征重要性数据。")
                
                # 音频特征重要性
                audio_imp_path = os.path.join(sample_dir, "audio_importance.npy")
                if os.path.exists(audio_imp_path):
                    audio_imp = np.load(audio_imp_path)
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(audio_imp, color='#D32F2F')
                    ax.set_title('音频特征重要性')
                    ax.set_xlabel('特征索引')
                    ax.set_ylabel('重要性')
                    st.pyplot(fig)
                    
                    # 计算前5个最重要的特征
                    top5_indices = np.argsort(audio_imp)[-5:][::-1]
                    top5_values = audio_imp[top5_indices]
                    
                    st.write("前5个最重要的音频特征:")
                    for i, (idx, val) in enumerate(zip(top5_indices, top5_values)):
                        st.write(f"{i+1}. 特征 #{idx}: {val:.4f}")
                else:
                    st.info("未找到音频特征重要性数据。")
                
                # 视频特征重要性
                video_imp_path = os.path.join(sample_dir, "video_importance.npy")
                if os.path.exists(video_imp_path):
                    video_imp = np.load(video_imp_path)
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(video_imp, color='#388E3C')
                    ax.set_title('视频特征重要性')
                    ax.set_xlabel('特征索引')
                    ax.set_ylabel('重要性')
                    st.pyplot(fig)
                    
                    # 计算前5个最重要的特征
                    top5_indices = np.argsort(video_imp)[-5:][::-1]
                    top5_values = video_imp[top5_indices]
                    
                    st.write("前5个最重要的视频特征:")
                    for i, (idx, val) in enumerate(zip(top5_indices, top5_values)):
                        st.write(f"{i+1}. 特征 #{idx}: {val:.4f}")
                else:
                    st.info("未找到视频特征重要性数据。")
            
            # 选项卡3: 虚假区域检测
            with tab3:
                st.markdown('<div class="section-header">虚假区域热图</div>', unsafe_allow_html=True)
                
                heatmap_path = os.path.join(sample_dir, "fake_region_heatmap.npy")
                if os.path.exists(heatmap_path):
                    heatmap = np.load(heatmap_path)
                    
                    # 将一维热图转换为二维图像进行可视化
                    hm_size = int(np.sqrt(len(heatmap)))
                    if hm_size**2 != len(heatmap):
                        # 如果不是完美平方数，选择最近的矩形形状
                        hm_width = hm_size
                        hm_height = len(heatmap) // hm_width + (1 if len(heatmap) % hm_width != 0 else 0)
                        heatmap_2d = np.zeros((hm_height, hm_width))
                        heatmap_2d.flat[:len(heatmap)] = heatmap
                    else:
                        heatmap_2d = heatmap.reshape(hm_size, hm_size)
                    
                    # 显示热图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(heatmap_2d, cmap='hot')
                    ax.set_title('虚假区域热图')
                    fig.colorbar(im, ax=ax, label='虚假程度')
                    st.pyplot(fig)
                    
                    # 检查是否有视频帧可视化
                    video_frames_dir = os.path.join(sample_dir, "video_frames")
                    if os.path.exists(video_frames_dir):
                        st.markdown('<div class="section-header">虚假区域定位（视频帧）</div>', unsafe_allow_html=True)
                        
                        # 获取所有可视化的视频帧
                        frame_paths = sorted(glob.glob(os.path.join(video_frames_dir, "frame_*.png")))
                        
                        if frame_paths:
                            # 创建一个图库显示所有帧
                            st.write("下面显示了应用热图后的视频帧，突出显示了可能的虚假区域:")
                            
                            # 每行显示3张图片
                            cols = st.columns(3)
                            for i, frame_path in enumerate(frame_paths):
                                with cols[i % 3]:
                                    img = Image.open(frame_path)
                                    st.image(img, caption=f"帧 {i}", use_column_width=True)
                        else:
                            st.info("未找到视频帧可视化。")
                    else:
                        st.info("未找到视频帧可视化目录。")
                else:
                    st.warning("未找到虚假区域热图数据。")
            
            # 选项卡4: 模态间注意力
            with tab4:
                st.markdown('<div class="section-header">模态间注意力</div>', unsafe_allow_html=True)
                
                # 文本-视频注意力
                text_video_attn_path = os.path.join(sample_dir, "text_video_attention.npy")
                if os.path.exists(text_video_attn_path):
                    attn = np.load(text_video_attn_path)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(attn, cmap='viridis', aspect='auto')
                    ax.set_title('文本-视频注意力图')
                    ax.set_xlabel('文本序列')
                    ax.set_ylabel('视频特征')
                    fig.colorbar(im, ax=ax, label='注意力权重')
                    st.pyplot(fig)
                    
                    # 分析注意力分布
                    mean_attn = np.mean(attn, axis=1)
                    max_video_feature = np.argmax(mean_attn)
                    st.markdown(f'<div class="highlight-text">分析: 视频特征 #{max_video_feature} 与文本特征的相关性最高</div>', unsafe_allow_html=True)
                else:
                    st.info("未找到文本-视频注意力数据。")
                
                # 音频-视频注意力
                audio_video_attn_path = os.path.join(sample_dir, "audio_video_attention.npy")
                if os.path.exists(audio_video_attn_path):
                    attn = np.load(audio_video_attn_path)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(attn, cmap='viridis', aspect='auto')
                    ax.set_title('音频-视频注意力图')
                    ax.set_xlabel('音频序列')
                    ax.set_ylabel('视频特征')
                    fig.colorbar(im, ax=ax, label='注意力权重')
                    st.pyplot(fig)
                    
                    # 分析注意力分布
                    mean_attn = np.mean(attn, axis=1)
                    max_video_feature = np.argmax(mean_attn)
                    st.markdown(f'<div class="highlight-text">分析: 视频特征 #{max_video_feature} 与音频特征的相关性最高</div>', unsafe_allow_html=True)
                else:
                    st.info("未找到音频-视频注意力数据。")
            
        else:
            st.warning(f"未找到样本信息文件: {sample_info_path}")
    else:
        st.warning(f"未找到批次元数据文件: {batch_metadata_path}")

if __name__ == "__main__":
    main() 