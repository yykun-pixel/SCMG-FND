import json
import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建保存目录
os.makedirs('data/gpt_description', exist_ok=True)

def load_dataset_ids():
    """
    从label目录中加载训练、验证和测试集的视频ID
    
    Returns:
        dict: 包含train、val和test集视频ID的字典
    """
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        # 加载对应的label文件
        label_path = f'data/label/label_{split}.pkl'
        try:
            with open(label_path, 'rb') as f:
                label_data = pickle.load(f)
                # 获取视频ID (键)
                video_ids = list(label_data.keys())
                datasets[split] = set(video_ids)
                print(f"从{label_path}加载了{len(video_ids)}个视频ID")
        except Exception as e:
            print(f"加载{label_path}时出错：{e}")
            datasets[split] = set()
    
    return datasets

def merge_json_to_text(item):
    """
    将结构化JSON合并为单一文本
    
    Args:
        item (dict): 包含隐式意见的JSON对象
        
    Returns:
        str: 合并后的文本
    """
    text = ""
    if 'news_type' in item:
        text += f"新闻类型：{item['news_type']}，"
        
        if item['news_type'] == '事件型':
            # 处理事件型新闻
            if '基础事件要素' in item:
                event = item['基础事件要素']
                text += f"事件时间：{event.get('事件时间', 'null')}，"
                text += f"发生地点：{event.get('发生地点', 'null')}，"
                if '关键实体' in event and isinstance(event['关键实体'], list):
                    text += f"关键实体：{'、'.join(event['关键实体'])}，"
                text += f"核心事件描述：{event.get('核心事件描述', 'null')}，"
            
            if '矛盾点分析' in item:
                contradiction = item['矛盾点分析']
                text += f"时间矛盾：{contradiction.get('时间矛盾', 'null')}，"
                text += f"数据矛盾：{contradiction.get('数据矛盾', 'null')}，"
                text += f"逻辑矛盾：{contradiction.get('逻辑矛盾', 'null')}，"
            
            if '可信度证据' in item and isinstance(item['可信度证据'], list):
                text += f"可信度证据：{'；'.join(item['可信度证据'])}，"
            
            if '语言分析' in item:
                lang = item['语言分析']
                text += f"情感倾向：{lang.get('情感倾向', 'null')}，"
                if '模糊点' in lang and isinstance(lang['模糊点'], list):
                    text += f"模糊点：{'、'.join(lang['模糊点'])}，"
        
        elif item['news_type'] == '常识型':
            # 处理常识型新闻
            if '核心主张' in item:
                claim = item['核心主张']
                text += f"主张类型：{claim.get('主张类型', 'null')}，"
                text += f"具体主张：{claim.get('具体主张', 'null')}，"
                text += f"涉及对象：{claim.get('涉及对象', 'null')}，"
                text += f"建议来源：{claim.get('建议来源', 'null')}，"
            
            if '常识矛盾' in item:
                contradiction = item['常识矛盾']
                text += f"医学矛盾：{contradiction.get('医学矛盾', 'null')}，"
                text += f"科学矛盾：{contradiction.get('科学矛盾', 'null')}，"
                text += f"生活矛盾：{contradiction.get('生活矛盾', 'null')}，"
            
            if '逻辑分析' in item:
                logic = item['逻辑分析']
                text += f"因果证据：{logic.get('因果证据', 'null')}，"
                text += f"证据来源：{logic.get('证据来源', 'null')}，"
                text += f"例外漏洞：{logic.get('例外漏洞', 'null')}，"
            
            if '语言陷阱' in item:
                trap = item['语言陷阱']
                if '绝对化词汇' in trap and isinstance(trap['绝对化词汇'], list):
                    text += f"绝对化词汇：{'、'.join(trap['绝对化词汇'])}，"
                text += f"权威风险：{trap.get('权威风险', 'null')}，"
                text += f"伪科学标签：{trap.get('伪科学标签', 'null')}，"
        
        # 通用字段
        if '可信度评分' in item:
            text += f"可信度评分：{item['可信度评分']}，"
        
        if '综合评价' in item:
            text += f"综合评价：{item['综合评价']}"
    
    # 如果有直接的gpt_description字段，可以直接使用
    elif 'gpt_description' in item:
        text = item['gpt_description']
        
    return text.strip()

def extract_bert_features(texts, batch_size=8):
    """
    使用BERT提取文本特征
    
    Args:
        texts (list): 文本列表
        batch_size (int): 批处理大小
        
    Returns:
        np.ndarray: 提取的特征，形状为 (len(texts), 768)
    """
    # 加载预训练BERT模型和分词器
    print("加载BERT模型和分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    model.eval()
    model.to(device)
    
    features = []
    
    # 按批次处理文本
    for i in tqdm(range(0, len(texts), batch_size), desc="提取BERT特征"):
        batch_texts = texts[i:i+batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, 
                               truncation=True, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # 使用[CLS]标记的输出作为文本特征
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(batch_features)
    
    return np.concatenate(features, axis=0)

def main():
    # 加载数据集ID划分
    dataset_ids = load_dataset_ids()
    
    # 读取新隐式意见数据
    print("读取implicit_features.json...")
    try:
        with open('implicit_features.json', 'r', encoding='utf-8') as f:
            implicit_data = json.load(f)
        print(f"成功读取，包含{len(implicit_data)}条记录")
    except Exception as e:
        print(f"读取implicit_features.json时出错：{e}")
        return
    
    # 处理每个数据集
    for split, video_ids in dataset_ids.items():
        print(f"\n开始处理{split}数据集...")
        
        # 为当前数据集创建特征字典
        features = {}
        
        # 收集缺失的ID
        missing_ids = []
        processed_count = 0
        
        # 记录视频ID和对应的文本，用于批量提取特征
        video_id_list = []
        text_list = []
        
        for video_id in video_ids:
            video_id_str = str(video_id)  # 确保ID是字符串格式
            
            if video_id_str in implicit_data:
                # 合并文本
                text = merge_json_to_text(implicit_data[video_id_str])
                
                # 收集ID和文本
                video_id_list.append(video_id)
                text_list.append(text)
                processed_count += 1
            else:
                missing_ids.append(video_id)
        
        print(f"收集了{len(text_list)}个文本，准备批量提取特征")
        
        if text_list:
            # 批量提取特征
            all_features = extract_bert_features(text_list)
            
            # 将特征与ID匹配
            for i, vid in enumerate(video_id_list):
                features[vid] = all_features[i]
        
        # 报告结果
        print(f"处理完成：{processed_count}个视频有特征，{len(missing_ids)}个视频缺失")
        if missing_ids:
            print(f"前5个缺失ID示例：{missing_ids[:5]}")
        
        # 保存特征
        output_path = f'data/gpt_description/gpt_description_{split}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"{split}数据集特征已保存至 {output_path}")
    
    print("\n所有数据集处理完成！")

if __name__ == "__main__":
    main() 