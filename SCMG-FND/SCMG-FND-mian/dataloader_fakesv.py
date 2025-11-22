import math
import os
import pickle
import json  # 新增：用于加载隐式意见数据

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertTokenizer
from torchvision import models
from transformers import BertModel, BertTokenizer
# from src.models import MULTModel
# from src.main import hyp_params
# avgpool = models.vgg19(pretrained=True).avgpool.cuda()
# classifier = models.vgg19(pretrained=True).classifier[:4].cuda()
import argparse
# from src.utils import *
from torch.utils.data import DataLoader
# from src import train


# 得到一个视频对应的所有数据
class SVFENDDataset(Dataset):

    def __init__(self, datamode='title+ocr', train_or_test='train', opinion_data_path='enhanced_results.json'):  #标题+转录

        print(f"初始化数据集，模式: {datamode}, 数据集: {train_or_test}")
        
        # 加载隐式意见数据
        self.opinion_data = self.load_opinion_data(opinion_data_path)
        
        # 读取各模态特征
        #音频特征vggish
        with open(os.path.join('data/audio', 'audio_'+train_or_test+'.pkl'), "rb") as fr:
            self.audio = pickle.load(fr)

        # 文本特征
        if datamode == 'title':
            with open(os.path.join('data/text_title_temporal', 'text_title_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                self.text = pickle.load(fr)
        elif datamode == 'title+ocr':
            with open(os.path.join('data/text_title_ocr_temporal', 'text_title_ocr_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                self.text = pickle.load(fr)
        elif datamode == 'both':
            # 'both' 模式使用 title+ocr 数据
            try:
                with open(os.path.join('data/text_title_ocr_temporal', 'text_title_ocr_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                    self.text = pickle.load(fr)
            except FileNotFoundError:
                # 如果找不到title+ocr，尝试title
                with open(os.path.join('data/text_title_temporal', 'text_title_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                    self.text = pickle.load(fr)
        else:
            # 默认使用title+ocr
            with open(os.path.join('data/text_title_ocr_temporal', 'text_title_ocr_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                self.text = pickle.load(fr)

        with open(os.path.join('data/comments', 'comments_' + train_or_test + '.pkl'), "rb") as fr:
            self.comments = pickle.load(fr)

        # gpt生成的文本分析
        with open(os.path.join('data/gpt_description', 'gpt_description_' + train_or_test + '.pkl'), "rb") as fr:
            self.gpt_description = pickle.load(fr)

        # label
        with open(os.path.join('data/label', 'label_'+train_or_test+'.pkl'), "rb") as fr:
            self.label = pickle.load(fr)

        #vgg9视频帧特征
        with open(os.path.join('data/video', 'video_'+train_or_test+'.pkl'), "rb") as fr:
            self.video = pickle.load(fr)

        # user_intro
        with open(os.path.join('data/user_intro', 'user_intro_'+train_or_test+'.pkl'), "rb") as fr:
            self.user_intro = pickle.load(fr)

        # vid
        with open(os.path.join('data/vid', 'vid_'+train_or_test+'.pkl'), "rb") as fr:
            self.vid = pickle.load(fr)

        # c3d
        with open(os.path.join('data/c3d', 'c3d_'+train_or_test+'.pkl'), "rb") as fr:
            self.c3d = pickle.load(fr)

        self.audio = dict(filter(lambda item: item[0] in self.vid, self.audio.items()))
        
        # 检查数据加载状态
        print(f"数据集加载完成. vid长度: {len(self.vid)}, text类型: {type(self.text)}")
        
        # 确保vid和text匹配
        if isinstance(self.text, dict):
            # 如果text是字典，确保所有vid的键都存在
            self.valid_indices = []
            for i, vid_key in enumerate(self.vid):
                if vid_key in self.text and vid_key in self.audio and vid_key in self.comments and vid_key in self.label:
                    self.valid_indices.append(i)
            print(f"有效索引数量: {len(self.valid_indices)}/{len(self.vid)}")
        else:
            # 如果text是列表，检查长度是否匹配
            if len(self.text) != len(self.vid):
                print(f"警告: text长度({len(self.text)})与vid长度({len(self.vid)})不匹配")
                # 取最小长度
                self.valid_indices = list(range(min(len(self.text), len(self.vid))))
            else:
                self.valid_indices = list(range(len(self.vid)))

    def load_opinion_data(self, opinion_data_path):
        """加载隐式意见分析数据"""
        try:
            with open(opinion_data_path, 'r', encoding='utf-8') as f:
                opinion_data = json.load(f)
            print(f"✅ 成功加载隐式意见数据: {len(opinion_data)} 条记录")
            
            # 如果数据是列表，转换为字典格式（以video_id为键）
            if isinstance(opinion_data, list):
                video_id_dict = {}
                for data in opinion_data:
                    if isinstance(data, dict) and 'video_id' in data:
                        # 使用video_id作为键，同时支持数字和字符串格式
                        video_id = data['video_id']
                        video_id_dict[video_id] = data
                        video_id_dict[str(video_id)] = data  # 同时存储字符串格式
                print(f"✅ 转换为video_id字典，包含 {len(video_id_dict)//2} 个有效映射")
                return video_id_dict
            elif isinstance(opinion_data, dict):
                return opinion_data
            else:
                print(f"⚠️ 意见数据格式不支持，使用空字典")
                return {}
                
        except FileNotFoundError:
            print(f"⚠️ 隐式意见数据文件不存在: {opinion_data_path}，将使用空数据")
            return {}
        except Exception as e:
            print(f"❌ 加载隐式意见数据失败: {e}，将使用空数据")
            return {}

    def __len__(self):
        return len(self.valid_indices)
     
    def __getitem__(self, idx):
        # 使用有效索引
        real_idx = self.valid_indices[idx]
        vid = self.vid[real_idx]
        
        # 准备所有模态数据
        if isinstance(self.text, dict):
            text = torch.tensor(self.text[vid], dtype=torch.float32)
        else:
            text = torch.tensor(self.text[real_idx], dtype=torch.float32)
            
        comments = self.comments[vid]
        gpt_description = self.gpt_description[vid]
        audio = torch.tensor(self.audio[vid], dtype=torch.float32)
        video = torch.tensor(self.video[vid], dtype=torch.float32)
        c3d = torch.tensor(self.c3d[vid], dtype=torch.float32)
        label = torch.tensor(self.label[vid])
        user_intro = self.user_intro[vid]

        # 获取对应的隐式意见数据
        opinion_data = None
        if self.opinion_data:
            # 通过video_id匹配意见数据
            if vid in self.opinion_data:
                opinion_data = self.opinion_data[vid]
                # print(f"🎯 匹配成功: vid={vid} -> opinion_data")
            elif str(vid) in self.opinion_data:
                opinion_data = self.opinion_data[str(vid)]
                # print(f"🎯 匹配成功: str(vid)={str(vid)} -> opinion_data")
            # else:
            #     print(f"⚠️ 未找到匹配: vid={vid}, 可用键示例: {list(self.opinion_data.keys())[:3]}")

        return {
            'label': label,  # 标签
            'text': text,
            'audioframes': audio,  # 音频帧
            'frames': video,  # 帧
            'comments': comments, # 评论
            'c3d': c3d,  # C3D特征
            'user_intro': user_intro,
            'gpt_description': gpt_description, # gpt生成的文本辅助分析
            'implicit_opinion_data': opinion_data  # 隐式意见数据
        }

def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len>=seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        # video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        video = video.squeeze()
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video, torch.zeros([seq_len-ori_len, video.shape[1]], dtype=torch.float32)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def SVFEND_collate_fn(batch):
    # num_comments = 23
    num_frames = 83
    num_audioframes = 50

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)
    frames = frames.squeeze()

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    comments = [item['comments'] for item in batch]
    # 确保comments是tensor
    processed_comments = []
    for comment in comments:
        # 如果是numpy数组，转换为tensor
        if isinstance(comment, np.ndarray):
            comment = torch.tensor(comment, dtype=torch.float32)
        # 如果是列表，也转换为tensor
        elif isinstance(comment, list):
            comment = torch.tensor(comment, dtype=torch.float32)
        processed_comments.append(comment)
    comments = torch.stack(processed_comments)

    # 确保gpt_description字段存在并且格式正确
    gpt_description = []
    for item in batch:
        if 'gpt_description' in item:
            gpt = item['gpt_description']
        else:
            # 如果不存在，使用零向量替代
            print("警告: 样本中不存在gpt_description字段，使用零向量替代")
            # 假设维度与模型中的t_in参数一致（论文中为768）
            gpt = np.zeros(768, dtype=np.float32)
        
        # 处理不同的数据类型
        if isinstance(gpt, np.ndarray):
            gpt = torch.tensor(gpt, dtype=torch.float32)
        elif isinstance(gpt, list):
            gpt = torch.tensor(gpt, dtype=torch.float32)
        elif not isinstance(gpt, torch.Tensor):
            # 如果既不是numpy数组也不是列表或tensor，尝试转换为tensor
            try:
                gpt = torch.tensor(gpt, dtype=torch.float32)
            except:
                print(f"警告: 无法将gpt_description转换为tensor，类型: {type(gpt)}")
                gpt = torch.zeros(768, dtype=torch.float32)
        
        gpt_description.append(gpt)

    # 尝试堆叠，如果形状不一致，打印详细信息并进行处理
    try:
        gpt_description = torch.stack(gpt_description)
    except RuntimeError as e:
        print(f"警告: 在stack gpt_description时出错: {e}")
        # 打印每个tensor的形状以诊断问题
        for i, gpt in enumerate(gpt_description):
            print(f"  gpt[{i}].shape = {gpt.shape}")
        
        # 将所有tensor转换为相同的形状（使用第一个非零tensor的形状）
        target_shape = None
        for gpt in gpt_description:
            if gpt.numel() > 0:
                target_shape = gpt.shape
                break
        
        if target_shape is None:
            target_shape = (768,)  # 默认形状
        
        processed_gpt = []
        for gpt in gpt_description:
            if gpt.shape != target_shape:
                # 如果形状不匹配，使用零tensor
                processed_gpt.append(torch.zeros(target_shape, dtype=torch.float32))
            else:
                processed_gpt.append(gpt)
        
        gpt_description = torch.stack(processed_gpt)

    user_intro = [item['user_intro'] for item in batch]
    # 确保user_intro是tensor
    processed_user_intro = []
    for intro in user_intro:
        if isinstance(intro, np.ndarray):
            intro = torch.tensor(intro, dtype=torch.float32)
        elif isinstance(intro, list):
            intro = torch.tensor(intro, dtype=torch.float32)
        processed_user_intro.append(intro)
    user_intro = torch.stack(processed_user_intro)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]
    text = [item['text'] for item in batch]
    text = torch.tensor([item.cpu().detach().numpy() for item in text])
    
    # 处理隐式意见数据
    implicit_opinion_data = []
    for item in batch:
        if 'implicit_opinion_data' in item and item['implicit_opinion_data'] is not None:
            implicit_opinion_data.append(item['implicit_opinion_data'])
        else:
            implicit_opinion_data.append(None)

    return {
        'label': torch.stack(label),
        'text': text,
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'comments': comments,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        'user_intro': user_intro,
        'gpt_description': gpt_description,
        'implicit_opinion_data': implicit_opinion_data,
    }

def _init_fn(worker_id):
    np.random.seed(2022)

def get_dataloader(modelConfig,data_type='SVFEND'):
    collate_fn=None

    if data_type == 'SVFEND':
        # 获取隐式意见数据路径
        opinion_data_path = modelConfig.get("opinion_data_path", "enhanced_results.json")
        
        dataset_train = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='train', opinion_data_path=opinion_data_path)
        dataset_val = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='val', opinion_data_path=opinion_data_path)
        dataset_test = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='test', opinion_data_path=opinion_data_path)
        collate_fn=SVFEND_collate_fn

    # 提取可选参数
    drop_last = modelConfig.get("drop_last_batch", False)
    num_workers = modelConfig.get("num_workers", 0)
    pin_memory = modelConfig.get("pin_memory", True)
    
    # 如果启用了drop_last，打印提示
    if drop_last:
        print("已启用drop_last，将丢弃不完整的最后一批次数据")

    train_dataloader = DataLoader(dataset_train, batch_size=modelConfig["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=drop_last,  # 使用配置参数
        worker_init_fn=_init_fn,
        collate_fn=collate_fn)
    
    val_dataloader = DataLoader(dataset_val, batch_size=modelConfig["batch_size"],
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=False,
                                drop_last=drop_last,  # 对验证集也使用相同配置
                                worker_init_fn=_init_fn,
                                collate_fn=collate_fn)
    
    test_dataloader = DataLoader(dataset_test, batch_size=modelConfig["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=drop_last,  # 对测试集也使用相同配置
        worker_init_fn=_init_fn,
        collate_fn=collate_fn)

    dataloaders = dict(zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]))

    return dataloaders

def split_word(df):  #去除停用词
    title = df['description'].values
    comments = df['comments'].apply(lambda x:' '.join(x)).values
    text = np.concatenate([title, comments],axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus