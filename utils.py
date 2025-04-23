import os
import re
import math
import torch
import random
import pickle
import datetime
from rouge import rouge
from bleu import compute_bleu

# New import
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
import torch.nn as nn
import pdb



def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, data_path, index_dir, tokenizer, seq_len, image_dir=None, use_images=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.user_dict = EntityDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # 添加图片处理所需的变量
        self.use_images = use_images
        self.device = device
        self.image_dir = image_dir
        self.item_json_path = os.path.join(os.path.dirname(data_path), 'item.json')

        # 初始化CLIP
        if self.use_images:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            # Freeze CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Load item descriptions from item.json
            self.item_descriptions = self.load_item_descriptions()
            
            # Generate image and text embeddings
            self.item_id_to_asin = self.create_item_id_to_asin_mapping(data_path)
            self.item_embeddings = self.generate_item_embeddings()

        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def create_item_id_to_asin_mapping(self, data_path):
        """Create a mapping from item_idx to ASIN (item id)"""
        item_id_to_asin = {}
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            item_idx = self.item_dict.entity2idx[review['item']]
            item_id_to_asin[item_idx] = review['item']  # Assuming 'item' field contains the ASIN
        return item_id_to_asin
    
    def load_item_descriptions(self):
        """Load item descriptions from item.json"""
        if not os.path.exists(self.item_json_path):
            print(f"Warning: item.json not found at {self.item_json_path}")
            return {}
        
        with open(self.item_json_path, 'r') as f:
            items = json.load(f)
        
        descriptions = {}
        for item in items:
            if 'item' in item and 'description' in item:
                descriptions[item['item']] = item['description']
        
        print(f"Loaded {len(descriptions)} item descriptions")
        return descriptions

    def generate_item_embeddings(self):
        """Generate combined image and text embeddings for all items"""
        if not self.item_id_to_asin:
            return {}
        
        item_embeddings = {}
        print("Generating item embeddings with CLIP...")
        
        for item_idx, asin in tqdm(self.item_id_to_asin.items()):
            # Process image if available
            image_path = os.path.join(self.image_dir, f"{asin}.jpg")
            image_embedding = None
            
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        image_embedding = self.clip_model.get_image_features(**image_inputs).cpu()
                except Exception as e:
                    print(f"Error processing image {asin}: {e}")
            
            # Process text description if available
            text_embedding = None
            if asin in self.item_descriptions:
                description = self.item_descriptions[asin]
                # breakpoint()
                try:
                    text_inputs = self.clip_processor(text=[description], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    with torch.no_grad():
                        text_embedding = self.clip_model.get_text_features(**text_inputs).cpu()
                except Exception as e:
                    print(f"Error processing description for {asin}: {e}")
            
            # 使用平均值方法而不是连接
            if image_embedding is not None and text_embedding is not None:
                # 使用平均值方法
                combined_embedding = (image_embedding + text_embedding) / 2
                item_embeddings[item_idx] = combined_embedding
            elif image_embedding is not None:
                item_embeddings[item_idx] = image_embedding
            elif text_embedding is not None:
                item_embeddings[item_idx] = text_embedding
        
        print(f"Generated embeddings for {len(item_embeddings)} items")
        return item_embeddings

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])  # keep seq_len tokens at most
            
            item_idx = self.item_dict.entity2idx[review['item']]
            
            data_item = {
                'user': self.user_dict.entity2idx[review['user']],
                'item': item_idx,
                'rating': review['rating'],
                'text': text,
                'feature': fea
            }
            
            # Add image embedding if available
            if self.use_images and item_idx in self.item_embeddings:
                data_item['item_embedding'] = self.item_embeddings[item_idx]
            
            data.append(data_item)
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        
        for idx in valid_index:
            valid.append(data[idx])
        
        for idx in test_index:
            test.append(data[idx])
        
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

class Batchify:
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        u, i, r, t, self.feature = [], [], [], [], []
        self.has_embeddings = False
        self.item_embeddings = []
        
        # 检查数据项中是否包含图像嵌入
        if len(data) > 0 and 'item_embedding' in data[0]:
            self.has_embeddings = True
            
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append('{} {} {} {}'.format(x['feature'], bos, x['text'], eos))
            self.feature.append(x['feature'])
            
            # 如果有嵌入向量，添加到列表中
            if self.has_embeddings and 'item_embedding' in x:
                self.item_embeddings.append(x['item_embedding'])
            elif self.has_embeddings:
                # 如果应该有嵌入但当前项没有，使用零向量
                # 假设第一个嵌入的shape作为标准
                self.item_embeddings.append(torch.zeros_like(self.item_embeddings[0] if self.item_embeddings else data[0]['item_embedding']))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        
        # 如果有嵌入向量，转换为张量
        if self.has_embeddings:
            self.item_embeddings = torch.cat(self.item_embeddings, dim=0).contiguous()
            
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        
        # 保存原始数据的长度，用于调试
        self.data_length = len(data)

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        
        result = {
            'user': user,
            'item': item,
            'rating': rating,
            'seq': seq,
            'mask': mask
        }
        
        # 如果有嵌入向量，添加到结果中
        if self.has_embeddings:
            result['item_embedding'] = self.item_embeddings[index]
            
        return result


class Batchify2:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        t, self.feature, features = [], [], []
        for x in data:
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            intersection = ufea & ifea
            difference = ufea | ifea - intersection
            features.append(' '.join(list(intersection) + list(difference)))
            t.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        return seq, mask, prompt


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
