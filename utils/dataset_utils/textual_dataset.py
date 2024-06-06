import sys
sys.path.append('./')
import logging
import torch
from torch.utils.data import Dataset
from utils.social_media_texts_utils import split_hash_tag, clean_text
from utils.tools import load_csv

class TextualAutoDataset(Dataset):
    def __init__(self):
        super(TextualAutoDataset, self).__init__()

    def encode_datas(self, tokenizer, sentences, targets):
        self.encodings = tokenizer(sentences, targets, padding=True, truncation='only_first', return_tensors='pt')
        self.encodings['classification_label'] = torch.tensor(self.labels)

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['classification_label'])

class TextualDataset(TextualAutoDataset):
    def __init__(self, dataset_config, tokenizer, target_name=None, if_split_hash_tag=True, in_target=None, zero_shot=None, train_data=None, valid_data=None, test_data=None, debug_mode=False):
        super(TextualAutoDataset, self).__init__()
        assert train_data or valid_data or test_data
        assert in_target or zero_shot
        assert not (in_target and zero_shot)
        self.target_name = target_name
        self.if_split_hash_tag = if_split_hash_tag
        self.dataset_config = dataset_config
        
        if in_target:
            if train_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{dataset_config.short_target_names[target_name]}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{dataset_config.short_target_names[target_name]}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{dataset_config.short_target_names[target_name]}/test.csv'
            
        elif zero_shot:
            if train_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{dataset_config.short_target_names[target_name]}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{dataset_config.short_target_names[target_name]}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{dataset_config.short_target_names[target_name]}/test.csv'

        self.sentences, self.targets, self.labels = self.read_data(self.data_path, debug_mode)
        self.encode_datas(tokenizer, self.sentences, self.targets)
        logging.info(f'{dataset_config.dataset_name} textual loading finished')

    def apply_cleaning(self, sentence):
        if self.if_split_hash_tag:
            sentence = split_hash_tag(sentence.lstrip().rstrip())
        else:
            sentence = sentence.lstrip().rstrip()
        if self.dataset_config.apply_cleaning:
            sentence = clean_text(sentence)
        return sentence

    def read_data(self, path, debug_mode=False):
        sentences = []
        targets = []
        labels = []
        label_num = {}
        for label_name in self.dataset_config.label2idx.keys():
            label_num[label_name] = 0
        all_datas = load_csv(path) 

        if debug_mode:
            all_datas = all_datas[:200]
        for data in all_datas:
            sentences.append(self.apply_cleaning(data['tweet_text']))
            targets.append(data['stance_target'])
            labels.append(self.dataset_config.label2idx[data['stance_label']])
            label_num[data['stance_label']] += 1
        logging.info(f'loading data {len(sentences)} from {path}')
        logging.info(f'label num ' + ' '.join([f'{k}: {v}' for k,v in label_num.items()]))
        return sentences, targets, labels
    

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from utils.dataset_utils.data_config import data_configs
    transformer_tokenizer_name = 'model_state/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_name)
    tokenizer.model_max_length=tokenizer.max_model_input_sizes['/'.join(transformer_tokenizer_name.split('/')[1:])]

    TextualDataset(
        dataset_config=data_configs['mtse'],
        tokenizer=tokenizer,
        target_name=data_configs['mtse'].target_names[0],
        if_split_hash_tag=False,
        zero_shot=True,
        train_data=True,
        debug_mode=True
    )