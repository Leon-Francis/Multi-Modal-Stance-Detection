import sys
sys.path.append('./')
import torch
import logging
from torch.utils.data import Dataset
from utils.social_media_texts_utils import split_hash_tag, clean_text
from utils.tools import load_csv, load_image

class MultiModalAutoDataset(Dataset):
    def __init__(self):
        super(MultiModalAutoDataset, self).__init__()
    
    def encode_datas(self, tokenizer, sentences, targets, images):
        textual_tokenizer, image_processor = tokenizer
        self.encodings = textual_tokenizer(sentences, targets, padding=True, truncation='only_first', return_tensors='pt')
        self.encodings.update(image_processor(images=images, return_tensors='pt'))
        self.encodings['classification_label'] = torch.tensor(self.labels)

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['classification_label'])

class MultiModalDataset(MultiModalAutoDataset):
    def __init__(self, dataset_config, tokenizer, target_name=None, if_split_hash_tag=True, in_target=None, zero_shot=None, train_data=None, valid_data=None, test_data=None, debug_mode=False):
        super(MultiModalAutoDataset, self).__init__()
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

        self.sentences, self.targets, self.images, self.labels = self.read_data(self.data_path, debug_mode)
        self.encode_datas(tokenizer, self.sentences, self.targets, self.images)
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
        images = []
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
            images.append(load_image(f"{self.dataset_config.data_dir}/{data['tweet_image']}"))
            labels.append(self.dataset_config.label2idx[data['stance_label']])
            label_num[data['stance_label']] += 1
        logging.info(f'loading data {len(sentences)} from {path}')
        logging.info(f'label num ' + ' '.join([f'{k}: {v}' for k,v in label_num.items()]))
        return sentences, targets, images, labels
    

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor
    from utils.dataset_utils.data_config import data_configs
    textual_tokenizer_name = 'model_state/openai/clip-vit-base-patch32'
    visual_processor_name = 'model_state/openai/clip-vit-base-patch32'
    processor = AutoProcessor.from_pretrained(visual_processor_name)
    if '/'.join(textual_tokenizer_name.split('/')[1:]) in processor.tokenizer.max_model_input_sizes:
        processor.tokenizer.model_max_length=processor.tokenizer.max_model_input_sizes['/'.join(textual_tokenizer_name.split('/')[1:])]

    MultiModalDataset(
        dataset_config=data_configs['mtse'],
        tokenizer=(processor.tokenizer, processor.image_processor),
        target_name=data_configs['mtse'].target_names[0],
        if_split_hash_tag=False,
        in_target=True,
        train_data=True,
        debug_mode=True
    )