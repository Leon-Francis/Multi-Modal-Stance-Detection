import csv
import os
import json
import base64
import yaml
import random
import torch
import numpy as np
from PIL import Image

def read_yaml(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def set_seed(seed_num=42):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_csv(data_path):
    all_datas = []
    with open(data_path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

def save_csv(data_path, datas, mode='w'):
    field_names = datas[0].keys()
    if not os.path.exists(data_path) or mode == 'w':
        with open(data_path, 'w', encoding='utf-8') as write_f:
            writer = csv.DictWriter(write_f, fieldnames=field_names)
            writer.writeheader()

    with open(data_path, 'a', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        for data in datas:
            writer.writerow(data)

def save_dict(data_path, data, mode='w'):
    with open(data_path, mode, encoding='utf-8') as write_f:
        json.dump(data, write_f, indent=4)

def save_predict_labels(data_path, labels):
    with open(data_path, 'w', encoding='utf-8') as write_f:
        for label in labels:
            write_f.write(f'{label}\n')

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def save_results(results, data_path):
    true_labels, predict_labels = results
    assert len(true_labels) == len(predict_labels)
    write_results = []
    for idx in range(len(true_labels)):
        write_results.append({'true_label': true_labels[idx], 'predict_label': predict_labels[idx]})
    save_csv(data_path, write_results)