from utils.dataset_utils.dataset_configs.mtse_config import MTSEConfig
from utils.dataset_utils.dataset_configs.mccq_config import MCCQConfig
from utils.dataset_utils.dataset_configs.mwtwt_config import MWTWTConfig
from utils.dataset_utils.dataset_configs.mruc_config import MRUCConfig
from utils.dataset_utils.dataset_configs.mtwq_config import MTWQConfig

data_configs = {
    'mtse': MTSEConfig,
    'mccq': MCCQConfig,
    'mwtwt': MWTWTConfig,
    'mruc': MRUCConfig,
    'mtwq': MTWQConfig,
}

from utils.dataset_utils.textual_dataset import TextualDataset
from utils.dataset_utils.visual_dataset import VisualDataset
from utils.dataset_utils.multimodal_dataset import MultiModalDataset
from utils.dataset_utils.tmpt_dataset import TMPTDataset

datasets = {
    'textual': TextualDataset,
    'visual': VisualDataset,
    'multimodal': MultiModalDataset,
    'tmpt': TMPTDataset,
    'tmpt_gpt_cot': TMPTDataset,
}