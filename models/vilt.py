import inspect
from collections import OrderedDict
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ViltModel(nn.Module):
    def __init__(self, args):
        super(ViltModel, self).__init__()
        self.args = args

        self.vilt_config = AutoConfig.from_pretrained(args.multimodal_model_name)
        self.vilt_model = AutoModel.from_pretrained(args.multimodal_model_name)

        if self.args.linear_injection == -1:
            linear_injection = self.vilt_config.hidden_size
        else:
            linear_injection = self.args.linear_injection

        self.vilt_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.vilt_config.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection, args.label_size)

    def forward(self, input_data):
        multimodal_outputs = self.vilt_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.vilt_model.forward).parameters})
        # (B, H)
        multimodal_pooled_output = self.vilt_linear(multimodal_outputs['pooler_output'])

        logits = self.classifier(multimodal_pooled_output)
        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        # model config
        multimodal_model_name = 'model_state/dandelin/vilt-b32-mlm'

    import torch
    args = Args()
    model = ViltModel(args)
    text_ids = torch.randint(low=0, high=30522, size=[16, 40], dtype=torch.long)
    text_masks = torch.ones(size=[16, 40], dtype=torch.long)
    text_types = torch.zeros(size=[16, 40], dtype=torch.long)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'token_type_ids': text_types, 'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')