import inspect
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ClipModel(nn.Module):
    def __init__(self, args):
        super(ClipModel, self).__init__()
        self.args = args

        self.clip_config = AutoConfig.from_pretrained(args.multimodal_model_name)
        self.clip_model = AutoModel.from_pretrained(args.multimodal_model_name)

        if self.args.linear_injection == -1:
            linear_injection = self.clip_config.projection_dim
        else:
            linear_injection = self.args.linear_injection

        self.textual_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.clip_config.projection_dim, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.visual_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.clip_config.projection_dim, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection*2, args.label_size)

    def forward(self, input_data):
        multimodal_outputs = self.clip_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.clip_model.forward).parameters})
        # (B, H)
        textual_pooled_output = self.textual_linear(multimodal_outputs['text_embeds'])
        visual_pooled_output = self.visual_linear(multimodal_outputs['image_embeds'])

        logits = self.classifier(torch.cat([textual_pooled_output, visual_pooled_output], dim=-1))
        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        # model config
        multimodal_model_name = 'model_state/openai/clip-vit-base-patch32'

    import torch
    args = Args()
    model = ClipModel(args)
    text_ids = torch.randint(low=0, high=49407, size=[16, 77], dtype=torch.long)
    text_masks = torch.ones(size=[16, 77], dtype=torch.long)
    text_types = torch.zeros(size=[16, 77], dtype=torch.long)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'token_type_ids': text_types, 'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')