import inspect
from collections import OrderedDict
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.args = args

        self.transformer_config = AutoConfig.from_pretrained(args.transformer_name)
        self.transformer_model = AutoModel.from_pretrained(args.transformer_name)

        if self.args.linear_injection == -1:
            linear_injection = self.transformer_config.hidden_sizes[-1]
        else:
            linear_injection = self.args.linear_injection

        self.transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.transformer_config.hidden_sizes[-1], linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection, args.label_size)

    def forward(self, input_data):
        visual_outputs = self.transformer_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.transformer_model.forward).parameters})
        # (B, H)
        visual_pooled_output = self.transformer_linear(visual_outputs['pooler_output'].squeeze(dim=-1).squeeze(dim=-1))

        logits = self.classifier(visual_pooled_output)

        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        # model config
        transformer_tokenizer_name = 'model_state/microsoft/resnet-50'
        transformer_name = 'model_state/microsoft/resnet-50'

    import torch
    args = Args()
    model = ResNetModel(args)
    pixel_values = torch.randn(size=[16, 3, 224, 224])
    input_data = {'pixel_values': pixel_values}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')