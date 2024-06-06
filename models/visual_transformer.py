import inspect
from collections import OrderedDict
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class VisualTransformerModel(nn.Module):
    def __init__(self, args):
        super(VisualTransformerModel, self).__init__()
        self.args = args

        self.transformer_config = AutoConfig.from_pretrained(args.transformer_name)
        self.transformer_model = AutoModel.from_pretrained(args.transformer_name, self.transformer_config)

        if self.args.linear_injection == -1:
            linear_injection = self.transformer_config.hidden_size
        else:
            linear_injection = self.args.linear_injection

        self.transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.transformer_config.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection, args.label_size)

    def forward(self, input_data):
        visual_outputs = self.transformer_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.transformer_model.forward).parameters})
        # (B, H)
        visual_pooled_output = self.transformer_linear(visual_outputs['pooler_output'])

        logits = self.classifier(visual_pooled_output)

        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        # model config
        transformer_tokenizer_name = 'model_state/microsoft/swinv2-base-patch4-window12-192-22k'
        transformer_name = 'model_state/microsoft/swinv2-base-patch4-window12-192-22k'

    import torch
    args = Args()
    model = VisualTransformerModel(args)
    pixel_values = torch.randn(size=[16, 3, 224, 224])
    input_data = {'pixel_values': pixel_values}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')