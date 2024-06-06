import inspect
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiModalTransformerModel(nn.Module):
    def __init__(self, args):
        super(MultiModalTransformerModel, self).__init__()
        self.args = args

        self.textual_transformer_config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.textual_transformer_model = AutoModel.from_pretrained(args.textual_transformer_name, self.textual_transformer_config)

        self.visual_transformer_config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_transformer_model = AutoModel.from_pretrained(args.visual_transformer_name, self.visual_transformer_config)

        if self.args.linear_injection == -1:
            linear_injection = self.textual_transformer_config.hidden_size
        else:
            linear_injection = self.args.linear_injection

        self.textual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.textual_transformer_config.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.visual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.visual_transformer_config.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection*2, args.label_size)

    def forward(self, input_data):
        textual_outputs = self.textual_transformer_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.textual_transformer_model.forward).parameters})
        visual_outputs = self.visual_transformer_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.visual_transformer_model.forward).parameters})

        # (B, H)
        textual_pooled_output = self.textual_transformer_linear(textual_outputs['pooler_output'])
        visual_pooled_output = self.visual_transformer_linear(visual_outputs['pooler_output'])

        logits = self.classifier(torch.cat([textual_pooled_output, visual_pooled_output], dim=-1))
        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        # model config
        textual_transformer_tokenizer_name = 'model_state/vinai/bertweet-base'
        textual_transformer_name = 'model_state/vinai/bertweet-base'

        visual_transformer_tokenizer_name = 'model_state/google/vit-base-patch16-224'
        visual_transformer_name = 'model_state/google/vit-base-patch16-224'

    import torch
    args = Args()
    model = MultiModalTransformerModel(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.long)
    text_types = torch.zeros(size=[16, 128], dtype=torch.long)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'token_type_ids': text_types, 'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')