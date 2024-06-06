import inspect
from collections import OrderedDict
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextualTransformerModel(nn.Module):
    def __init__(self, args):
        super(TextualTransformerModel, self).__init__()
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
        textual_outputs = self.transformer_model(**{k: v for k, v in input_data.items() if k in inspect.signature(self.transformer_model.forward).parameters})
        # (B, H)
        textual_pooled_output = self.transformer_linear(textual_outputs['pooler_output'])

        logits = self.classifier(textual_pooled_output)

        return logits


if __name__ == '__main__':
    class Args():
        max_tokenization_length = 128
        label_size = 2
        linear_injection = -1
        # model config
        transformer_tokenizer_name = 'model_state/vinai/bertweet-base'
        transformer_name = 'model_state/vinai/bertweet-base'

    import torch
    args = Args()
    model = TextualTransformerModel(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.long)
    text_types = torch.zeros(size=[16, 128], dtype=torch.long)
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'token_type_ids': text_types}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')