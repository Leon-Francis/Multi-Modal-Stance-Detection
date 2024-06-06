import sys
sys.path.append('./')
from collections import OrderedDict
import torch
import torch.nn as nn
from models.tmpt.tmpt_textual_model import TMPTTextualModel
from models.tmpt.tmpt_visual_model import TMPTVisualModel

class TMPTModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tmpt_textual_model = TMPTTextualModel(args)
        self.tmpt_visual_model = TMPTVisualModel(args)
        for k, p in self.tmpt_visual_model.named_parameters():
            if "prompt" not in k and "pooler" not in k:
                p.requires_grad = False

        if args.linear_injection == -1:
            linear_injection = min(self.tmpt_textual_model.hidden_size, self.tmpt_visual_model.hidden_size)
        else:
            linear_injection = args.linear_injection

        self.textual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_textual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.visual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_visual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection*2, args.label_size)


    def forward(self, input_data):
        textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])

        logits = self.classifier(torch.cat([text_pooled_output, image_pooled_output], dim=-1))
        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        prompt_dropout = 0.2
        visual_soft_tokens = 5
        visual_soft_prompt_dropout = 0.2
        # model config
        textual_transformer_tokenizer_name = 'model_state/vinai/bertweet-base'
        textual_transformer_name = 'model_state/vinai/bertweet-base'

        visual_transformer_tokenizer_name = 'model_state/google/vit-base-patch16-224'
        visual_transformer_name = 'model_state/google/vit-base-patch16-224'

    import torch
    args = Args()
    model = TMPTModel(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.long)
    text_loss_ids = torch.zeros(size=[16, 512], dtype=torch.long)
    text_loss_ids[:, 100] = 1
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'text_loss_ids': text_loss_ids, 'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')