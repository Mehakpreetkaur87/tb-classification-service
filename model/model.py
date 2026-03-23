import torch
import torch.nn as nn

class CheXagentSigLIPBinary(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()

        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.pooling_attn_weights = None  

        self.vision_encoder.head.attention.register_forward_hook(
            self._pooling_attn_hook
        )
       

        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_dim ,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def _pooling_attn_hook(self, module, input, output):
        # output = (attn_output, attn_weights)
        self.pooling_attn_weights = output[1]  

    def forward(self, inputs):
        outputs = self.vision_encoder(pixel_values=inputs,output_attentions = True,return_dict = True)
        attention = outputs.attentions
        embeddings = outputs.pooler_output   
        logits = self.classifier(embeddings)     
        return logits,attention,self.pooling_attn_weights