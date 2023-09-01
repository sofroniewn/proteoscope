import torch
import torch.nn as nn


class ESMBottleneck(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(ESMBottleneck, self).__init__()

        self.guidance_embedding = nn.Embedding(1, config.d_model)

        if config.d_model > 0:
            self.projection = nn.Linear(config.d_init, config.d_model)
        else:
            self.projection = None

        if config.num_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=config.dropout,
                dim_feedforward=config.dim_feedforward,
                batch_first=True,
                norm_first=True,
            )
            encoder_norm = nn.LayerNorm(config.d_model)
            self.encoder = nn.TransformerEncoder(
                encoder_layer, config.num_encoder_layers, encoder_norm
            )
        else:
            self.encoder = None

        self.return_type = config.return_type

    def forward(self, seqs, mask, unconditioned=None):
        batch_size = seqs.shape[0]
        device = seqs.device

        if self.projection is not None:
            # project
            seqs = self.projection(seqs)

        if self.return_type == 'token':
            # add guidance token
            guidance_token = self.guidance_embedding(
                torch.zeros((batch_size, 1), dtype=int, device=device)
            )
            seqs = torch.cat([guidance_token, seqs], dim=1)
            guidance_mask = torch.ones((batch_size, 1), dtype=bool, device=device)
            mask = torch.cat([guidance_mask, mask], dim=1)

        if self.encoder is not None:
            # pass through transformer
            seqs = self.encoder(seqs, src_key_padding_mask=~mask)

        if self.return_type == 'token':
            # Take just guidence token
            mask = torch.ones((batch_size, 1), dtype=bool, device=device)
            seqs = seqs[:, 0].unsqueeze(dim=1)
        elif self.return_type == 'mean':
            # Take the mean
            mask = torch.ones((batch_size, 1), dtype=bool, device=device)
            seqs = seqs.mean(dim=1, keepdim=True)
        elif self.return_type == 'full':
            # Return full sequence
            pass
        else:
            raise ValueError(f'Unrecognized return type {self.return_type}')

        if unconditioned is not None:
            index = torch.rand(batch_size) < unconditioned
            seqs[index] = torch.zeros_like(seqs[index])
            mask[index] = torch.ones_like(mask[index])
        
        return seqs, mask
