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
        self.unconditioned_type = config.unconditioned_type

    def forward(self, seqs, mask, unconditioned=False):
        if unconditioned and self.unconditioned_type == 'token':
            guidance_token = self.guidance_embedding(
                torch.zeros((seqs.shape[0], 1), dtype=int, device=seqs.device)
            )
            mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
            return guidance_token, mask
        elif unconditioned and self.unconditioned_type == 'zeros':
            seqs = torch.zeros((seqs.shape[0], 1), dtype=float, device=seqs.device)
            mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
            return seqs, mask
        elif unconditioned:
            raise ValueError(f'Unrecognized unconditioned type {self.unconditioned_type}')

        if self.projection is not None:
            # project
            seqs = self.projection(seqs)

        if self.return_type == 'token':
            # add guidance token
            guidance_token = self.guidance_embedding(
                torch.zeros((seqs.shape[0], 1), dtype=int, device=seqs.device)
            )
            seqs = torch.cat([guidance_token, seqs], dim=1)
            guidance_mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
            mask = torch.cat([guidance_mask, mask], dim=1)

        if self.encoder is not None:
            # pass through transformer
            seqs = self.encoder(seqs, src_key_padding_mask=~mask)

        if self.return_type == 'token':
            # Take just guidence token
            mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
            return seqs[:, 0].unsqueeze(dim=1), mask
        elif self.return_type == 'mean':
            # Take the mean
            mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
            return seqs.mean(dim=1, keepdim=True), mask
        elif self.return_type == 'full':
            # Return full sequence
            return seqs, mask
        else:
            raise ValueError(f'Unrecognized return type {self.return_type}')
