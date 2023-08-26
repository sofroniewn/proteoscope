import torch
import torch.nn as nn


class ESMBottleneck(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(ESMBottleneck, self).__init__()

        self.embedding = nn.Embedding(1, config.d_model)
        self.projection = nn.Linear(config.d_init, config.d_model)

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

    def forward(self, seqs, mask):
        # project
        projected = self.projection(seqs)

        # add guidance token
        guidance_token = self.embedding(
            torch.zeros((seqs.shape[0], 1), dtype=int, device=projected.device)
        )
        projected = torch.cat([guidance_token, projected], dim=1)
        guidance_mask = torch.ones((mask.shape[0], 1), dtype=bool, device=mask.device)
        mask = torch.cat([guidance_mask, mask], dim=1)

        # pass through transformer
        transformed = self.encoder(projected, src_key_padding_mask=~mask)

        # Take just guidence token
        return transformed[:, 0].unsqueeze(dim=1)
