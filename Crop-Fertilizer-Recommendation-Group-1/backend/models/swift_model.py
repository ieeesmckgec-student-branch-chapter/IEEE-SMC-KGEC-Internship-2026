import torch
import torch.nn as nn


class SwiFT(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(
                embedding_dim,
                num_classes
            )
        )

    def forward(self, x):

        x = self.embedding(x)

        # Add sequence dimension
        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x[:, 0, :]

        x = self.classifier(x)

        return x