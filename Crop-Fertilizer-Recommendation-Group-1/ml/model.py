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

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.feature_embedding = nn.Linear(
            1,
            embedding_dim
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                input_dim,
                embedding_dim
            )
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

        self.fusion = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch_size, 7]

        x = x.unsqueeze(-1)

        # [batch, 7, 1]

        x = self.feature_embedding(x)

        # [batch, 7, embedding_dim]

        x = x + self.position_embedding

        x = self.transformer(x)

        # Mean pooling

        x = x.mean(dim=1)

        x = self.fusion(x)

        output = self.classifier(x)

        return output