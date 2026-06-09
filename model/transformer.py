import torch
import torch.nn as nn

class Lightweight(nn.Module):
    """Lightweight transformer encoder for sequence-to-sequence vector regression.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Transformer hidden size.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout probability for encoder and projection layers.
        dim_feedforward: Feed-forward hidden size inside encoder layers.
    """

    def __init__(self, input_dim=14, d_model=32, nhead=2, num_layers=2, dropout=0.0, dim_feedforward=4096):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, 3)  # Output: x, y, z

    def forward(self, x, src_key_padding_mask=None):
        """Run a forward pass.

        Args:
            x: Input tensor shaped (batch, seq_len, input_dim).
            src_key_padding_mask: Optional bool mask shaped (batch, seq_len),
                where True marks padded positions.

        Returns:
            Tensor shaped (batch, seq_len, 3) for Bx/By/Bz predictions.
        """
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_dropout(x)
        x = self.decoder(x)
        return x  # Shape: (batch, seq_len, 3)
