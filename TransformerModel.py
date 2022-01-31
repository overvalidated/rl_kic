from torch import nn
import torch
import numpy as np


class TransformerExtractor(nn.Module):
    def __init__(self, features_dim=128):
        super().__init__()

        class HiddenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hours_processing = nn.Sequential(
                    nn.LayerNorm(52),
                    nn.Linear(in_features=52, out_features=features_dim),
                    nn.TransformerEncoder(
                        encoder_layer=nn.TransformerEncoderLayer(
                            features_dim,
                            8,
                            features_dim,
                            batch_first=True
                        ),
                        num_layers=1
                    )
                    # GTrXL(INTERMEDIATE_SIZE, 2, 2, hidden_dims=INTERMEDIATE_SIZE, batch_first=True)
                )

            def forward(self, x):
                return self.hours_processing(x)

        self.transformer = HiddenModel()
        self.layer_norm1 = nn.LayerNorm(42)
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
        )
        # self.act = nn.ReLU()

    def forward(self, observations: torch.Tensor, total_hours: torch.Tensor) -> torch.Tensor:
        # input shape is n_persons * 10 + n_persons + n_shifts
        target_hours = observations[:, -42:]
        placed_hours = observations[:, :-42].view(-1, N_PERSONS, 10)
        # attended = self.transformer(placed_hours)
        out = self.transformer(
            torch.concat([
                placed_hours,
                torch.repeat_interleave(self.layer_norm1(target_hours).reshape(-1, 1, 42), N_PERSONS, dim=1)
            ], dim=2))

        return self.policy_net(out).reshape(-1, N_PERSONS), None  # self.value_net(torch.mean(out, dim=1))
