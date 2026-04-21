from __future__ import annotations

import torch

from paraformer_v2 import ParaformerV2, ParaformerV2Config


def main() -> None:
    torch.manual_seed(7)
    config = ParaformerV2Config(
        input_dim=80,
        vocab_size=32,
        encoder_dim=64,
        decoder_dim=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ff_dim=128,
        decoder_ff_dim=128,
        attention_heads=4,
        dropout=0.0,
    )
    model = ParaformerV2(config)
    features = torch.randn(2, 96, config.input_dim)
    feature_lengths = torch.tensor([96, 88])
    targets = torch.tensor(
        [
            [1, 4, 9, 2, 0],
            [6, 3, 8, 0, 0],
        ],
        dtype=torch.long,
    )
    target_lengths = torch.tensor([4, 3])

    losses = model.loss(features, feature_lengths, targets, target_lengths)
    losses["loss"].backward()

    print(
        {
            "loss": round(float(losses["loss"].item()), 4),
            "ctc_loss": round(float(losses["ctc_loss"].item()), 4),
            "ce_loss": round(float(losses["ce_loss"].item()), 4),
        }
    )


if __name__ == "__main__":
    main()
