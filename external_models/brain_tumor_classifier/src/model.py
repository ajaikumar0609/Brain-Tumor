import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class BrainTumorEfficientNet(nn.Module):
    """
    EfficientNet-B3 backbone fine-tuned for grayscale brain-MRI classification.

    Two key adaptations over a stock EfficientNet:

    1. Grayscale input - the first Conv2d is replaced with a 1-channel version.
       Its weights are initialized by *averaging* the 3 pretrained RGB planes so
       the model retains its ImageNet edge/texture detectors from day one.

    2. Custom classifier head - Dropout(0.4) -> Linear(1536 -> num_classes).
       Dropout sits *before* the final linear layer.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # -- Grayscale adaptation -----------------------------------------
        old_conv = backbone.features[0][0]          # Conv2d(3, 40, ...)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Average RGB planes -> preserve pretrained spatial feature detectors (super helpful)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        # attach the new changes to the backbone
        backbone.features[0][0] = new_conv

        self.backbone = backbone

        # -- Custom classifier head -----------------------------------------
        in_features = self.backbone.classifier[1].in_features  # 1536 for B3
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze the feature extractor — only the head trains (Phase 1)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the full network for end-to-end fine-tuning (Phase 2)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
