import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DetectionHead(nn.Module):
    """Predicts single box + class scores per grid cell (no anchors)"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, 5 + num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class FullModel(nn.Module):
    """Wraps the backbone and head for full model control and access"""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

class ModelBuilder:
    """Builds model from config file"""

    @staticmethod
    def build_layer(layer_config):
        """Creates a single layer from config"""
        layer_type = layer_config[0]

        if layer_type == "conv":
            in_c, out_c, k, s, p = layer_config[1:]
            return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        elif layer_type == "relu":
            return nn.ReLU()
        elif layer_type == "maxpool":
            return nn.MaxPool2d(kernel_size=layer_config[1])
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @classmethod
    def build_backbone(cls, config):
        """Constructs the feature extractor"""
        if config["type"] == "sequential":
            layers = [cls.build_layer(layer) for layer in config["layers"]]
            return nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported backbone type: {config['type']}")

    @classmethod
    def from_config(cls, config_path):
        """Full model builder entry point"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        backbone = cls.build_backbone(config["backbone"])
        head = DetectionHead(**config["head"]["args"])
        return FullModel(backbone, head)

# Helper functions (for testing/debugging)
def test_model_shapes(model, input_size=(3, 480, 480)):
    """Verifies input/output shapes"""
    summary(model, input_size=input_size)

def visualize_architecture(model):
    """Prints layer structure"""
    print(model)

if __name__ == "__main__":
    model = ModelBuilder.from_config("configs/simple_model.yaml")

    print("Model Architecture:")
    visualize_architecture(model)

    print("\nShape Verification:")
    test_model_shapes(model)