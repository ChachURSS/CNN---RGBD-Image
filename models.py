import torch
import torchvision.models as models

class FeatureExtractorFactory:
    """Factory to instantiate different pre-trained architectures."""
    
    @staticmethod
    def get_extractor(model_name, device="cuda"):
        if model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
        elif model_name == "EfficientNet_B0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier = torch.nn.Identity()
            
        elif model_name == "ViT_B16":
            model = models.vision_transformer.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads = torch.nn.Identity()
            
        elif model_name == "DenseNet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = torch.nn.Identity()
        
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        model = model.to(device)
        model.eval()
        return model