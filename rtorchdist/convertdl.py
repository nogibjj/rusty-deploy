import torch
from torchvision.models import resnet34
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.resnet import ResNet, ResNet34_Weights, BasicBlock


model_pt = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
model_pt.eval()
scripted_model = torch.jit.script(model_pt)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model.save("model/resnet34.ot")
