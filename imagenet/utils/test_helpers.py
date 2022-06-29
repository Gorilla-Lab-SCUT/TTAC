import torch
from model.resnet import SupCEResNet

def build_model():
    print("Building ResNet50...")
    model = SupCEResNet().cuda()
    ext = model.encoder
    classifier = model.fc
    return model, ext, classifier


def test(dataloader, model, **kwargs):
    model.eval()
    correct = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if type(inputs) == list:
            inputs = inputs[0]
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs, **kwargs)
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    model.train()
    return 1-correct.mean(), correct
