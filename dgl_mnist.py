from pytorch_models.mnist_pytorch import ResNet

model = ResNet((1, 3, 28, 28))
modules = list(model.resnet.children())
modules.append(model.fc)
