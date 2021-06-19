'''Train CIFAR100 with PyTorch.'''

import torch.backends.cudnn as cudnn
import os
import argparse
from model1 import *
from PIL import Image
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint1')
args = parser.parse_known_args()[0]
args.resume = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint1.
    print('==> Resuming from checkpoint1..')
    assert os.path.isdir(r'E:\PyCharmProjects\AI\ruanjianbei\checkpoint1'), 'Error: no checkpoint1 directory found!'
    checkpoint1 = torch.load(r'E:\PyCharmProjects\AI\ruanjianbei\checkpoint1\ckpt.pth')
    net.load_state_dict(checkpoint1['net'])
    best_acc = checkpoint1['acc']
    print(best_acc)
    start_epoch = checkpoint1['epoch']
    print(start_epoch)
def mymodel():
    return net
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def test(filepath):
    net.eval()
    img = Image.open(filepath)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    outputs = net(img)
    predict = torch.max(outputs, dim=1)[1].data
    print(classes[int(predict[0])])
    return classes[int(predict[0])]

if __name__ == "__main__":
    os.chdir('..')
    childfile = os.listdir()
    for child in childfile:
        if(child.split('.')[-1]=="png"):
            test(child)

