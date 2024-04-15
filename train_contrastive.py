import torch
import timm
import torchvision
from timm.data.auto_augment import auto_augment_transform
from torch.utils.data import DataLoader
from datasets import MY_CIFAR10, MY_CIFAR100, MY_STL10
import torch.optim as optim
from util import fast_kmeans, acc
from datasets import load_Cifar10, load_Cifar20, load_STL_10
from utils.loss import SimCLRLoss, AverageMeter, ProgressMeter
from utils.wideresnet import build_wideresnet
from sklearn.cluster import KMeans
from timm.data import create_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
out_dim = 200
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=out_dim).to(device)   #convnext_tiny

batch_size = 128

# 数据增强
tfm = auto_augment_transform(config_str = 'original',
hparams = {'translate_const':
100, 'img_mean': (124, 116, 104)})
class auto(object):
    def __init__(self, aa=1):
        self.aa = aa
    def __call__(self, img):
        return tfm(img)
transform = torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((96, 96)),
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
                                         ])
aug_transform = torchvision.transforms.Compose([
                                                torchvision.transforms.RandomResizedCrop(size=32, scale = (0.2,1.0)),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomApply([
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                                                ], p=0.8),
                                                torchvision.transforms.RandomGrayscale(p=0.2),
                                                torchvision.transforms.Resize((96, 96)),
                                                torchvision.transforms.ToTensor(),  # 转换成张量
                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
                                              ])
stl10_transform = torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 标准化
                                         ])
stl10_aug_transform = torchvision.transforms.Compose([
                                                torchvision.transforms.RandomResizedCrop(size=96, scale = (0.2,1.0)),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomApply([
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                                                ], p=0.8),
                                                torchvision.transforms.RandomGrayscale(p=0.2),
                                                torchvision.transforms.Resize((96, 96)),
                                                torchvision.transforms.ToTensor(),  # 转换成张量
                                                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 标准化
                                              ])
# train_data = MY_CIFAR10(root="./Dataset", train=True, download=False, transform=transform, aug_transform = aug_transform)
# test_data = MY_CIFAR10(root="./Dataset", train=False, download=False, transform=transform, aug_transform = aug_transform)
# data = train_data + test_data
# dataLoad = DataLoader(dataset=data, batch_size=batch_size, num_workers=6, shuffle=True)

# optimizer
optimizer = optim.Adam((model.parameters()),
                           lr=5e-7,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 256)

# loss_fn = torch.nn.MSELoss(reduction='mean')

model.train()
criterion = SimCLRLoss(0.7)
def pretext_model(dataset):
    if dataset == 'cifar10':
        train_data = MY_CIFAR10(root="./Dataset", train=True, download=False, transform=transform,
                                aug_transform=aug_transform)
        test_data = MY_CIFAR10(root="./Dataset", train=False, download=False, transform=transform,
                               aug_transform=aug_transform)
        num_cluster = 10
    elif dataset == 'cifar100':
        train_data = MY_CIFAR100(root="./Dataset", train=True, download=False, transform=transform,
                                aug_transform=aug_transform)
        test_data = MY_CIFAR100(root="./Dataset", train=False, download=False, transform=transform,
                               aug_transform=aug_transform)
        num_cluster = 100
    elif dataset == 'stl10':
        train_data = MY_STL10(root="./Dataset", split="train", download=False, transform=stl10_transform,
                                 aug_transform=stl10_aug_transform)
        test_data = MY_STL10(root="./Dataset", split="test", download=False, transform=stl10_transform,
                                aug_transform=stl10_aug_transform)
        num_cluster = 10
    data = train_data + test_data
    dataLoad = DataLoader(dataset=data, batch_size=batch_size, num_workers=6, shuffle=True)
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataLoad),
                             [losses],
                             prefix="Epoch: [{}]".format(10))
    print("pretext")
    for i in range(12):
        print(str(i+1))
        for i, batch in enumerate(dataLoad):
            img = batch[0]
            aug_img= batch[1]
            b, c, h, w = img.size()
            input_ = torch.cat([img.unsqueeze(1), aug_img.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            output = model(input_).view(b, 2, -1)
            loss = criterion(output)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 25 == 0:
                progress.display(i)
            scheduler.step()
        tmp, label, image = load_Cifar20(model)
        y = KMeans(n_clusters=num_cluster, random_state=0).fit_predict(tmp)



    return model