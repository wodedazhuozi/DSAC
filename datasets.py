import os
import pickle
import numpy as np
import timm
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn import datasets
from timm.data.auto_augment import auto_augment_transform
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from util import LR



batch_size = 128  # 一次训练的样本数目
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

tfm = auto_augment_transform(config_str = 'original',
hparams = {'translate_const':
100, 'img_mean': (124, 116, 104)})
class auto(object):
    def __init__(self, aa=1):
        self.aa = aa
    def __call__(self, img):
        return tfm(img)

def loadMnist():
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=10).to(device)  # convnext_tiny
    trainSet = torchvision.datasets.MNIST(root="./Dataset", train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.Resize((96, 96)),
                                              torchvision.transforms.ToTensor(),  # 转换成张量
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                          ]))
    train_loader = DataLoader(trainSet, batch_size=batch_size, num_workers=6, shuffle=True)

    # 获取测试集
    testSet = torchvision.datasets.MNIST(root="./Dataset", train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize((96, 96)),
                                             torchvision.transforms.ToTensor(),  # 转换成张量
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                         ]))
    test_loader = DataLoader(testSet, batch_size=batch_size, num_workers=6, shuffle=True)

    # 加载数据
    for batch_idx, (image, target) in enumerate(train_loader):
        if batch_idx == 0:
            image = image.expand(-1, 3, -1, -1)
            image1 = image.cpu().detach().numpy()
        else:
            image = image.expand(-1, 3, -1, -1)
            image1 = np.vstack((image1, image.cpu().detach().numpy()))


        image = image.to(device)
        if batch_idx == 0:
            tmp = model(image).cpu().detach().numpy()
            label = target.cpu().numpy()
        else:
            tmp1 = model(image).cpu().detach().numpy()
            tmp = np.vstack((tmp, tmp1))
            label = np.hstack((label, target.cpu().numpy()))
        torch.cuda.empty_cache()
        del image, target
    # a = torch.unsqueeze(image1[2], dim = 0)
    for batch_idx, (image, target) in enumerate(test_loader):
        image = image.expand(-1, 3, -1, -1)
        image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        tmp1 = model(image).cpu().detach().numpy()
        tmp = np.vstack((tmp, tmp1))
        label = np.hstack((label, target.cpu().numpy()))


    return tmp, label, image1

def loadFashionMnist():
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=10).to(device)  # convnext_tiny
    trainSet = torchvision.datasets.FashionMNIST(root="./Dataset", train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.Resize((96, 96)),
                                              torchvision.transforms.ToTensor(),  # 转换成张量
                                              torchvision.transforms.Normalize((0.2860,), (0.3530,))  # 标准化
                                          ]))
    train_loader = DataLoader(trainSet, batch_size=batch_size, num_workers=6, shuffle=True)

    # 获取测试集
    testSet = torchvision.datasets.FashionMNIST(root="./Dataset", train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize((96, 96)),
                                             torchvision.transforms.ToTensor(),  # 转换成张量
                                             torchvision.transforms.Normalize((0.2860,), (0.3530,))  # 标准化
                                         ]))
    test_loader = DataLoader(testSet, batch_size=batch_size, num_workers=6, shuffle=True)

    # 加载数据
    for batch_idx, (image, target) in enumerate(train_loader):
        if batch_idx == 0:
            image = image.expand(-1, 3, -1, -1)
            image1 = image.cpu().detach().numpy()
        else:
            image = image.expand(-1, 3, -1, -1)
            image1 = np.vstack((image1, image.cpu().detach().numpy()))


        image = image.to(device)
        if batch_idx == 0:
            tmp = model(image).cpu().detach().numpy()
            label = target.cpu().numpy()
        else:
            tmp1 = model(image).cpu().detach().numpy()
            tmp = np.vstack((tmp, tmp1))
            label = np.hstack((label, target.cpu().numpy()))
        torch.cuda.empty_cache()
        del image, target
    # a = torch.unsqueeze(image1[2], dim = 0)
    for batch_idx, (image, target) in enumerate(test_loader):
        image = image.expand(-1, 3, -1, -1)
        image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        tmp1 = model(image).cpu().detach().numpy()
        tmp = np.vstack((tmp, tmp1))
        label = np.hstack((label, target.cpu().numpy()))


    return tmp, label, image1

def load_Cifar10(model):
    model1 = LR(200, 10).to(device)
    transform = torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((96, 96)),
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
                                         ])
    aug_transform = torchvision.transforms.Compose([auto()])
    train_data = torchvision.datasets.CIFAR10(root="./Dataset", train=True, download=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize((96, 96)),
                                                   torchvision.transforms.ToTensor(),  # 转换成张量
                                                   torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
                                               ]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=6, shuffle=True)

    test_data = torchvision.datasets.CIFAR10(root="./Dataset", train=False, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((96, 96)),
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
                                              ]))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=6, shuffle=True)

    # 加载数据
    step = 0
    for image, label1 in train_loader:
        if step == 0:
            image1 = image.cpu().detach().numpy()
        else:
            image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        if step == 0:
            # tmp = torch.squeeze(model(image.to(device)).detach().numpy()).cpu()
            tmp = model(image).cpu().detach().numpy()
            tmp_10 = model1(torch.from_numpy(tmp).to(device)).cpu().detach().numpy()
            # predict = torch.softmax(tmp, dim=0).cpu().detach().numpy()
            # predict_cla = np.argmax(predict, axis=1)
            label = label1.cpu().numpy()
        else:
            tmp1 = model(image).cpu().detach().numpy()
            tmp_10_1 = model1(torch.from_numpy(tmp1).to(device)).cpu().detach().numpy()
            tmp = np.vstack((tmp, tmp1))
            tmp_10 = np.vstack((tmp_10, tmp_10_1))
            label = np.hstack((label, label1.cpu().numpy()))
        step = step + 1
    for image, label1 in test_loader:
        image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        tmp1 = model(image).cpu().detach().numpy()
        tmp_10_1 = model1(torch.from_numpy(tmp1).to(device)).cpu().detach().numpy()
        tmp = np.vstack((tmp, tmp1))
        tmp_10 = np.vstack((tmp_10, tmp_10_1))
        label = np.hstack((label, label1.cpu().numpy()))

    return tmp, tmp_10, label, image1

def load_STL_10(model):
    # model = timm.create_model('convnext_tiny', pretrained=True, num_classes=10).to(device)  # convnext_tiny
    # model = WideResNet(28, 10, 2).to(device)  # WRN
    train_data = torchvision.datasets.STL10(root="./Dataset", split='train', download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                                                  # 标准化
                                              ]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=6, shuffle=True)

    test_data = torchvision.datasets.STL10(root="./Dataset", split='test', download=False,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),  # 转换成张量
                                                 torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                                                 # 标准化
                                             ]))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=6, shuffle=True)

    # 加载数据
    step = 0
    for image, label1 in train_loader:
        if step == 0:
            image1 = image.cpu().detach().numpy()
        else:
            image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        if step == 0:
            tmp = model(image).cpu().detach().numpy()
            label = label1.cpu().numpy()
        else:
            tmp1 = model(image).cpu().detach().numpy()
            tmp = np.vstack((tmp, tmp1))
            label = np.hstack((label, label1.cpu().numpy()))
        step = step + 1
    for image, label1 in test_loader:
        image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        tmp1 = model(image).cpu().detach().numpy()
        tmp = np.vstack((tmp, tmp1))
        label = np.hstack((label, label1.cpu().numpy()))

    return tmp, label, image1

def load_Cifar20(model):
    # model = timm.create_model('convnext_tiny', pretrained=True, num_classes=100).to(device)  # convnext_tiny
    train_data = torchvision.datasets.CIFAR100(root="./Dataset", train=True, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((96, 96)),
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))  # 标准化
                                         ]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=6, shuffle=True)

    test_data = torchvision.datasets.CIFAR100(root="./Dataset", train=False, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((96, 96)),
                                                  torchvision.transforms.ToTensor(),  # 转换成张量
                                                  torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)) # 标准化
                                              ]))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=6, shuffle=True)

    # 加载数据
    step = 0
    for image, label1 in train_loader:
        if step == 0:
            image1 = image.cpu().detach().numpy()
        else:
            image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        if step == 0:
            tmp = model(image).cpu().detach().numpy()
            label = label1.cpu().numpy()
        else:
            tmp1 = model(image).cpu().detach().numpy()
            tmp = np.vstack((tmp, tmp1))
            label = np.hstack((label, label1.cpu().numpy()))
        step = step + 1
    for image, label1 in test_loader:
        image1 = np.vstack((image1, image.cpu().detach().numpy()))
        image = image.to(device)
        tmp1 = model(image).cpu().detach().numpy()
        tmp = np.vstack((tmp, tmp1))
        label = np.hstack((label, label1.cpu().numpy()))

    # label = [cifar100_to_cifar20(i) for i in label]
    # label = np.array(label)

    return tmp, label, image1

class MY_STL10(Dataset):
    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(self, root, split = "train", folds=None, transform=None, aug_transform=None, target_transform=None,
                 download: bool = False):
        self.root = root
        self.split = split
        self.folds = folds
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_transform = target_transform

        if download:
            self.download()

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])

        elif self.split == 'labeled':
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            test_data, test_labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])
            self.data = np.concatenate((self.data, test_data))
            self.labels = np.concatenate((self.labels, test_labels))
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def __load_folds(self, folds):
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=' ')
            self.data, self.labels = self.data[list_idx, :, :, :], self.labels[list_idx]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img1 = self.transform(img)

        if self.aug_transform is not None:
            # img = img.numpy()
            # img = np.transpose(img,(1,2,0))
            # img = Image.fromarray(img)
            aug_img = self.aug_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, aug_img, target

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def __len__(self) -> int:
        return len(self.data)


class MY_CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    # def __init__(self, root=None, train=True, transform=None,
    #              download=False):
    def __init__(self, root, train: bool = True, transform = None, aug_transform = None, target_transform = None, download: bool = False):

        super(MY_CIFAR10, self).__init__()
        self.root = root
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.aug_transform is not None:
            # img = img.numpy()
            # img = np.transpose(img,(1,2,0))
            # img = Image.fromarray(img)
            aug_img = self.aug_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, aug_img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class MY_CIFAR100(Dataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    def __init__(self, root, train: bool = True, transform=None, aug_transform=None, target_transform=None, download=False):

        super(MY_CIFAR100, self).__init__()
        self.root = root
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.aug_transform is not None:
            # img = img.numpy()
            # img = np.transpose(img,(1,2,0))
            # img = Image.fromarray(img)
            aug_img = self.aug_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, aug_img, target

    def __len__(self) -> int:
        return len(self.data)

def cifar100_to_cifar20(target):
    """
    CIFAR100 to CIFAR 20 dictionary.
    This function is from IIC github.
    """

    class_dict = {0: 4,
                  1: 1,
                  2: 14,
                  3: 8,
                  4: 0,
                  5: 6,
                  6: 7,
                  7: 7,
                  8: 18,
                  9: 3,
                  10: 3,
                  11: 14,
                  12: 9,
                  13: 18,
                  14: 7,
                  15: 11,
                  16: 3,
                  17: 9,
                  18: 7,
                  19: 11,
                  20: 6,
                  21: 11,
                  22: 5,
                  23: 10,
                  24: 7,
                  25: 6,
                  26: 13,
                  27: 15,
                  28: 3,
                  29: 15,
                  30: 0,
                  31: 11,
                  32: 1,
                  33: 10,
                  34: 12,
                  35: 14,
                  36: 16,
                  37: 9,
                  38: 11,
                  39: 5,
                  40: 5,
                  41: 19,
                  42: 8,
                  43: 8,
                  44: 15,
                  45: 13,
                  46: 14,
                  47: 17,
                  48: 18,
                  49: 10,
                  50: 16,
                  51: 4,
                  52: 17,
                  53: 4,
                  54: 2,
                  55: 0,
                  56: 17,
                  57: 4,
                  58: 18,
                  59: 17,
                  60: 10,
                  61: 3,
                  62: 2,
                  63: 12,
                  64: 12,
                  65: 16,
                  66: 12,
                  67: 1,
                  68: 9,
                  69: 19,
                  70: 2,
                  71: 10,
                  72: 0,
                  73: 1,
                  74: 16,
                  75: 12,
                  76: 9,
                  77: 13,
                  78: 15,
                  79: 13,
                  80: 16,
                  81: 19,
                  82: 2,
                  83: 4,
                  84: 6,
                  85: 19,
                  86: 5,
                  87: 5,
                  88: 8,
                  89: 19,
                  90: 18,
                  91: 1,
                  92: 2,
                  93: 15,
                  94: 6,
                  95: 0,
                  96: 17,
                  97: 8,
                  98: 14,
                  99: 13}

    return class_dict[target]