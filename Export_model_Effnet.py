import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
import random
import argparse
from argparse import Namespace, ArgumentParser
from typing import List, Optional
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from efficientnet_pytorch import EfficientNet
import wandb
# Настройка matplotlib для безголового режима
matplotlib.use('Agg')

# Настройки среды
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Подготовка данных
data_dir = '/content/drive/MyDrive/Colab Notebooks/МорозовДС_проект/car_data'
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomCrop(350),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std, inplace=True)
])

train = ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
batch_size = 10
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_transforms = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.CenterCrop(350),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val = ImageFolder(os.path.join(data_dir, 'test'), val_transforms)
val_dl = DataLoader(val, batch_size=batch_size, num_workers=0, pin_memory=True)

val_dataset = val
val_dataloader = val_dl

# Визуализация
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

def show_batch(dl):
    for images, _ in dl:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xticks([])
        ax.set_yticks([])
        imshow(make_grid(images[:10], nrow=5))
        plt.show()
        break

show_batch(train_dl)

# Альбументация
def pre_transforms():
    return [albu.Resize(400, 400, p=1),
            albu.PadIfNeeded(min_height=400, min_width=400, always_apply=True, border_mode=0)]

def hard_transforms():
    result = [
        albu.RandomCrop(352, 352, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
        ], p=0.9),
        albu.OneOf([
            albu.IAASharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        albu.IAAAdditiveGaussianNoise(p=0.2)
    ]
    return result

def post_transforms():
    return [albu.Normalize(), ToTensorV2()]

def compose(transforms_to_compose):
    return albu.Compose([item for sublist in transforms_to_compose for item in sublist])

# Визуализация Примеров
ROOT = Path("/content/drive/MyDrive/Colab Notebooks/МорозовДС_проект/car_data")
train_image_path = ROOT / "train/Audi S6 Sedan 2011/"
ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))

def show_examples(name: str, image: np.ndarray, image2: np.ndarray):
    plt.figure(figsize=(12, 16))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

def show(index: int, images: List[Path], transforms=None) -> None:
    image_path = images[index]
    image_path2 = images[index + 4]
    name = image_path.name
    image = np.array(Image.open(image_path))
    image2 = np.array(Image.open(image_path2))
    temp = transforms(image=image)
    temp2 = transforms(image=image2)
    image = temp["image"]
    image2 = temp2['image']
    show_examples(name, image, image2)

def show_random(images: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 5)
    show(index, images, transforms)

# Набор Данных
class ClassificationDataset(Dataset):
    def __init__(self, images: List[Path], transforms=None) -> None:
        self.images = images
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image_path2 = self.images[idx + 4]
        image = np.array(Image.open(image_path))
        image2 = np.array(Image.open(image_path2))
        result = {"image": image, 'image2': image2}
        result = self.transforms(**result)
        result["filename"] = image_path.name
        return result

show_transforms = compose([pre_transforms(), hard_transforms()])
show_random(ALL_IMAGES, transforms=show_transforms)

# Определение Модели
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def _make_trainable(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def _recursive_freeze(module: nn.Module, train_bn: bool = True) -> None:
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)

def freeze(module: nn.Module, n: Optional[int] = None, train_bn: bool = True) -> None:
    children = list(module.children())
    n_max = len(children) if n is None else int(n)
    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)
    for child in children[n_max:]:
        _make_trainable(module=child)

# Определение Модели EffNet
class EffNet(LightningModule):
    def __init__(self, num_target_classes=246, backbone: str = 'efficientnet-b7', batch_size: int = 8,
                 lr: float = 5e-4, wd: float = 0, num_workers: int = 4, factor: float = 0.5, **kwargs):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.num_workers = num_workers
        self.factor = factor
        self.save_hyperparameters()
        self.__build_model()
        self.train_dataset = None

    def __build_model(self):
        self.net = EfficientNet.from_pretrained(self.backbone)
        in_features = self.net._fc.in_features
        _fc_layers = [nn.Linear(in_features, self.num_target_classes)]
        self.net._fc = nn.Sequential(*_fc_layers)

    def setup(self, stage: str):
        data_dir = '/content/drive/MyDrive/Colab Notebooks/МорозовДС_проект/car_data'
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = (
            transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(350),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ]))
        train = ImageFolder(os.path.join(data_dir, 'train'), train_transforms)

        val_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val = ImageFolder(os.path.join(data_dir, 'test'), val_transforms)
        valid, _ = random_split(val, [len(val), 0])

        self.train_dataset = train
        self.val_dataset = valid

    def val_acc(self, y_pred, y_true):
        return accuracy(y_pred, y_true)

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        train_loss = F.cross_entropy(y_logits, y)
        return train_loss

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        val_loss = F.cross_entropy(y_logits, y)
        acc = self.val_acc(y_logits, y)
        return {'val_loss': val_loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([output['val_loss'] for output in outputs]).mean()
        val_acc_mean = torch.stack([output['val_acc'] for output in outputs]).mean()
        self.log('val_loss', val_loss_mean)
        self.log('val_acc', val_acc_mean)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, factor=self.factor, patience=2, mode='max'),
                        'name': 'learning_rate', 'monitor': 'val_acc'}
        return [optimizer], [lr_scheduler]

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)

# Командный Интерфейс
class EffNetCLI(EffNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone', type=str, default='efficientnet-b7', help='Backbone architecture')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
        parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
        parser.add_argument('--wd', type=float, default=0, help='Weight decay')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--factor', type=float, default=0.5, help='Factor for reducing learning rate')
        return parser

    @staticmethod
    def get_args() -> Namespace:
        parent_parser = ArgumentParser(add_help=False)
        parent_parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
        parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true', help='Use 16-bit precision')
        parent_parser.add_argument('--epochs', default=15, type=int, metavar='N', help='Total number of epochs', dest='nb_epochs')
        parent_parser.add_argument('--patience', default=3, type=int, metavar='ES', help='Early stopping patience', dest='patience')

        parser = EffNetCLI.add_model_specific_args(parent_parser)
        return parser.parse_args()

def main(args):
    seed_everything(42)
    model = EffNetCLI(**vars(args))
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    wandb_logger = WandbLogger(name='Name', project="Project")
    checkpoint_cb = ModelCheckpoint(dirpath='./', filename='cars-{epoch:02d}-{val_acc:.4f}', monitor='val_acc', mode='max', save_top_k=1)
    early = EarlyStopping(patience=5, monitor='val_acc', mode='max')

    trainer = Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        max_epochs=args.nb_epochs,
        deterministic=True,
        precision=16 if args.use_16bit else 32,
        callbacks=[checkpoint_cb, LearningRateMonitor(), early],
    )
    trainer.fit(model)

if __name__ == '__main__':
    args = EffNetCLI.get_args()
    main(args)
