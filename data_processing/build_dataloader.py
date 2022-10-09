from data_processing.Gaussian_Blur import GaussianBlur
from data_processing.Coseg_Dataset import Coseg_Dataset
from data_processing.Path_Mapper import Path_Mapper
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def build_dataloader(data_path,params):
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD = [0.229, 0.224, 0.225]
    image_size = 224
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])
    dataset = Coseg_Dataset(data_path, params, transform_train, params['window_length'])
    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'],
                        shuffle=True, num_workers=params['num_workers'], drop_last=True)
    return loader


def build_infer_loader(data_path,params):
    image_size = 224
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD = [0.229, 0.224, 0.225]
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])

    dataset = Path_Mapper(data_path, transform=transform_eval)
    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'],
                        shuffle=False, num_workers=params['num_workers'], drop_last=False)
    return loader
