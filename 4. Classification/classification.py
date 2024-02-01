from typing import Any
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision
from albumentations.pytorch import ToTensorV2
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

CLASSES_CNT = 50
VAL_RATE = 0.2

train_transform = A.Compose([A.RandomResizedCrop(256, 256, scale = (0.8, 1.0)),
                                           A.ToRGB(),
                                           A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
                                           A.Normalize(),
                                           A.Rotate(limit=30, p = 0.5),
                                           A.HorizontalFlip(p = 0.5),
                                           ToTensorV2()])
val_transform = A.ReplayCompose([A.Resize(256, 256), A.ToRGB(), A.Normalize(), ToTensorV2()])

def vis(img_, gt_ = None):
    fig, ax = plt.subplots()
    plt.imshow(img_.permute(1, 2, 0).numpy())
    plt.show()

class BirdsDataset(Dataset):
    def __init__(self, imgs_path, gt_csv = None, transform = None, custom_len = None, filenames = None):
        super().__init__()
        self.imgs_path = imgs_path
        self.custom_len = custom_len
        self.gt_csv = gt_csv
        if filenames is None:
            self.list_img_names = os.listdir(imgs_path)
        else:
            self.list_img_names = filenames
        self.transform = transform

    def __len__(self):
        if self.custom_len is None:
            return len(self.list_img_names)
        return self.custom_len


    def __getitem__(self, idx):
        img_name_ = self.list_img_names[idx]
        img_ = np.array(Image.open(os.path.join(self.imgs_path, img_name_)))
        if self.gt_csv is not None:
            gt_ = self.gt_csv[img_name_]
            transformed_img = self.transform(image=img_)
            img_trans_  =  transformed_img['image'].float()
            return img_trans_, gt_
        transformed_img_ = self.transform(image=img_)
        return transformed_img_['image'].float(), img_name_

class BirdsDataloader(pl.LightningModule):
    def __init__(self, train_img_dir, train_gt, test_img_dir = None, test_gt = None):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_gt = train_gt
        self.test_img_dir = test_img_dir
        self.test_gt = test_gt

    def train_dataloader(self):
        train_dataset = BirdsDataset(self.train_img_dir, self.train_gt, transform=train_transform)
        return DataLoader(train_dataset, batch_size = 64, shuffle=True,  num_workers = 2)

    def val_dataloader(self):
        val_dataset = BirdsDataset(self.test_img_dir, self.test_gt, transform= val_transform)
        return DataLoader(val_dataset, batch_size = 64,  num_workers = 2)

class BirdsTrainSplittingDataloader(pl.LightningModule):
    def __init__(self, img_dir, gt):
        super().__init__()
        self.img_dir = img_dir
        self.gt = gt
        gt_path_buff = pd.read_csv(gt).set_index('filename').T.to_dict('list')
        file_to_gt = {path: class_[0] for path, class_ in gt_path_buff.items()}
        images = os.listdir(img_dir)
        gt_to_file = [[] for i in range(CLASSES_CNT)]
        for filename_, class_ in file_to_gt.items():
            gt_to_file[class_].append(filename_)
        self.train_filenames_, self.test_filenames_ = [], []
        for i in range(CLASSES_CNT):
             test_size = int(len(gt_to_file[i]) * VAL_RATE)
             self.test_filenames_.extend(gt_to_file[i][:test_size])
             self.train_filenames_.extend(gt_to_file[i][test_size:])

    def train_dataloader(self):
        train_dataset = BirdsDataset(self.img_dir, self.gt, transform=train_transform, filenames = self.train_filenames_)
        return DataLoader(train_dataset, batch_size = 64, shuffle=True)

    def val_dataloader(self):
        val_dataset = BirdsDataset(self.img_dir, self.gt, transform= val_transform, filenames = self.test_filenames_)
        return DataLoader(val_dataset, batch_size = 64)

class Resnet50_based_model(pl.LightningModule):
    def __init__(self, features_criterion = None, internal_features = 128, is_init = False):
        super().__init__()
        self.features_criterion = features_criterion
        if is_init:
            self.res_model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.res_model = torchvision.models.resnet50(weights = None)
        linear_size = list(self.res_model.children())[-1].in_features
        self.res_model.fc = torch.nn.Linear(linear_size, internal_features)
        self.batch_norm_l = torch.nn.BatchNorm1d(internal_features)
        self.relu_ = torch.nn.ReLU()

        self.fc_last_ = torch.nn.Linear(internal_features, CLASSES_CNT)
        for child in list(self.res_model.children()):
            for param in child.parameters():
                param.requires_grad = True

        freeze = "most"
        if freeze == "most":
            for child in list(self.res_model.children())[:-3]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.fc_last_(self.relu_(self.batch_norm_l(self.res_model(x))))


def time_decay(step):
    k = 0.001
    lrate = 1 / (1 + k * step)
    return lrate

class CustomNetwork(pl.LightningModule):
    def __init__(self, features_criterion = None, internal_features = 512, is_init = False):
        super(CustomNetwork, self).__init__()
        self.features_criterion = features_criterion
        self.model = Resnet50_based_model(features_criterion = features_criterion, internal_features = internal_features, is_init = is_init)
        self.lr_rate = 1e-3
    def predict(self, x):
        with torch.no_grad():
            y_pred = torch.argmax(self.forward(x), dim = 1)
        return y_pred

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        X, y_true = train_batch
        f_pred = self.forward(X)
        y_pred = F.log_softmax(f_pred, dim = 1)
        loss = F.nll_loss(y_pred, y_true)
        if self.features_criterion is not None:
            loss += self.features_criterion(y_true, f_pred)
        acc = torch.sum(y_pred.argmax(axis=1) == y_true) / y_true.shape[0]
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y_true = val_batch
        f_pred = self.forward(X)
        y_pred = F.log_softmax(f_pred, dim = 1)
        loss = F.nll_loss(y_pred, y_true)
        if self.features_criterion is not None:
            loss += self.features_criterion(y_true, f_pred)
        acc = torch.sum(y_pred.argmax(axis=1) == y_true) / y_true.shape[0]
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=1e-3)

        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=time_decay,
            ),
            "interval": "step",
            "frequency": 1,
            "monitor": "val_acc",
            "strict": True,
            "name": None,
        }

        return [optimizer], [lr_dict]

def train_classifier(train_gt, train_img_dir, fast_train = True, test_gt = None, test_img_dir = None):
    if fast_train:
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            default_root_dir ='./',
            enable_checkpointing=False,
            logger=False,
        )

        model = CustomNetwork()
        faces_dataset = BirdsDataset(train_img_dir, train_gt, transform=train_transform, custom_len=64*1)
        faces_dataloader = DataLoader(faces_dataset, batch_size = 64, shuffle=True)
        trainer.fit(model, faces_dataloader)
        return model

    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath="runs/pl_classifier",
        filename="{epoch}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[MyTrainingModuleCheckpoint],
        default_root_dir ='./',
    )

    model = CustomNetwork(is_init=True)
    if test_gt is None:
        faces_dataloader = BirdsDataloader(train_img_dir, train_gt)
    else:
        faces_dataloader = BirdsTrainSplittingDataloader(train_img_dir, train_gt)
    trainer.fit(model, datamodule= faces_dataloader)
    trainer.save_checkpoint('birds_model.ckpt')
    return model

def classify(model_filename, test_img_dir):
    model = CustomNetwork.load_from_checkpoint(model_filename, map_location = 'cpu')
    model.eval()
    test_dataset = BirdsDataset(test_img_dir, transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 64)
    classified = {}
    for img_, img_name_ in tqdm(test_dataloader):
        gt_pred = model.predict(img_).numpy()
        for i, name_ in enumerate(img_name_):
            classified[name_] = gt_pred[i]
    return classified


# import warnings
# import gc
# gc.collect()
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     train_classifier('./tests/00_test_img_input/train/gt.csv', './tests/00_test_img_input/train/images', False, './tests/00_test_img_gt/gt.csv', './tests/00_test_img_input/test/images')
# detect('facepoints_model.ckpt', './tests/00_test_img_input/test/images')

