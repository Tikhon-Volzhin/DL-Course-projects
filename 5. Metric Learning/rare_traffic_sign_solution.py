import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import os
import csv
import cv2
import json
import tqdm
import pickle
import random
import typing
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import gc

CLASSES_CNT = 205

class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = []
        self.classes_to_samples = {i[1]: [] for i in self.class_to_idx.items()}
        self.class_to_sample_paths = {i[1]: [] for i in self.class_to_idx.items()}
        for root_folder in root_folders:
            for class_name in os.listdir(root_folder):
                dir_class_folder = os.path.join(root_folder, class_name)
                class_dir_list = os.listdir(dir_class_folder)
                class_idx = self.class_to_idx[class_name]
                self.samples += [(os.path.join(dir_class_folder, img_name), class_idx) for img_name in class_dir_list]
                self.classes_to_samples[class_idx] += class_dir_list
                self.class_to_sample_paths[class_idx] += [os.path.join(dir_class_folder, img_name) for img_name in class_dir_list]
        self.transform = A.Compose([A.Resize(64, 64), A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        root, class_idx = self.samples[index]
        img_ = np.array(Image.open(root))
        return self.transform(image = img_)['image'], root, class_idx

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        """
        with open(path_to_classes_json) as f:
            js_dict = json.load(f)
        class_to_idx = {name: val['id'] for name, val in js_dict.items()}
        buff = sorted(list(class_to_idx.items()), key=lambda s: s[1])
        classes = [x[0] for x in buff]
        return classes, class_to_idx

    @staticmethod
    def get_freq_dict(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        """
        with open(path_to_classes_json) as f:
            js_dict = json.load(f)
        idx_to_type = {val['id']:(0 if val['type'] == "freq" else 1)  for name, val in js_dict.items()}
        return idx_to_type


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    """
    def __init__(self, root, path_to_classes_json, annotations_file = None):
        super(TestData, self).__init__()
        self.root = root
        self.samples = os.listdir(root)
        self.transform = A.Compose([A.Resize(64, 64), A.Normalize(), ToTensorV2()])
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        self.idx_to_type = DatasetRTSD.get_freq_dict(path_to_classes_json)

        self.targets = None
        if annotations_file is not None:
            annotations = pd.read_csv(annotations_file)
            annotations['class'] = annotations['class'].apply(lambda x: self.class_to_idx[x]).astype('int64')
            self.targets = dict(zip(annotations['filename'], annotations['class']))

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_path = os.path.join(self.root, sample)
        img_ = np.array(Image.open(sample_path))
        if self.targets is not None:
            class_idx = self.targets[sample]
            return self.transform(image = img_)['image'], sample, class_idx, self.idx_to_type[class_idx]
        return self.transform(image = img_)['image'], sample, -1

    def __len__(self):
        return len(self.samples)


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        self.class_to_sample_paths = data_source.class_to_sample_paths
        self.roots_to_samples_idx = {path: idx for idx, (path, _) in enumerate(data_source.samples)}
        self.classes = [class_ for class_ in self.class_to_sample_paths]
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.curr_steps, self.full_steps = 0, len(data_source)//(elems_per_class * classes_per_batch) + 1

    def __len__(self):
        return self.full_steps//20
    
    def __iter__(self):
        result_batches = []
        while self.curr_steps <= self.full_steps:
            chosen_classes = [self.class_to_sample_paths[class_] for class_ in random.sample(self.classes, self.classes_per_batch)]
            chosen_paths = [random.choices(elems, k = self.elems_per_class) for elems in chosen_classes]
            filenames_ = [path for paths in chosen_paths for path in paths]
            result_batches.append([self.roots_to_samples_idx[path] for path in filenames_])
            self.curr_steps += 1
        self.curr_steps = 0
        return iter(result_batches)


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    """
    def __init__(self, data_source, examples_per_class) -> None:
        self.class_to_sample_paths = data_source.class_to_sample_paths
        self.num_steps = len(data_source)
        self.roots_to_samples_idx = {path: idx for idx, (path, _) in enumerate(data_source.samples)}
        self.classes = [class_ for class_ in self.class_to_sample_paths]
        self.examples_per_class = examples_per_class

    def __iter__(self):
        chosen_paths = [random.sample(self.class_to_sample_paths[class_], self.examples_per_class) for class_ in self.classes]
        filenames_ = [path for paths in chosen_paths for path in paths]
        return iter([self.roots_to_samples_idx[path] for path in filenames_])


class IndexSampler_(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    """
    def __init__(self, data_source, examples_per_class) -> None:
        self.class_to_sample_paths = data_source.class_to_sample_paths
        self.num_steps = len(data_source)
        self.roots_to_samples_idx = {path: idx for idx, (path, _) in enumerate(data_source.samples)}
        self.classes = [class_ for class_ in self.class_to_sample_paths]
        self.examples_per_class = examples_per_class

    def __iter__(self):
        chosen_paths = [random.sample(self.class_to_sample_paths[class_], self.examples_per_class) for class_ in self.classes]
        filenames_ = [path for paths in chosen_paths for path in paths]
        return iter([[self.roots_to_samples_idx[path] for path in filenames_]])


class Resnet50_based_model(pl.LightningModule):
    def __init__(self, features_criterion = None, internal_features = 1024, is_init = False):
        super().__init__()
        self.features_criterion = features_criterion
        if is_init:
            self.res_model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.res_model = torchvision.models.resnet50(weights = None)
        linear_size = list(self.res_model.children())[-1].in_features
        self.res_model.fc = torch.nn.Linear(linear_size, internal_features)
        self.relu_ = torch.nn.ReLU()

        self.fc_last_ = torch.nn.Linear(internal_features, CLASSES_CNT)
        for child in list(self.res_model.children()):
            for param in child.parameters():
                param.requires_grad = True

        freeze = ""
        if freeze == "most":
            for child in list(self.res_model.children())[:-4]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.fc_last_(self.relu_(self.res_model(x)))
    

def time_decay(step):
    k = 0.001

    lrate = 1 / (1 + k * step)

    return lrate


class CustomNetwork(pl.LightningModule):

    def __init__(self, features_criterion = None, internal_features = 1024, is_init = False):
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
        X, _ ,  y_true = train_batch
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
        X, _ ,  y_true, y_type = val_batch
        f_pred = self.forward(X)
        y_pred = F.log_softmax(f_pred, dim = 1)
        loss = F.nll_loss(y_pred, y_true)
        if self.features_criterion is not None:
            loss += self.features_criterion(y_true, f_pred)
        y_true_mask = y_pred.argmax(axis=1) == y_true
        acc = torch.sum(y_true_mask) / y_true.shape[0]
        freq_acc = torch.sum(y_true_mask * (y_type == 0)) / (y_type == 0).sum()
        rare_acc = torch.sum(y_true_mask * (y_type == 1)) / (y_type == 1).sum()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_freq_acc", freq_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rare_acc", rare_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)

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


class FeaturesLoss(torch.nn.Module):
    """
    Contrastive loss
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        self.margin = margin

    def forward(self, y_true, f_pred):
        batch_len = y_true.size()[0]
        mask = (y_true.reshape(1, -1) - y_true.reshape(-1, 1)) == 0
        f_mtx_buff = f_pred.reshape(1, batch_len, -1) - f_pred.reshape(batch_len, 1, -1)
        f_sq_norm = torch.sum(f_mtx_buff * f_mtx_buff, dim=-1)
        f_norm = torch.sqrt(f_sq_norm + 1e-15)
        eq_sum, non_eq_sum = torch.sum(mask), torch.sum(~mask)
        f_margin_max =  F.relu(self.margin - f_norm)
        loss = 0.5 * torch.where(mask, f_sq_norm/eq_sum, f_margin_max*f_margin_max/non_eq_sum).sum()
        return loss
    

def apply_classifier(model, test_folder, path_to_classes_json):
    test_data = TestData(test_folder, path_to_classes_json)
    results = []
    for i in range(len(test_data)):
        img_, filename_, class_ = test_data[i]
        results.append({'filename' : filename_, 'class': test_data.classes[int(model.predict(img_[None,...])[0])]})
    return results

def calc_metrics_(y_pred, y_true):
    all_cnt = len(y_pred)
    ok_cnt = (y_pred == y_true).sum()
    return ok_cnt/max(1, all_cnt)
    
def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    with open(path_to_classes_json) as f:
        js_dict = json.load(f)
    idx_to_type = {val['id']: val['type'] for _, val in js_dict.items()}
    test_data = TestData(test_folder, path_to_classes_json, annotations_file)
    rare = []
    freq = []
    for i in range(len(test_data)):
        img_, filename_, class_ = test_data[i]
        y_pred = model.predict(img_[None,...])[0,...]
        if idx_to_type[class_] == 'freq':
            freq.append([y_pred, class_])
        else:
            rare.append([y_pred, class_])
    freq_np, rare_np = np.array(freq), np.array(rare)
    total_np = np.vstack([freq_np, rare_np])
    total_acc = calc_metrics_(total_np[:, 0], total_np[:, 1])
    rare_recall = calc_metrics_(rare_np[:, 0], rare_np[:, 1])
    freq_recall = calc_metrics_(freq_np[:, 0], freq_np[:, 1])
    
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    """
    def __init__(self, background_path):
        self.bgr_list = os.listdir(background_path)
        self.bgr_path = background_path
        self.icon_transforms = A.Compose([A.Resize(128, 128),
                                          A.RandomScale([-0.1, 0.], p = 1.),
                                          A.SafeRotate([-15, 15], p = 0.8, border_mode=cv2.BORDER_CONSTANT, value=(0,)), 
                                          A.ColorJitter(brightness=(0.45, 0.7), saturation=(0.45, 0.7), hue = 0., p = 1.0), 
                                          A.MotionBlur(blur_limit = (11, 19), allow_shifted = False,  p=1.),
                                          A.PadIfNeeded(128, 128, border_mode=cv2.BORDER_CONSTANT, value=(0,)),
                                          A.RandomResizedCrop(128, 128, scale=(0.65, 1.0), p=0.8),
                                          A.Resize(64, 64),
                                          ])
        self.bgr_transforms = A.Compose([A.RandomSizedCrop ([63, 65], 64, 64)])
        self.final_transforms = A.Compose([A.GaussianBlur(blur_limit = (3, 5), sigma_limit=(1., 1.1), p=1.), 
                                           A.ColorJitter(brightness=(0.8, 1.2), hue = 0., p = 0.5)])

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        """
        bg = Image.open(os.path.join(self.bgr_path, random.choices(self.bgr_list)[0]))
        bg_np = np.array(bg)
        h_img, w_img, _ = bg_np.shape
        bg_cropped = bg_np[h_img//3: 2 * h_img//3, 2 * w_img//3:, ...]
        bg_tr = self.bgr_transforms(image = bg_cropped)['image']
        icon_img_ = np.array(Image.open(icon))
        icon_tr = self.icon_transforms(image = icon_img_[..., :-1], mask = icon_img_[..., -1][..., None])
        icon_img, icon_mask = icon_tr['image'], icon_tr['mask']
        bg_mask = ~(icon_mask > 254)
        res_img_ = np.clip(bg_tr.astype(np.uint16) * bg_mask + icon_img.astype(np.uint16) * (~bg_mask), 0, 255).astype(np.uint8)
        return self.final_transforms(image = res_img_)['image']


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    """
    icon, output_folder, background_path, samples_per_class = args
    sign_gen = SignGenerator(background_path)
    class_name = (icon.rpartition('.')[0]).rpartition('\\')[2]
    os.mkdir(os.path.join(output_folder, class_name))
    for i in range(samples_per_class):
        res_img_ = sign_gen.get_sample(icon)
        res_img_PIL_ = Image.fromarray(res_img_)
        res_img_PIL_.save(os.path.join(output_folder, class_name, str(i) + '.png'), 'PNG')


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    """
    
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))

# if __name__ == '__main__':
#     import sys
#     np.set_printoptions(threshold=sys.maxsize)
#     output_folder = './synt_train/synt_train'
#     icons_path = './icons/icons'
#     background_path = './background_images/background_images'
#     generate_all_data(output_folder, icons_path, background_path)


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    """
    knn_model = KNeighborsClassifier(n_neighbors=20)
    model = CustomNetwork(features_criterion = FeaturesLoss(margin = 2.), is_init = True)
    model.load_state_dict(torch.load(nn_weights_path, map_location=torch.device('cpu')))
    model.eval()
    train_dataset = DatasetRTSD(['./synt_train/synt_train'], 'classes.json')
    train_dataloader = DataLoader(train_dataset, batch_sampler= IndexSampler_(train_dataset, 1))
    init = True
    f_pred_total, y_total = None, None
    for _ in range(examples_per_class):
        print(_)
        X, _, y = next(iter(train_dataloader))
        f_pred = model(X).detach().numpy()
        if init:
            f_pred_total = f_pred
            y_total = y
            init = not init
        else:
            f_pred_total = np.concatenate([f_pred_total, f_pred], axis=0)
            y_total = np.concatenate([y_total, y])
    knn_model = knn_model.fit(f_pred_total/np.linalg.norm(f_pred_total, axis=1)[:, None], y_total)
    with open('knn_model.bin', 'wb') as file:
        pickle.dump(knn_model, file, protocol=pickle.HIGHEST_PROTOCOL)

def test_head(nn_weights_path):
    knn_model = None
    with open('knn_model.bin', 'rb') as file:
        knn_model = pickle.load(file)
    model = CustomNetwork(features_criterion = FeaturesLoss(margin = 2.), is_init = True)
    model.load_state_dict(torch.load(nn_weights_path, map_location=torch.device('cpu')))
    model.eval()
    test_dataset = TestData('./smalltest/smalltest', './test_classes.json', './test_annotations.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    for batch in test_dataloader:
        X, _, y, y_type = batch
        f_pred = model(X)
        f_pred = f_pred.detach().numpy()
        y_pred = knn_model.predict(f_pred/np.linalg.norm(f_pred, axis=1)[:, None])
        freq_mask = y_type == 0
        print('val acc', accuracy_score(y, y_pred))
        print('val freq acc', accuracy_score(y[freq_mask], y_pred[freq_mask]))
        print('val rare acc', accuracy_score(y[~freq_mask], y_pred[~freq_mask]))

if __name__ == '__main__':
    nn_weights_path = 'improved_features_model.pth'
    if os.path.exists('knn_model.bin'):
        test_head('improved_features_model.pth')
    else:
        train_head('improved_features_model.pth', 40)
        test_head('improved_features_model.pth')
    
class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors        

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        """
        self.model = CustomNetwork(features_criterion = FeaturesLoss(margin = 2.))
        self.model.load_state_dict(torch.load(nn_weights_path, map_location='cpu'))
        self.model.eval()

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        """
        with open(knn_path, 'rb') as file:
            self.knn_head = pickle.load(file)

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        """
        features = self.model(imgs).detach().numpy()
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = self.knn_head.predict(features)
        return knn_pred
        


