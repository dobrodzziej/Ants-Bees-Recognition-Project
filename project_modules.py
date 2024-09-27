from typing import Any

from pathlib import Path
import shutil

from datetime import datetime

import math
import numpy as np

from PIL import Image, ImageOps
import albumentations as A
import cv2

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

from sklearn.metrics import accuracy_score

import mlflow
import mlflow.pytorch as ml_pt


def get_new_image_size(image: Image, desired_wh: int=500) -> tuple[int, int]:
    w, h = image.size

    if w > h:
        new_w = desired_wh
        new_h = int((desired_wh / w) * h)
    else:
        new_h = desired_wh
        new_w = int((desired_wh / h) * w)
    print(new_w, new_h)
    return new_w, new_h


def get_borders_size(new_image_size: tuple[int, int], desired_size: tuple[int, int]=(500, 500)):
    missing_sizes = [dim2-dim1 for dim1, dim2 in zip(new_image_size, desired_size)]
    left = int(np.floor(missing_sizes[0] / 2))
    right = int(np.ceil(missing_sizes[0] / 2))
    top = int(np.floor(missing_sizes[1] / 2))
    bottom = int(np.ceil(missing_sizes[1] / 2))
    return left, top, right, bottom


def modify_image(image: Image, desired_wh: int=214):
    new_image_size = get_new_image_size(image=image, desired_wh=desired_wh)
    borders = get_borders_size(new_image_size=new_image_size, desired_size=(desired_wh, desired_wh))
    image_mod = image.resize(size=new_image_size)
    image_mod = ImageOps.expand(image=image_mod, border=borders)
    return image_mod


def prep_images(original_files_dirs, modified_files_dirs, image_modifier, alb_trans=None, add_pictures_with_borders=False):
    try:
        zipped_dirs = zip(original_files_dirs, modified_files_dirs)
        for dir_original, dir_modified in zipped_dirs:
            # print(original, modified)
            if dir_modified.exists():
                shutil.rmtree(dir_modified)
            dir_modified.mkdir(parents=True)
            for image_original in dir_original.iterdir():
                im = Image.open(image_original)
                im_mod = image_modifier.modify_image(image=im, add_borders=False)
                im_mod2 = np.array(im_mod)
                fname = dir_modified/(image_original.stem + "0" + image_original.suffix)
                plt.imsave(fname=fname, arr=im_mod2)
                if im_mod2.shape[0] != im_mod2.shape[1] and add_pictures_with_borders:
                    im_mod = image_modifier.modify_image(image=im, add_borders=True)
                    im_mod2 = np.array(im_mod)
                    fname = dir_modified/(image_original.stem + "1" + image_original.suffix)
                    plt.imsave(fname=fname, arr=im_mod2)
                if alb_trans is not None:
                    for i in range(3):
                        add_borders = (np.random.random(1) > 0.75)[0]
                        im_mod = image_modifier.modify_image(image=im, add_borders=add_borders)
                        im_mod2 = np.array(im_mod)
                        im_mod2 = alb_trans(image=im_mod2)
                        im_mod2 = im_mod2["image"]
                        fname = dir_modified/(image_original.stem + str(i+2) + image_original.suffix)
                        plt.imsave(fname=fname, arr=im_mod2)
    except TypeError:
        print(f"Error with file: {image_original}")


class ImageModifier():
    def __init__(self, desired_wh: int) -> None:
        self.desired_wh = desired_wh
        self.new_image_size = None
        self.borders = None

    def __get_new_image_size(self, image: Image, add_borders: bool) -> tuple[int, int]:
        w, h = image.size

        if add_borders:
            if w > h:
                new_w = self.desired_wh
                new_h = int((self.desired_wh / w) * h)
            else:
                new_h = self.desired_wh
                new_w = int((self.desired_wh / h) * w)
        else:
            if w < h:
                new_w = self.desired_wh
                new_h = int((self.desired_wh / w) * h)
            else:
                new_h = self.desired_wh
                new_w = int((self.desired_wh / h) * w)

        self.new_image_size = (new_w, new_h)
    
    def __get_borders_size(self):
        desired_size = (self.desired_wh, self.desired_wh)
        missing_sizes = [dim2-dim1 for dim1, dim2 in zip(self.new_image_size, desired_size)]
        left = int(np.floor(missing_sizes[0] / 2))
        right = int(np.ceil(missing_sizes[0] / 2))
        top = int(np.floor(missing_sizes[1] / 2))
        bottom = int(np.ceil(missing_sizes[1] / 2))
        self.borders = (left, top, right, bottom)
    
    def modify_image(self, image: Image, add_borders: False):
        self.__get_new_image_size(image=image, add_borders=add_borders)
        image_mod = image.resize(size=self.new_image_size)
        self.__get_borders_size()
        image_mod = ImageOps.expand(image=image_mod, border=self.borders)
        return image_mod
    

class MyResidualUnitChangeDepth(nn.Module):
    def __init__(self, in_channels, out_channels, second_stride, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
        self.skip_conv_batch_norm = nn.BatchNorm2d(num_features=out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(num_features=out_channels), 
            self.relu, 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=second_stride, padding=1, bias=False), 
            nn.BatchNorm2d(num_features=out_channels)
            )
        
    def forward(self, x):
        x_skip = x
        x_skip = self.skip_conv(x_skip)
        x_skip = self.skip_conv_batch_norm(x_skip)
        out = self.conv(x)
        out += x_skip
        out = self.relu(out)
        return out


class MyResidualUnit(MyResidualUnitChangeDepth):
    def __init__(self, in_out_channels, second_stride, *args, **kwargs) -> None:
        super().__init__(in_channels=in_out_channels, out_channels=in_out_channels, second_stride=second_stride, *args, **kwargs)
    
    def forward(self, x):
        x_skip = x
        out = self.conv(x)
        out += x_skip
        out = self.relu(out)
        return out


class MyResNet34(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_channels = 3
        out_channels = 64

        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.first_relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=512, out_features=1)
        self.softmax = nn.Softmax()
        self.residual_units = nn.ModuleList()
        channel_size = 64
        for channel in [channel_size] * 3:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 128
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 3:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 256
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 5:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 512
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 2:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))


    def forward(self, x):
        out = self.first_conv(x)
        out = self.first_batch_norm(out)
        out = self.first_relu(out)
        out = self.max_pool(out)
        for idx, unit in enumerate(self.residual_units):
            out = unit(out)
        out = nn.AvgPool2d(kernel_size=out.shape[2])(out)
        out = out.view(out.shape[0], -1)
        # out = torch.mean(out, dim=(2, 3))
        out = self.fc(out)
        return out # Sigmoid will be applied in the BCEWithLogitsLoss loss function
    

class MyResidualUnitChangeSize50(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, first_stride, padding=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # in_channels and out_channels are saved as parameters to detect if additional convolution should be later performed
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=first_stride, padding=padding,  bias=False)
        self.skip_conv_batch_norm = nn.BatchNorm2d(num_features=out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=first_stride, padding=padding, bias=False), 
            nn.BatchNorm2d(num_features=inner_channels), 
            self.relu, 
            nn.Conv2d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=3, stride=1, padding="same", bias=False), 
            nn.BatchNorm2d(num_features=inner_channels), 
            self.relu, 
            nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(num_features=out_channels)
            )
        
    def forward(self, x):
        x_skip = x
        x_skip = self.skip_conv(x_skip)
        x_skip = self.skip_conv_batch_norm(x_skip)
        out = self.conv(x)
        out += x_skip
        out = self.relu(out)
        return out


class MyResidualUnit50(MyResidualUnitChangeSize50):
    def __init__(self, in_out_channels, inner_channels, stride, padding=0, *args, **kwargs) -> None:
        super().__init__(in_channels=in_out_channels, inner_channels=inner_channels, out_channels=in_out_channels, 
                         first_stride=stride, padding=padding, *args, **kwargs)
    
    def forward(self, x):
        x_skip = x
        out = self.conv(x)
        out += x_skip
        out = self.relu(out)
        return out
    

class MyResNet50(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # input images are RGB
        in_channels = 3
        first_channel_size = 64
        # Kernel size used in all layers except the first one
        main_kernel_size = 3

        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=first_channel_size, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_batch_norm = nn.BatchNorm2d(num_features=first_channel_size)
        self.first_relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=main_kernel_size, stride=2, padding=1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=2048, out_features=1)
        self.softmax = nn.Softmax()
        self.residual_units = nn.ModuleList()

        channel_size = 64
        # first stride is exceptionally 1 and padding is 1 as we do not change image size here
        self.residual_units.append(MyResidualUnitChangeSize50(in_channels=channel_size, inner_channels=channel_size, out_channels=(channel_size*4), 
                                                              first_stride=1, padding=0))
        for channel in [channel_size] * 2:
            self.residual_units.append(MyResidualUnit50(in_out_channels=(channel*4), inner_channels=channel, stride=1, padding="same"))

        channel_size = 128
        self.residual_units.append(MyResidualUnitChangeSize50(in_channels=(channel_size*2), inner_channels=channel_size, out_channels=(channel_size*4), 
                                                        first_stride=2, padding=0))
        for channel in [channel_size] * 3:
            self.residual_units.append(MyResidualUnit50(in_out_channels=(channel*4), inner_channels=channel, stride=1, padding="same"))

        channel_size = 256
        self.residual_units.append(MyResidualUnitChangeSize50(in_channels=(channel_size*2), inner_channels=channel_size, out_channels=(channel_size*4), 
                                                        first_stride=2, padding=0))
        for channel in [channel_size] * 5:
            self.residual_units.append(MyResidualUnit50(in_out_channels=(channel*4), inner_channels=channel, stride=1, padding="same"))

        channel_size = 512
        self.residual_units.append(MyResidualUnitChangeSize50(in_channels=(channel_size*2), inner_channels=channel_size, out_channels=(channel_size*4), 
                                                        first_stride=2, padding=0))
        for channel in [channel_size] * 2:
            self.residual_units.append(MyResidualUnit50(in_out_channels=(channel*4), inner_channels=channel, stride=1, padding="same"))


    def forward(self, x):
        out = self.first_conv(x)
        out = self.first_batch_norm(out)
        out = self.first_relu(out)
        out = self.max_pool(out)
        for idx, unit in enumerate(self.residual_units):
            out = unit(out)
        out = nn.AvgPool2d(kernel_size=out.shape[2])(out)
        out = out.view(out.shape[0], -1)
        # out = torch.mean(out, dim=(2, 3))
        out = self.fc(out)
        return out # Sigmoid will be applied in the BCEWithLogitsLoss loss function
    

class ToTensorWithGrad:
    def __call__(self, pic):
        tensor = transforms.functional.to_tensor(pic)
        tensor.requires_grad = True
        return tensor
    

class TargetTransform():
    def __call__(self, y):
        # print(y)
        tensor = torch.tensor(y, dtype=torch.float).view(1)
        return tensor
    

class TrainingSupervisor():
    def __init__(self, model: nn.Module, early_stop_patience, lr, lr_patience, lr_reduce_factor, checkpoints_default_filepath: Path) -> None:
        self.es_patience = early_stop_patience
        self.es_patience_counter = 0
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_patience_counter = 0
        self.best_val_loss = np.infty
        self.model = model
        self.checkpoints_default_filepath = checkpoints_default_filepath
        self.mlflow_model_checkpoints_saved = 0
        if not checkpoints_default_filepath.parent.exists():
            checkpoints_default_filepath.parent.mkdir(parents=True, exist_ok=True)
        # self.dataloader_train = dataloader_train
        # self.dataloader_val = dataloader_val

    def save_model_state_dict(self, file_path: Path=None):
        if file_path is None:
            origin_fpath = self.checkpoints_default_filepath
            current_datetime_str = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
            file_path = origin_fpath.parent/(current_datetime_str+"--"+origin_fpath.name)
        print("Saving model...")
        torch.save(self.model.state_dict(), file_path)

    def mlflow_save_model_state_dict(self, artifact_path: str):
        print("Saving model...")
        ml_pt.log_state_dict(state_dict=self.model.state_dict(), artifact_path=artifact_path)
        self.mlflow_model_checkpoints_saved += 1


    def update_patience_counters(self, current_val_loss):
        # return whether to save the model or not
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.es_patience_counter = 0
            self.lr_patience_counter = 0
            return True
        self.es_patience_counter += 1
        self.lr_patience_counter += 1
        return False

    def early_stop(self): 
        # returns whether to early stop or not
        if self.es_patience_counter > self.es_patience:
            print("Early Stopping")
            return True
        return False
            
    def update_lr(self):
        if self.lr_patience_counter > self.lr_patience:
            self.lr_patience_counter = 0
            self.lr /= self.lr_reduce_factor
            print("Decreasing lr")
            return self.lr
        return self.lr

    def check_accuracy(self, loss_fn: nn.Module, dataloader: DataLoader, device: str, return_y_scores=False):
        num_all = len(dataloader.dataset)
        num_correct = 0
        if return_y_scores:
            self.model.eval().to(device)
            y_scores = torch.zeros(num_all, 1).to(device)
            with torch.no_grad():
                loss_sum = torch.zeros(len(dataloader)).to(device)
                for idx, (X, y) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)
                    output = self.model(X)
                    val_loss = loss_fn(output, y)
                    loss_sum[idx] = val_loss

                    start_idx = idx * dataloader.batch_size
                    end_idx = start_idx + output.size(0)
                    y_scores[start_idx:end_idx] = output

                    output = F.sigmoid(output)
                    y_pred = (output > 0.5).float()

                    num_correct += sum(y_pred == y)
                    # print(f"y_pred: {y_pred.ravel()}")
                    # print(y.ravel())
                    # print(f"idx: {idx}, num_correct: {num_correct}")
                loss_relative_sum = float(torch.sum(loss_sum)) / num_all
                accuracy = float((num_correct / num_all * 100))
            self.model.train()
            return accuracy, loss_relative_sum, y_scores
        
        else:
            self.model.eval().to(device)
            with torch.no_grad():
                loss_sum = torch.zeros(len(dataloader)).to(device)
                for idx, (X, y) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)
                    output = self.model(X)
                    val_loss = loss_fn(output, y)
                    loss_sum[idx] = val_loss
                    output = F.sigmoid(output)
                    y_pred = (output > 0.5).float()
                    num_correct += sum(y_pred == y)
                    # print(f"y_pred: {y_pred.ravel()}")
                    # print(y.ravel())
                    # print(f"idx: {idx}, num_correct: {num_correct}")
                loss_relative_sum = float(torch.sum(loss_sum)) / num_all
                accuracy = float((num_correct / num_all * 100))
            self.model.train()
            return accuracy, loss_relative_sum


def mlflow_log_figure(X, Y, artifact_file: str, Y2=None, title="", xlabel="", ylabel="", ylim=None, legend=None):
    figure = plt.figure()
    plt.plot(X, Y)
    if Y2 is not None:
        plt.plot(X, Y2) 
    plt.grid(visible=True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    if legend is not None:
        plt.legend(legend)
    mlflow.log_figure(figure, artifact_file=artifact_file)
    plt.close()


def create_learning_curves_dict(epochs, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates):
    learning_curves_dict = {"Epoch": epochs, "Training loss": train_losses, "Validation loss": val_losses, 
                            "Training accuracy": train_accuracies, "Validation accuracy": val_accuracies, 
                            "Learning rate": learning_rates}
    return learning_curves_dict