import pathlib
import time

import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import json
import pytorch_warmup as warmup
from copy import deepcopy
from math import floor
from enum import Enum
from operator import itemgetter

from .base import Experiment
from .. import datasets
from .. import models
from ..metrics import correct
from ..models.head import mark_classifier
from ..util import printc, OnlineStats


class Run(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class TrainingExperiment(Experiment):

    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'lr': 1e-3,
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 run_on_device=True,
                 warmup_iterations=0,
                 k_iteration_save=-1,
                 logging=True):

        # Default children kwargs
        super(TrainingExperiment, self).__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs

        self.logging = logging
        self.warmup_iterations = warmup_iterations
        self.k_iteration_save = k_iteration_save
        self.k_iteration_params = None
        self.add_params(**params)
        # Save params

        self.build_dataloader(dataset, **dl_kwargs)

        self.build_model(model, pretrained, resume)

        self.build_train(resume_optim=resume_optim, **train_kwargs)

        self.path = path
        self.save_freq = save_freq
        self.run_on_device = run_on_device

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        if self.logging:
            self.build_logging(self.train_metrics, self.path)
        self.run_epochs()

    def build_dataloader(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)

        self.train_dataset = constructor(train=True)
        self.val_dataset = list()

        if split_ratio := dl_kwargs.get("split_ratio", None):
            dl_kwargs.pop("split_ratio", None)
            data_size = len(self.train_dataset)
            train_data_size = floor(split_ratio * data_size)

            self.train_dataset, self.val_dataset = \
                torch.utils.data.random_split(
                    self.train_dataset,
                    lengths=(train_data_size, data_size - train_data_size),
                    generator=torch.Generator().manual_seed(42)
                )

        self.test_dataset = constructor(train=False)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)
        self.test_dl = DataLoader(self.test_dataset, shuffle=False, **dl_kwargs)

    def build_model(self, model, pretrained=True, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

    def reset_lr(self):
        self.current_lr = self.lr

    def build_train(self, optim, epochs, resume_optim=False, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3},
            'Adam': {'momentum': 0.9, 'betas': (.9, .99), 'lr': 1e-4}
        }

        self.epochs = epochs
        self.lr = optim_kwargs.get("lr")
        self.current_lr = self.lr

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim
        if self.warmup_iterations:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optim,
                lr_lambda=lambda _: self.current_lr
            )

            self.warmup_scheduler = warmup.LinearWarmup(
                optimizer=optim,
                warmup_period=self.warmup_iterations
            )

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda'
           if torch.cuda.is_available() and self.run_on_device else 'cpu')
        if not torch.cuda.is_available() or not self.run_on_device:
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self):
        if not self.logging:
            return
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{epoch}.pt')

    # returns whether lr was decreased
    def decrease_lr(self, epoch: int, train_metrics, val_metrics) -> bool:
        return False

    # returns whether early stop criterion was met
    def early_stop(self, epoch, train_metrics, val_metrics, lr_decreased) -> bool:
        return False

    def run_epochs(self):
        train_metrics, val_metrics, lr_list = [], [], []
        self.reset_lr()

        since = time.time()
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                train_metrics.append(self.train(epoch))
                val_metrics.append(self.validate(epoch))

                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                lr_decreased = self.decrease_lr(epoch, train_metrics, val_metrics)
                lr_list.append(self.current_lr)

                if self.logging:
                    self.log(timestamp=time.time()-since)
                    self.log_epoch(epoch)

                if self.early_stop(epoch, train_metrics, val_metrics, lr_decreased):
                    break

            # train loss, train acc1, val loss, val acc1, lr_list
            return list(map(itemgetter(0), train_metrics)), \
                   list(map(itemgetter(1), train_metrics)), \
                   list(map(itemgetter(0), val_metrics)), \
                   list(map(itemgetter(1), val_metrics)), \
                   lr_list

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def run_epoch(self, train: Run, epoch:int=0):
        if train == Run.TRAIN:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        elif train == Run.VAL:
            prefix = 'val'
            dl = self.val_dl
            self.model.eval()
        else:
            prefix = "test"
            dl = self.test_dl
            self.model.eval()
        train = train == Run.TRAIN

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch+1}/{self.epochs}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train:
                    if self.k_iteration_save == epoch * len(dl) + i - 1 \
                            and self.k_iteration_save != -1:
                        self.k_iteration_params = deepcopy(self.model.state_dict())

                    loss.backward()

                    self.optim.step()

                    if self.warmup_iterations:
                        self.lr_scheduler.step(self.lr_scheduler.last_epoch + 1)
                        self.warmup_scheduler.dampen()
                    self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        if self.logging:
            self.log(**{
                f'{prefix}_loss': total_loss.mean,
                f'{prefix}_acc1': acc1.mean,
                f'{prefix}_acc5': acc5.mean,
            })

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch=0):
        return self.run_epoch(Run.TRAIN, epoch)

    def validate(self, epoch=0):
        return self.run_epoch(Run.VAL, epoch)

    def eval(self, epoch=-1):
        return self.run_epoch(Run.TEST, epoch)

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5',
                ]

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__
        
        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)
