import json
import wandb
import itertools

from typing import List, Tuple
from operator import itemgetter
from typing import Dict
from math import log
from .prune import PruningExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


class LotteryTicketExperiment(PruningExperiment):
    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 run_on_device=True,
                 warmup_iterations=0,
                 rewinding_it=0,
                 pruning_rate=0.15,
                 logging=True,
                 lr_factor=0.2):

        super().__init__(
            dataset=dataset,
            model=model,
            strategy=strategy,
            compression=1.,     # TODO check
            seed=seed,
            path=path,
            dl_kwargs=dl_kwargs,
            train_kwargs=train_kwargs,
            debug=debug,
            pretrained=False,   # TODO checl
            resume=None,
            resume_optim=False,
            save_freq=save_freq,
            run_on_device=run_on_device,
            warmup_iterations=warmup_iterations,
            k_iteration_save=rewinding_it,
            logging=logging
        )

        self.compression = compression
        self.rewinding_it = rewinding_it
        self.pruning_rate = pruning_rate
        self.strategy = strategy
        self.train_metrics_list = list()
        self.pruning_levels_list = list()
        self.lr_factor = lr_factor
        self.lr_decreased_count = 0

        self.add_params(pruning_rate=pruning_rate)
        self.add_params(rewinding_it=rewinding_it)
        self.add_params(compression=compression)
        self.add_params(lr_factor=lr_factor)

    def init_logger(self, **config):
        wandb.init(project='knn-pruning', entity='sonypony', config=config)

    def log(self, **kwargs):
        wandb.log(kwargs)

    def log_end_summary(self):
        train_accs = list(map(itemgetter(1), self.train_metrics_list))
        val_accs = list(map(itemgetter(3), self.train_metrics_list))
        pruning_keys = itertools.product(self.pruning_levels_list, ["train", "val"])
        lr_list = list(map(itemgetter(4), self.train_metrics_list))

        wandb.log({"pruning_training_course": wandb.plot.line_series(
            xs=[e * len(self.train_dl)
                for e in range(self.params["train_kwargs"]["epochs"])],
            ys=list(itertools.chain(*zip(train_accs, val_accs))), # interlace runs - [tr, val, tr, val, ...]
            xname="Iteration",
            title="Accuracy During Training",
            keys=["p% = {:.4f} - {}".format(*x) for x in pruning_keys]
        )})

        wandb.log({"pruning_lr_course": wandb.plot.line_series(
            xs=[e * len(self.train_dl)
                for e in range(self.params["train_kwargs"]["epochs"])],
            ys=lr_list,
            xname="Iteration",
            title="LR During Training",
            keys=["p% = {:.4f}".format(x) for x in self.pruning_levels_list]
        )})

    def early_stop(self, epoch, train_metrics, val_metrics, lr_decreased) -> bool:
        """self.lr_decreased_count += lr_decreased

                if self.lr_decreased_count >= 3:
                    printc("Early stop", color="RED")
                    return True
                return False"""

        return False

    # TODO link to repo
    def decrease_lr(self, epoch: int, train_metrics: List[Tuple[float]], val_metrics: List[Tuple[float]]) -> bool:
        val_accs = list(map(itemgetter(1), val_metrics))

        if len(val_accs) <= 10 or self.lr_decreased_count >= 2:
            return False

        # validation did not improve (at all or at least by 0.2% absolute)
        # after 3 epochs
        min_acc_last_three_epochs = min(val_accs[-3:])
        last_val_acc = val_accs[-1]

        if last_val_acc < min_acc_last_three_epochs or \
                abs(last_val_acc - min_acc_last_three_epochs) < 0.002:
            self.current_lr = self.current_lr * self.lr_factor
            printc("Decrease lr to:", self.current_lr, color="GREEN")
            self.lr_decreased_count += 1
            return True
        return False

    def pruning_iteration_run(self, target_compression_level: float):
        if target_compression_level != 1.:
            # prune
            pruning_masks = self.pruning_masks(self.strategy, compression=target_compression_level)
            self.model.load_state_dict(self.k_iteration_params)
            self.apply_pruning_masks(masks=pruning_masks)

        self.freeze()
        self.to_device()
        self.lr_decreased_count = 0

        printc("{}\n\nCurrent p% level: {}".format("-" * 50, 1 / target_compression_level), color="YELLOW")

        init_metrics = self.pruning_metrics()
        printc("Test Init-Acc-1: {}, Test Init-Acc-5: {}".format(
            init_metrics.get("test_acc1"), init_metrics.get("test_acc5")), color="YELLOW")

        # train
        self.train_metrics_list.append(self.run_epochs())
        self.pruning_levels_list.append(1 / target_compression_level)

        metrics = self.pruning_metrics()
        printc(json.dumps(metrics, indent=4), color='GRASS')

        # log
        self.log(
            test_init_acc_1=init_metrics.get("test_acc1"),
            test_init_acc_5=init_metrics.get("test_acc5"),
            test_acc_1=metrics.get("test_acc1"),
            test_acc_5=metrics.get("test_acc5"),
            p_perc=1 / target_compression_level,
            flops=metrics.get("flops_nz"),
            theoretical_speedup=metrics.get("theoretical_speedup")
        )

    def run(self):
        self.init_logger(**self.summary_params())

        printc(f"Running {repr(self)}", color='YELLOW')

        target_pruning_ratio_inv = 1 / self.compression
        pruning_iterations = round(log(target_pruning_ratio_inv, 1 - self.pruning_rate) - 1)

        self.pruning_iteration_run(target_compression_level=1.)

        for i in range(pruning_iterations):
            target_compression_level = 1 / (1 - self.pruning_rate) ** (i + 1)
            self.pruning_iteration_run(target_compression_level=target_compression_level)
        self.log_end_summary()

    def summary_params(self) -> Dict:
        return {
            "dataset": self.params["dataset"],
            "model": self.params["model"],
            "seed": self.params["seed"],
            "batch_size": self.params["dl_kwargs"]["batch_size"],
            "split_ratio": self.params["dl_kwargs"]["split_ratio"],
            "optim": self.params["train_kwargs"]["optim"],
            "train_iterations": self.params["train_kwargs"]["epochs"] * len(self.train_dl),
            "lr": self.params["train_kwargs"]["lr"],
            "pretrained": self.params["pretrained"],
            "warmup_iterations": self.params["warmup_iterations"],
            "rewind_iteration": self.params["rewinding_it"],
            "pruning_rate": self.params["pruning_rate"],
            "strategy": self.params["strategy"],
            "lr_factor": self.lr_factor
        }