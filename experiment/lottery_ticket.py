import json

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
                 pruning_rate=0.15):

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
            k_iteration_save=rewinding_it
        )

        # TODO check
        self.compression = compression
        self.rewinding_it = rewinding_it
        self.pruning_rate = pruning_rate
        self.strategy = strategy

        self.add_params(pruning_rate=pruning_rate)
        self.add_params(rewinding_it=rewinding_it)
        self.add_params(compression=compression)

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        target_pruning_ratio_inv = 1 / self.compression
        pruning_iterations = round(log(target_pruning_ratio_inv, 1 - self.pruning_rate) - 1)

        # train model without pruning
        self.run_epochs()

        for i in range(pruning_iterations):
            target_compression_level = 1 / (1 - self.pruning_rate) ** (i + 1)

            # prune
            # TODO fix: restoring weights ignors mask??? Yes
            pruning_masks = self.pruning_masks(self.strategy, compression=target_compression_level)
            self.model.load_state_dict(self.k_iteration_params)
            self.apply_pruning_masks(masks=pruning_masks)

            self.freeze()
            self.to_device()
            self.build_logging(self.train_metrics, self.path)
            self.save_metrics()

            self.run_epochs()