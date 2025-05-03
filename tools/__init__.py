# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net
from .runner_module import run_net as module_run_net
from .runner_finetune_seg import run_net as finetune_seg_run_net
from .runner_module_seg import run_net as module_seg_run_net
from .runner_module_seman_seg import run_net as module_seman_seg_run_net