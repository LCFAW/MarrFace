import os
import glob
from datetime import datetime
import torch
from networks.dataloadersTest_ddp import get_data_loaders

class Trainer():
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cpu')
        self.batch_size = cfgs.get('batch_size', 20)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 1)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 100)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.load_checkpoint_name = cfgs.get('load_checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs

        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.load_checkpoint_name is not None:
            checkpoint_path = os.path.join(self.load_checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0
            checkpoint_path = checkpoints[-1]
        self.checkpoint_name = os.path.basename(checkpoint_path)
            
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        epoch = cp['epoch']
        return epoch

    def test(self):
        """Perform testing."""
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")
        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs_test', datetime.now().strftime("%Y%m%d-%H%M%S")))

        with torch.no_grad():
            self.model.set_eval()
            for iter, input in enumerate(self.test_loader):
                self.model.forward(input)
                 
            self.model.forward_and_draw()
