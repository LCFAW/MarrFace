import os
import argparse
from utils.utils import setup_runtime
from trainer.tester_wintcapsule import Trainer
from networks.WinCapsule_Test import WinCapsule


def main(args):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  cfgs = setup_runtime(args)
  trainer = Trainer(cfgs, WinCapsule)
  trainer.test()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Training configurations.')
  parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
  
  #* model parameters
  parser.add_argument('--run_test', default=True, type=bool, help='')
  parser.add_argument('--batch_size', default=1, type=int, help='')
  parser.add_argument('--seed', default=20, type=int, help='Specify a random seed')
  parser.add_argument('--model_name', default='wint', type=str, help='')

  #* test settings
  parser.add_argument('--resume', default=True, type=bool, help='')
  parser.add_argument('--use_logger', default=True, type=bool, help='')
  parser.add_argument('--log_freq', default=1, type=int, help='')
  parser.add_argument('--save_checkpoint_freq', default=1, type=int, help='')
  parser.add_argument('--keep_num_checkpoint', default=-1, type=int, help='-1 to svae all')
  # parser.add_argument('--checkpoint_dir', default='/path/to/save/results', type=str, help='dir to save results')
  # parser.add_argument('--load_checkpoint_name', default='/path/to/pretrained/checkpoint020.pth',type=str)
  parser.add_argument('--checkpoint_dir', default='results', type=str, help='dir to save results')
  parser.add_argument('--load_checkpoint_name', default='pretrained/checkpoint020.pth',type=str)

  #* dataloader
  parser.add_argument('--num_workers', default=4, type=int, help='')
  parser.add_argument('--image_size', default=224, type=int, help='')
  parser.add_argument('--load_gt_depth', default=False, type=bool, help='')

  #* specify data_dir
  parser.add_argument('--test_data_dir', default=['dataset/BP4D/test'], type=str, help='')
  parser.add_argument('--test_data_file', default=['dataset/BP4D/test.txt'], type=str, help='')

  #* backbone
  parser.add_argument('--patch_size', default=4, type=int, help='')
  parser.add_argument('--embed_dim', default=864, type=int, help='')
  parser.add_argument('--depths', default=[2,2,6,2], type=list, help='')
  parser.add_argument('--num_heads', default=[3,6,12,24 ], type=list, help='')
  parser.add_argument('--num_cap_part', default=6, type=int, help='')
  parser.add_argument('--num_cap_obj', default=1, type=int, help='')

  #* renderer
  parser.add_argument('--map_size', default=64, type=int, help='')
  parser.add_argument('--min_depth', default=0.9, type=float, help='')
  parser.add_argument('--max_depth', default=1.1, type=float, help='')
  parser.add_argument('--xyz_rotation_range', default=80, type=float, help='')
  parser.add_argument('--xy_translation_range', default=0.1, type=float, help='')
  parser.add_argument('--z_translation_range', default=0.1, type=float, help='')
  parser.add_argument('--rot_center_depth', default=1.0, type=float, help='')
  parser.add_argument('--fov', default=10, type=float, help='')
  parser.add_argument('--tex_cube_size', default=2, type=int, help='')
  parser.add_argument('--archive_code', default=False, type=bool, help='')

  main(args = parser.parse_args())