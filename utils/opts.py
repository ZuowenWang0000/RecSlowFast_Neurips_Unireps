import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of hGRU")

# ========================= MLflow(logging) configs ==========================
parser.add_argument('--exp_name', type=str, default="experiment")
parser.add_argument('--run_name', type=str, default="run")
parser.add_argument('--seed', type=int, default=0)

# ========================= checkpoint configs ==========================
parser.add_argument('--finetune', default=False, action='store_true')
parser.add_argument('--checkpoint', type=str, default=None)

# ========================= Data Configs ==========================
parser.add_argument('--pf_root', type=str, 
    default='./pathfinder_no_crossing_10kvid') #tp-easy
parser.add_argument('--train_list', type=str, default="train_vid_id_list.txt")
parser.add_argument('--test_list', type=str, default="test_vid_id_list.txt")

parser.add_argument('--name', type=str, default="hgrufs")
parser.add_argument('--model', type=str, default="hgrufs")
parser.add_argument('--algo', type=str, default="bptt")
parser.add_argument('--penalty', default=False, action='store_true')
parser.add_argument('--iou_th', type=float, default=0.5)


# ======================= Model Configs ========================
parser.add_argument('--filt_size', type=int, default=15)
parser.add_argument('--random_init', default=False, action='store_true')

# ======================= Optimization Configs ========================
parser.add_argument('--stateful', default=False, action='store_true')
parser.add_argument('--train_timesteps', type=int, default=6)
parser.add_argument('--slowsteps', type=int, default=6)
parser.add_argument('--faststeps', type=int, default=1)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[25,50,75], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

# ========================= Monitor Configs ==========================
parser.add_argument('--print_freq', '-p', default=200, type=int,metavar='N', help='print frequency (default: 50)')
parser.add_argument('--val_freq', '-ef', default=5, type=int,metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--test_freq', '-sf', default=10, type=int, metavar='N', help='test frequency (default: 10)')

parser.add_argument('-parallel', '--parallel', default= False, action='store_true', help='Wanna parallelize the training')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')