import os
from options.config_argument_parser import ConfigArgumentParser


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def get_train_model_dir(args):
    if args.checkpoints_dir is not None:
        if not os.path.exists(args.checkpoints_dir):
            os.system('mkdir -p ' + args.checkpoints_dir)
    else:
        checkpoints_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
        if not os.path.exists(checkpoints_dir):
            os.system('mkdir -p ' + checkpoints_dir)
        args.checkpoints_dir = checkpoints_dir


def prepare_train_args():
    train_parser = ConfigArgumentParser()
    train_parser.add_override_argument('--seed', type=int,
                                       help='a random seed')
    train_parser.add_override_argument('--gpus', nargs='+', type=int,
                                       help='numbers of GPU')
    train_parser.add_override_argument('--epochs', type=int,
                                       help='total epochs')
    train_parser.add_override_argument('--batch_size', type=int,
                                       help='batch size')
    train_parser.add_override_argument('--lr', type=float,
                                       help='learning rate')
    train_parser.add_override_argument('--momentum', type=float,
                                       help='momentum for sgd, alpha parameter for adam')
    train_parser.add_override_argument('--beta', default=0.999, type=float,
                                       help='beta parameters for adam')
    train_parser.add_override_argument('--weight_decay', '--wd', type=float,
                                       help='weight decay')
    train_parser.add_override_argument('--save_prefix', type=str,
                                       help='some comment for model or test result dir')
    train_parser.add_override_argument('--model_type', type=str,
                                       help='used in model_interface.py')
    train_parser.add_override_argument('--is_load_strict', action='store_false',
                                       help='allow to load only common state dicts')
    train_parser.add_override_argument('--is_load_pretrained_weight', action='store_true',
                                       help='True means try to load pretrained weights')
    train_parser.add_override_argument('--pretrained_weights_path', type=str,
                                       help='pretrained weights path')
    train_parser.add_override_argument('--is_resuming_training', action='store_true',
                                       help='True means try to resume previous train')
    train_parser.add_override_argument('--checkpoint_path', type=str,
                                       help='checkpoints path')
    train_parser.add_override_argument('--dataset_dir', type=str,
                                       help='dataset directory')
    train_parser.add_override_argument('--checkpoints_dir', type=str,
                                       help='checkpoints directory')
    args = train_parser.parse_args()
    get_train_model_dir(args)
    save_args(args, args.checkpoints_dir)
    return args


def prepare_eval_args():
    eval_parser = ConfigArgumentParser()
    eval_parser.add_override_argument('--seed', type=int,
                                      help='a random seed')
    eval_parser.add_override_argument('--gpus', nargs='+', type=int,
                                      help='numbers of GPU')
    eval_parser.add_override_argument('--model_type', type=str,
                                      help='used in model_interface.py')
    eval_parser.add_override_argument('--weights_path', type=str,
                                      help='weights path')
    eval_parser.add_override_argument('--dataset_dir', type=str,
                                      help='dataset directory')
    eval_parser.add_override_argument('--submission_file_path', type=str,
                                      help='submission.csv path')
    args = eval_parser.parse_args()
    return args


def prepare_split_dataset_args():
    split_parser = ConfigArgumentParser()
    split_parser.add_override_argument('--seed', type=int,
                                       help='a random seed')
    split_parser.add_override_argument('--valid_ratio', type=float,
                                       help='valid ratio')
    split_parser.add_override_argument('--dataset_dir', type=str,
                                       help='dataset directory')
    args = split_parser.parse_args()
    save_args(args, args.dataset_dir)
    return args
