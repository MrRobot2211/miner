import argparse

import arguments
from src import utils
from src.trainer import Trainer
from src.trainer_fastformer import Trainer as TrainerFast

def _train(args):
    trainer = Trainer(args)
    trainer.train()

def _train_fast(args):
    trainer = TrainerFast(args)
    trainer.train()


def _eval(args):
    trainer = Trainer(args)
    trainer.eval()


def _eval_fast(args):
    trainer = TrainerFast(args)
    trainer.eval()



def main():
    parser = argparse.ArgumentParser(description='Arguments for Miner model', fromfile_prefix_chars='@',
                                     allow_abbrev=False)
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
    subparsers = parser.add_subparsers(dest='mode', help='Mode of the process: train or test')

    train_parser = subparsers.add_parser('train', help='Training phase')
    arguments.add_train_arguments(train_parser)
    train_parser = subparsers.add_parser('train_fastformer', help='Training fastformerphase')
    arguments.add_train_arguments(train_parser)
    eval_parser = subparsers.add_parser('eval', help='Evaluation phase')
    arguments.add_eval_arguments(eval_parser)
    eval_parser = subparsers.add_parser('eval_fastformer', help='Evaluation phase')
    arguments.add_eval_arguments(eval_parser)

    args = parser.parse_args()
    if args.mode == 'train':
        _train(args)
    elif args.mode == 'train_fastformer':
        _train_fast(args)
    elif args.mode == 'eval':
        _eval(args)
    elif args.mode == 'eval_fastformer':
        _eval_fast(args)

if __name__ == '__main__':
    main()
