# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from glob import glob
import os
import sys
import torch.multiprocessing as mp

from config import CONFIGURATIONS
from play import play_against_human
from train import train


def process_train():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            f'python {sys.argv[0]} train [--model_name MODEL_NAME] '
            f'[--config CONFIG] [--num_workers NUM_WORKERS]\n       '
            f'python {sys.argv[0]} train [-h]\n'
        )
    )
    sub_parser.add_argument(
        '--model_name',
        type=str,
        default='',
        help='name of the model, will use timestamp as name if not specified',
    )
    sub_parser.add_argument(
        '--config',
        type=str,
        default='19x19',
        help=(
            'configuration for the training, must be one of the configurations '
            'defined in config.py (default: 19x19)'
        ),
    )
    sub_parser.add_argument(
        '--num_workers',
        type=int,
        default=4*mp.cpu_count(),
        help=(
            f'number of processes to perform self-play and compare networks '
            f'(default: 4 * num_of_cpus = {4 * mp.cpu_count()})'
        ),
    )
    sub_args = sub_parser.parse_args(sys.argv[2:])

    if sub_args.config not in CONFIGURATIONS:
        print(f'configuration {sub_args.config} not found')
        exit(-1)

    model_name = datetime.now().strftime('%Y-%m-%d_%H%M%S') \
        if sub_args.model_name == '' else sub_args.model_name
    model_dir = os.path.abspath(
        os.path.join(os.getcwd(), f'../models/{model_name}'))
    if os.path.exists(model_dir):
        print(f'directory {model_dir} already exists')
        exit(-1)
    os.makedirs(model_dir, exist_ok=True)

    train(model_dir, sub_args.config, sub_args.num_workers)


def process_resume():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            f'python {sys.argv[0]} resume <model_name> '
            f'[--checkpoint CHECKPOINT] [--config CONFIG] '
            f'[--num_workers NUM_WORKERS]\n       '
            f'python {sys.argv[0]} resume [-h]\n'
        )
    )
    sub_parser.add_argument(
        'model_name',
        type=str,
        help='name of the model to resume training',
    )
    sub_parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help=(
            'name of the checkpoint file, '
            'will load the latest checkpoint if not specified'
        ),
    )
    sub_parser.add_argument(
        '--config',
        type=str,
        default='',
        help=(
            f'load the specified configuration from config.py and '
            f'override the one saved in the checkpoint'
        ),
    )
    sub_parser.add_argument(
        '--num_workers',
        type=int,
        default=4*mp.cpu_count(),
        help=(
            f'number of processes to perform self-play and compare networks, '
            f'(default: 4 * num_of_cpus = {4 * mp.cpu_count()})'
        ),
    )
    sub_args = sub_parser.parse_args(sys.argv[2:])

    model_dir = os.path.abspath(
        os.path.join(os.getcwd(), f'../models/{sub_args.model_name}'))
    if not os.path.isdir(model_dir):
        print(f'directory {model_dir} not found')
        exit(-1)

    if sub_args.checkpoint == '':
        # load the latest checkpoint
        checkpoint_files = list(
            filter(
                lambda x: os.path.isfile(x),
                glob(f'{model_dir}/checkpoint_*.pt'),
            )
        )
        if len(checkpoint_files) == 0:
            print(f'no checkpoint file found in {model_dir}')
            exit(-1)
        checkpoint_file = max(checkpoint_files, key=os.path.getmtime)
    else:
        checkpoint_file = f'{model_dir}/{sub_args.checkpoint}'
        if not os.path.isfile(checkpoint_file):
            print(f'checkpoint file {checkpoint_file} not found')
            exit(-1)

    if sub_args.config != '' and sub_args.config not in CONFIGURATIONS:
        print(f'configuration {sub_args.config} not found')
        sub_args.config = ''

    # resume training
    train(model_dir, sub_args.config, sub_args.num_workers,
          checkpoint_file=checkpoint_file)


def process_play():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            f'python {sys.argv[0]} play <model_name> '
            f'[--black_player BLACK_PLAYER]\n       '
            f'python {sys.argv[0]} play [-h]\n'
        )
    )
    sub_parser.add_argument(
        'model_name',
        type=str,
        help='name of the model to play against',
    )
    sub_parser.add_argument(
        '--black_player',
        type=str,
        default='human',
        help=(
            'player who plays black and moves first, '
            'should be one of human/computer (default: human)'
        ),
    )
    sub_args = sub_parser.parse_args(sys.argv[2:])

    model_file = os.path.abspath(
        os.path.join(os.getcwd(), f'../models/{sub_args.model_name}/model.pt'))
    if not os.path.isfile(model_file):
        print(f'model file {model_file} not found')
        exit(-1)

    sub_args.black_player = sub_args.black_player.lower()
    if sub_args.black_player not in ('human', 'computer'):
        print('illegal black_player, set it to human')
        sub_args.black_player = 'human'

    play_against_human(model_file, sub_args.black_player == 'human')


def main():
    parser = argparse.ArgumentParser(
        usage=(
            f'python {sys.argv[0]} <command> [<args>]...\n       '
            f'python {sys.argv[0]} [-h]\n\n'
            f'Currently supported commands:\n'
            f'    train    Train a model\n'
            f'    resume   Resume training from a checkpoint\n'
            f'    play     Play Go with computer\n\n'
            f'Type "python {sys.argv[0]} <command> -h" to show help message '
            f'for each command.\n'
        )
    )
    parser.add_argument(
        'command',
        type=str,
        help='command to run, must be one of train/resume/play',
    )
    args = parser.parse_args(sys.argv[1:2])

    if args.command == 'train':
        process_train()
    elif args.command == 'resume':
        process_resume()
    elif args.command == 'play':
        process_play()
    else:
        print(f'unrecognized command: {args.command}')
        exit(-1)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
