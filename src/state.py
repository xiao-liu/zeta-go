# -*- coding: utf-8 -*-

import torch
import torch.optim as optim

from compare import Comparator
from config import get_conf
from example import ExamplePool
from network import ZetaGoNetwork

STAGE_UNKNOWN = 0
STAGE_GENERATING = 1
STAGE_TRAINING = 2


class State:

    def __init__(self):
        self.model_dir = ''
        self.conf = None
        self.stage = STAGE_UNKNOWN
        self.iteration = 0
        self.step = 0
        self.network = None
        self.best_network = None
        self.optimizer = None
        self.example_pool = None
        self.comparator = None

    def create(self, model_dir, conf_name, device):
        self.model_dir = model_dir

        self.conf = get_conf(conf_name)
        self.stage = STAGE_GENERATING
        self.iteration = 0
        self.step = 0

        # randomly initialize a network and let it be the best network
        self.network = ZetaGoNetwork(self.conf)
        self.best_network = ZetaGoNetwork(self.conf)
        self.best_network.load_state_dict(self.network.state_dict())
        self.network.to(device)
        self.best_network.to(device)

        # setup the optimizer
        # notice that the L2 regularization is implemented by
        # introducing a weight decay
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2*self.conf.L2_REG,
        )

        # create an example pool
        self.example_pool = ExamplePool(self.model_dir, self.conf)
        self.example_pool.prepare_generation()

        # create a comparator to compare two networks
        self.comparator = Comparator(self.conf)

    def load(self, checkpoint_file, model_dir, conf_name, device):
        checkpoint = torch.load(checkpoint_file)

        self.model_dir = model_dir
        if conf_name != '':
            self.conf = get_conf(conf_name)
        else:
            self.conf = checkpoint['conf']
        self.stage = checkpoint['stage']
        self.iteration = checkpoint['iteration']
        self.step = checkpoint['step']

        self.network = ZetaGoNetwork(self.conf)
        self.network.load_state_dict(checkpoint['network'])
        self.best_network = ZetaGoNetwork(self.conf)
        self.best_network.load_state_dict(checkpoint['best_network'])
        self.network.to(device)
        self.best_network.to(device)

        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2*self.conf.L2_REG,
        )
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.example_pool = ExamplePool(self.model_dir, self.conf)
        if self.stage == STAGE_GENERATING:
            self.example_pool.load(
                torch.load(
                    f'{self.model_dir}/'
                    f'example_pool_checkpoint_[iter_{self.iteration}].pt'
                )
            )
        else:
            self.example_pool.load(checkpoint['example_pool'])

        self.comparator = Comparator(self.conf)

    def save(self):
        torch.save(
            {
                'conf': self.conf,
                'iteration': self.iteration,
                'step': self.step,
                'network': self.network.state_dict(),
                'best_network': self.best_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'example_pool': self.example_pool.state_dict(),
                'stage': self.stage,
            },
            f'{self.model_dir}/'
            f'checkpoint_[iter_{self.iteration}][step_{self.step}].pt',
        )
