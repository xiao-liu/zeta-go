# -*- coding: utf-8 -*-

import glog as log
import numpy as np
import torch

from play import self_play


class ExamplePool:

    def __init__(self, conf):
        self.conf = conf

        # the list that stores examples
        self.example_pool = []

        # the length (i.e., number of examples) of each game
        self.game_length = []

        # the random permutation used to shuffle the examples
        self.permutation = None

        # records the current position when traversing the examples
        self.pos = 0

    def generate_examples(self, network, device):
        for i in range(self.conf.GAMES_PER_ITERATION):
            log.info(f'starting self-play {i}...')
            examples = self_play(network, device, self.conf)
            log.info(f'{len(examples)} examples generated in self-play {i}')
            self.example_pool += examples
            self.game_length.append(len(examples))

        # discard old examples when pool is full
        if len(self.game_length) > self.conf.EXAMPLE_POOL_SIZE:
            m = len(self.game_length) - self.conf.EXAMPLE_POOL_SIZE
            n = sum(self.game_length[:m])
            self.example_pool = self.example_pool[n:]
            self.game_length = self.game_length[m:]

    def shuffle(self):
        self.permutation = np.random.permutation(len(self.example_pool))
        self.pos = 0

    def has_batch(self):
        return self.pos + self.conf.BATCH_SIZE < len(self.example_pool)

    def load_batch(self):
        features, pi, z = [], [], []
        for i in self.permutation[self.pos:self.pos+self.conf.BATCH_SIZE]:
            features.append(torch.from_numpy(self.example_pool[i][0]))
            pi.append(torch.from_numpy(self.example_pool[i][1]))
            z.append(torch.from_numpy(self.example_pool[i][2]))
        features = torch.stack(features)
        pi = torch.stack(pi)
        z = torch.stack(z)
        self.pos += self.conf.BATCH_SIZE
        return features, pi, z
