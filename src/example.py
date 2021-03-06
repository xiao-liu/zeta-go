# -*- coding: utf-8 -*-

import glog as log
import numpy as np
import torch
from torch.multiprocessing import Process, SimpleQueue, Value

from evaluate import BulkEvaluatorManager, DefaultEvaluator
from play import self_play
from resign import ResignManager


class ExamplePool:

    def __init__(self, model_dir, conf):
        self.model_dir = model_dir
        self.conf = conf

        # the list that stores examples
        self.example_pool = []

        # the length (i.e., number of examples) of each game
        self.game_length = []

        # number of remaining games
        # used to resume generation from an interrupted one
        self.remaining_games = 0

        # the random permutation used to shuffle the examples
        self.permutation = None

        # records the current position when traversing the examples
        self.pos = 0

        # resignation manager
        self.resign_mgr = ResignManager(self.conf)

    def state_dict(self):
        return {
            'example_pool': self.example_pool,
            'game_length': self.game_length,
            'remaining_games': self.remaining_games,
            'permutation': self.permutation,
            'pos': self.pos,
            'resign_mgr': self.resign_mgr,
        }

    def save(self, iteration):
        torch.save(
            self.state_dict(),
            f'{self.model_dir}/example_pool_checkpoint_[iter_{iteration}].pt',
        )

    def load(self, state_dict):
        self.example_pool = state_dict['example_pool']
        self.game_length = state_dict['game_length']
        self.remaining_games = state_dict['remaining_games']
        self.permutation = state_dict['permutation']
        self.pos = state_dict['pos']
        self.resign_mgr = state_dict['resign_mgr']

    def prepare_generation(self):
        self.remaining_games = self.conf.GAMES_PER_ITERATION
        self.permutation = None
        self.pos = 0

    def _worker_job(self, worker_id, num_games, num_active_workers,
                    resign_threshold, evaluator, output_queue):
        for i in range(num_games):
            log.info(f'worker {worker_id}: starting self-play {i}...')
            examples, resign_value_history, result = self_play(
                evaluator, resign_threshold.value, self.conf)
            output_queue.put((examples, resign_value_history, result))
            log.info(
                f'worker {worker_id}: {len(examples)} examples generated '
                f'in self-play {i}'
            )
        num_active_workers.value -= 1

    def _generate_parallel(self, iteration, network, device, num_workers):
        q, r = divmod(self.remaining_games, num_workers)
        num_active_workers = Value('i', num_workers)
        resign_threshold = Value('d', self.resign_mgr.threshold())
        evaluator_mgr = BulkEvaluatorManager([network], device, num_workers)
        output_queue = SimpleQueue()

        # start the workers
        workers = []
        for worker_id in range(num_workers):
            num_games = q + 1 if worker_id < r else q
            evaluator = evaluator_mgr.get_evaluator(worker_id, 0)
            worker = Process(
                target=self._worker_job,
                args=(worker_id, num_games, num_active_workers,
                      resign_threshold, evaluator, output_queue),
            )
            workers.append(worker)
            worker.start()

        # start evaluator server
        server = evaluator_mgr.get_server(num_active_workers)
        server.start()

        # collect the examples generated by workers
        while num_active_workers.value > 0 or not output_queue.empty():
            examples, resign_value_history, result = output_queue.get()
            self.example_pool += examples
            self.game_length.append(len(examples))

            # add the history into resignation manager to update the threshold
            if resign_value_history is not None:
                self.resign_mgr.add(resign_value_history, result)
                resign_threshold.value = self.resign_mgr.threshold()

            self.remaining_games -= 1

            # periodically save the progress
            if (self.conf.GAMES_PER_ITERATION - self.remaining_games) \
                    % self.conf.EXAMPLE_POOL_SAVE_FREQUENCY == 0:
                self.save(iteration)
                log.info(
                    f'[iter={iteration}] ExamplePool: checkpoint saved, '
                    f'{self.remaining_games} games remaining'
                )

        for worker in workers:
            worker.join()
        server.join()

    def _generate(self, iteration, network, device):
        evaluator = DefaultEvaluator(network, device)
        for i in range(self.remaining_games):
            log.info(f'starting self-play {i}...')
            examples, resign_value_history, result = self_play(
                evaluator, self.resign_mgr.threshold(), self.conf)
            log.info(f'{len(examples)} examples generated in self-play {i}')
            self.example_pool += examples
            self.game_length.append(len(examples))

            # add the history into resignation manager to update the threshold
            if resign_value_history is not None:
                self.resign_mgr.add(resign_value_history, result)

            self.remaining_games -= 1

            # periodically save the progress
            if (self.conf.GAMES_PER_ITERATION - self.remaining_games) \
                    % self.conf.EXAMPLE_POOL_SAVE_FREQUENCY == 0:
                self.save(iteration)
                log.info(
                    f'[iter={iteration}] ExamplePool: checkpoint saved, '
                    f'{self.remaining_games} games remaining'
                )

    def generate_examples(self, iteration, network, device, num_workers):
        if num_workers > 1:
            self._generate_parallel(iteration, network, device, num_workers)
        else:
            self._generate(iteration, network, device)

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
