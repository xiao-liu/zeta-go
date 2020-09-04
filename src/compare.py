# -*- coding: utf-8 -*-

from torch.multiprocessing import Process, Value

from evaluate import BulkEvaluatorManager, DefaultEvaluator
from go import BLACK, WHITE
from play import play_against_network


class Comparator:

    def __init__(self, conf):
        self.conf = conf

    def _worker_job(self, num_games, num_active_workers,
                    evaluator, opponent_evaluator, color, score):
        for i in range(num_games):
            if play_against_network(
                    evaluator, opponent_evaluator, color, self.conf):
                score.value += 1
            color = -color
        num_active_workers.value -= 1

    def _compare_parallel(self, network, opponent_network, device, num_workers):
        q, r = divmod(self.conf.GAMES_PER_COMPARISON, num_workers)
        num_active_workers = Value('i', num_workers)
        evaluator_mgr = BulkEvaluatorManager(
            [network, opponent_network], device, num_workers)
        score = Value('i', 0)

        workers = []
        s = 0
        for worker_id in range(num_workers):
            num_games = q + 1 if worker_id < r else q
            evaluator = evaluator_mgr.get_evaluator(worker_id, 0)
            opponent_evaluator = evaluator_mgr.get_evaluator(worker_id, 1)
            color = BLACK if s % 2 == 0 else WHITE
            s += num_games
            worker = Process(
                target=self._worker_job,
                args=(num_games, num_active_workers,
                      evaluator, opponent_evaluator, color, score),
            )
            workers.append(worker)
            worker.start()

        # start evaluator server
        server = evaluator_mgr.get_server(num_active_workers)
        server.start()

        for worker in workers:
            worker.join()
        server.join()

        return score.value / self.conf.GAMES_PER_COMPARISON

    def _compare(self, network, opponent_network, device):
        evaluator = DefaultEvaluator(network, device)
        opponent_evaluator = DefaultEvaluator(opponent_network, device)
        color = BLACK
        score = 0
        for i in range(self.conf.GAMES_PER_COMPARISON):
            if play_against_network(
                    evaluator, opponent_evaluator, color, self.conf):
                score += 1
            color = -color
        return score / self.conf.GAMES_PER_COMPARISON

    def estimate_win_rate(self, network, opponent_network, device, num_workers):
        if num_workers > 1:
            return self._compare_parallel(
                network, opponent_network, device, num_workers)
        else:
            return self._compare(network, opponent_network, device)
