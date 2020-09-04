# -*- coding: utf-8 -*-

import torch
from torch.multiprocessing import Pipe, Process, Queue
import torch.nn.functional as F
import queue


class DefaultEvaluator:

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def evaluate(self, feature):
        features = torch.stack([torch.from_numpy(feature)]).to(self.device)
        self.network.eval()
        with torch.no_grad():
            log_p, v = self.network(features)
            p = F.softmax(log_p, dim=1)
            p = p.cpu().numpy()
            v = v.cpu().numpy()
            return p[0], v[0]


class BulkEvaluator:

    def __init__(self, evaluator_id, network_id, job_queue, conn):
        self.evaluator_id = evaluator_id
        self.network_id = network_id
        self.job_queue = job_queue
        self.conn = conn

    def evaluate(self, feature):
        self.job_queue.put((feature, self.network_id, self.evaluator_id))
        return self.conn.recv()


class BulkEvaluatorManager:

    def __init__(self, networks, device, num_evaluators, timeout=5):
        self.networks = networks
        self.device = device
        self.timeout = timeout
        self.job_queue = Queue()
        self.parent_conns = []
        self.child_conns = []
        for i in range(num_evaluators):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def server_job(self, num_active_workers):
        num_networks = len(self.networks)
        while num_active_workers.value > 0:
            features = [[] for _ in range(num_networks)]
            conns = [[] for _ in range(num_networks)]
            for _ in range(num_active_workers.value):
                try:
                    feature, network_id, evaluator_id = \
                        self.job_queue.get(timeout=self.timeout)
                    features[network_id].append(torch.from_numpy(feature))
                    conns[network_id].append(self.parent_conns[evaluator_id])
                except queue.Empty:
                    break
            for network_id in range(num_networks):
                if len(features[network_id]) == 0:
                    continue

                network = self.networks[network_id]
                network.eval()
                with torch.no_grad():
                    log_p, v = network(
                        torch.stack(features[network_id]).to(self.device))
                    p = F.softmax(log_p, dim=1)
                    p = p.cpu().numpy()
                    v = v.cpu().numpy()

                for i in range(len(conns[network_id])):
                    conns[network_id][i].send((p[i], v[i]))

    def get_server(self, num_active_workers):
        return Process(target=self.server_job, args=(num_active_workers,))

    def get_evaluator(self, evaluator_id, network_id):
        return BulkEvaluator(
            evaluator_id,
            network_id,
            self.job_queue,
            self.child_conns[evaluator_id],
        )
