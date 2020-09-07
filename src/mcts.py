# -*- coding: utf-8 -*-

import numpy as np

from go import Go
from predict import predict


# define the Monte Carlo search tree node
# the definitions of n, w, q and p are the same as that in the paper
class TreeNode:

    def __init__(self, parent, action, evaluator, conf):
        if parent is None:
            self.go = Go(board_size=conf.BOARD_SIZE, komi=conf.KOMI)
        else:
            self.go = Go(copy=parent.go)
            if action == conf.PASS:
                self.go.pass_()
            else:
                self.go.play(*divmod(action, conf.BOARD_SIZE))
        self.parent = parent
        self.children = [None] * conf.NUM_ACTIONS
        self.action = action
        self.n = np.zeros(conf.NUM_ACTIONS, dtype=np.int)
        self.w = np.zeros(conf.NUM_ACTIONS, dtype=np.float32)
        self.p, self.v = predict(evaluator, self, conf, random_trans=True)


def tree_search(root, evaluator, conf):
    node = root

    # prepare Dirichlet noise for the root node
    num_legal_actions_of_root = 0
    for action in range(conf.NUM_ACTIONS - 1):
        if root.go.legal_play(*divmod(action, conf.BOARD_SIZE)):
            num_legal_actions_of_root += 1
    noise = list(np.random.dirichlet(
        np.full(num_legal_actions_of_root, conf.DIRICHLET_ALPHA)))

    # select
    while True:
        # break if node is in a terminated state (i.e., the game ends)
        if node.action == conf.PASS \
                and node.parent is not None \
                and node.parent.action == conf.PASS:
            break

        # find actions with maximum upper confidence bound
        best_action = None
        best_ucb = 0.0
        sqrt_sum_n = np.sqrt(sum(node.n))
        for action in range(conf.NUM_ACTIONS):
            # skip illegal actions, notice that pass is always legal
            if action != conf.PASS and not node.go.legal_play(
                    *divmod(action, conf.BOARD_SIZE)):
                continue
            q = 0.0 if node.n[action] == 0 else node.w[action] / node.n[action]
            p = node.p[action]
            if node == root and action != conf.PASS:
                # introduce additional Dirichlet noise for the root
                p += conf.DIRICHLET_EPSILON * (noise.pop() - p)
            ucb = q + conf.C_PUCT * p * sqrt_sum_n / (1.0 + node.n[action])

            # there can be multiple best actions with super rare
            # possibility, we will ignore this
            if best_action is None or ucb > best_ucb:
                best_action = action
                best_ucb = ucb

        if node.children[best_action] is not None:
            node = node.children[best_action]
        else:
            # reach a leaf node, evaluate and expand
            node.children[best_action] = \
                TreeNode(node, best_action, evaluator, conf)
            node = node.children[best_action]
            break

    # backup
    # notice that it is necessary to alternate the sign of v because one
    # player's win is another player's loss, and vice versa
    # the AlphaGo Zero paper does not emphasize this
    v = node.v
    while node != root:
        v = -v
        node.parent.n[node.action] += 1
        node.parent.w[node.action] += v
        node = node.parent
