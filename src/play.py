# -*- coding: utf-8 -*-

import numpy as np
import torch

from go import BLACK, WHITE
from gui import GUI
from mcts import TreeNode, tree_search
from network import ZetaGoNetwork
from predict import extract_feature


def self_play(network, device, conf):
    examples = []

    # create a search tree
    root = TreeNode(None, None, network, device, conf)

    previous_action = None
    t = 0
    while t < conf.MAX_GAME_LENGTH:
        # perform MCTS
        for _ in range(conf.NUM_SIMULATIONS):
            tree_search(root, network, device, conf)

        # calculate the distribution of action selection
        # notice that illegal actions always have zero probability as
        # long as NUM_SIMULATION > 0
        if t < conf.EXPLORATION_TIME:
            # temperature tau = 1
            s = sum(root.n)
            pi = [x / s for x in root.n]
        else:
            # temperature tau -> 0
            m = max(root.n)
            p = [0 if x < m else 1 for x in root.n]
            s = sum(p)
            pi = [x / s for x in p]

        # save position, distribution of action selection and turn
        examples.append(
            [
                extract_feature(root, conf),
                np.array(pi, dtype=np.float32),
                np.array([root.go.turn], dtype=np.float32),
            ]
        )

        # choose an action
        action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        root = root.children[action]

        # release memory
        root.parent.children = None

        t += 1

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            break
        previous_action = action

    score_black, score_white = root.go.score()
    result = 1.0 if score_black > score_white else -1.0

    # update the the game winner from the perspective of each player
    for i in range(len(examples)):
        examples[i][2] *= result

    return examples


def play_against_network(network, opponent_network, device, color, conf):
    # networks[0] for black player, networks[1] for white player
    networks = [network, opponent_network]
    if color == WHITE:
        networks[0], networks[1] = networks[1], networks[0]
    
    # create search trees for both players
    roots = [None, None]
    for i in range(2):
        roots[i] = TreeNode(None, None, networks[i], device, conf)

    # black player goes first (0 for black, 1 for white)
    player = 0

    previous_action = None
    t = 0
    while t < conf.MAX_GAME_LENGTH:
        # perform MCTS
        for _ in range(conf.NUM_SIMULATIONS):
            tree_search(roots[player], networks[player], device, conf)

        # calculate the distribution of action selection
        # temperature tau -> 0
        m = max(roots[player].n)
        p = [0 if x < m else 1 for x in roots[player].n]
        s = sum(p)
        pi = np.array([x / s for x in p], dtype=np.float32)

        # choose an action
        action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        for i in range(2):
            if roots[i].children[action] is None:
                roots[i].children[action] = \
                    TreeNode(roots[i], action, networks[i], device, conf)
            roots[i] = roots[i].children[action]

            # release memory
            roots[i].parent.children = None

        t += 1

        # switch to the other player
        player = 1 - player

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            break
        previous_action = action

    score_black, score_white = roots[0].go.score()

    return (score_black > score_white) == (color == BLACK)


def play_against_human(model_file, human_plays_black):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_file)
    conf = model['conf']

    # load the network
    network = ZetaGoNetwork(conf)
    network.load_state_dict(model['best_network'])
    network.to(device)

    # create a search tree
    root = TreeNode(None, None, network, device, conf)

    gui = GUI(conf)

    human_turn = human_plays_black
    previous_action = None
    while True:
        if human_turn:
            # wait for human player's action
            action = gui.wait_for_action(root.go)
        else:
            # calculate computer's action
            gui.update_text('Computer is thinking...')

            # perform MCTS
            for _ in range(conf.NUM_SIMULATIONS):
                tree_search(root, network, device, conf)

            # calculate the distribution of action selection
            # temperature tau -> 0
            m = max(root.n)
            p = [0 if x < m else 1 for x in root.n]
            s = sum(p)
            pi = np.array([x / s for x in p], dtype=np.float32)

            # choose an action
            action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        if root.children[action] is None:
            root.children[action] = \
                TreeNode(root, action, network, device, conf)
        root = root.children[action]

        # release memory
        root.parent.children = None

        # update GUI
        gui.update_go(root.go)
        gui.update_text('Computer passes' if action == conf.PASS else '')

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            black_score, white_score = root.go.score()
            winner = 'BLACK' if black_score > white_score else 'WHITE'
            gui.update_text(f'{winner} wins, {black_score} : {white_score}')
            gui.freeze()

        previous_action = action
        human_turn = not human_turn
