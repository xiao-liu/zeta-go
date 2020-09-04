# -*- coding: utf-8 -*-

import collections


CONFIGURATIONS = {
    '19x19': {
        # ---- basic game settings ----
        # size of the board
        'BOARD_SIZE': 19,

        # komi
        'KOMI': 7.5,

        # ---- neural network settings ----
        # number of historical positions contained in the input
        'HISTORY_LENGTH': 3,

        # kernel size, stride and padding of the convolutional block
        'CONV_KERNEL_SIZE': 3,
        'CONV_STRIDE': 1,
        'CONV_PADDING': 1,

        # number of filters in the residual blocks
        'RESIDUAL_FILTERS': 256,

        # kernel size, stride and padding of the convolution in the
        # residual blocks
        'RESIDUAL_KERNEL_SIZE': 3,
        'RESIDUAL_STRIDE': 1,
        'RESIDUAL_PADDING': 1,

        # number of residual blocks
        # 19 for small instance, 39 for large instance
        'RESIDUAL_BLOCKS': 39,

        # number of filters in the policy head
        'POLICY_FILTERS': 2,

        # kernel size, stride and padding of the convolution in the
        # policy head
        'POLICY_KERNEL_SIZE': 1,
        'POLICY_STRIDE': 1,
        'POLICY_PADDING': 0,

        # number of filters in the value head
        'VALUE_FILTERS': 1,

        # kernel size, stride, and padding of the convolution in the
        # value head
        'VALUE_KERNEL_SIZE': 1,
        'VALUE_STRIDE': 1,
        'VALUE_PADDING': 0,

        # number of neurons in the hidden layer of the value head
        'VALUE_HIDDEN_LAYER_SIZE': 256,

        # ---- training settings ----
        # total number of self-play games during the training course
        # 4900000 for small instance, 29000000 for large instance
        'TOTAL_GAMES': 29000000,

        # number of self-play games during each iteration
        'GAMES_PER_ITERATION': 25000,

        # batch size
        'BATCH_SIZE': 2048,

        # L2 regularization parameter
        'L2_REG': 0.0001,

        # learning rate schedule
        # lr = 0.01 for 0 <= step < 400000
        # lr = 0.001 for 400000 <= step < 600000
        # lr = 0.0001 for 600000 <= step < infinity
        'LR_SCHEDULE': ((400000, 0.01), (600000, 0.001), (-1, 0.0001)),

        # temperature = 1 when t < EXPLORATION_TIME, and 0 afterward
        'EXPLORATION_TIME': 30,

        # number of simulation in each MCTS
        'NUM_SIMULATIONS': 1600,

        # the constant for the PUCT algorithm
        'C_PUCT': 0.1,

        # parameters for the Dirichlet noise
        'DIRICHLET_ALPHA': 0.03,
        'DIRICHLET_EPSILON': 0.25,

        # number of games played when comparing two networks
        'GAMES_PER_COMPARISON': 400,

        # the minimal win rate a new network must have in order for it
        # to be considered as the winner
        'WIN_RATE_MARGIN': 0.55,

        # maximum number of self-play games stored in the example pool
        # old games will be discarded if the pool is full
        'EXAMPLE_POOL_SIZE': 500000,

        # frequency of checkpoint
        'CHECKPOINT_FREQUENCY': 1000,
    },
    '9x9': {
        # ---- basic game settings ----
        'BOARD_SIZE': 9,
        'KOMI': 7.5,

        # ---- neural network settings ----
        'HISTORY_LENGTH': 3,
        'CONV_KERNEL_SIZE': 3,
        'CONV_STRIDE': 1,
        'CONV_PADDING': 1,
        'RESIDUAL_FILTERS': 128,
        'RESIDUAL_KERNEL_SIZE': 3,
        'RESIDUAL_STRIDE': 1,
        'RESIDUAL_PADDING': 1,
        'RESIDUAL_BLOCKS': 13,
        'POLICY_FILTERS': 2,
        'POLICY_KERNEL_SIZE': 1,
        'POLICY_STRIDE': 1,
        'POLICY_PADDING': 0,
        'VALUE_FILTERS': 1,
        'VALUE_KERNEL_SIZE': 1,
        'VALUE_STRIDE': 1,
        'VALUE_PADDING': 0,
        'VALUE_HIDDEN_LAYER_SIZE': 128,

        # ---- training settings ----
        'TOTAL_GAMES': 1000000,
        'GAMES_PER_ITERATION': 10000,
        'BATCH_SIZE': 64,
        'L2_REG': 0.0001,
        'LR_SCHEDULE': ((400000, 0.01), (600000, 0.001), (-1, 0.0001)),
        'EXPLORATION_TIME': 8,
        'NUM_SIMULATIONS': 360,
        'C_PUCT': 0.1,
        'DIRICHLET_ALPHA': 0.03,
        'DIRICHLET_EPSILON': 0.25,
        'GAMES_PER_COMPARISON': 100,
        'WIN_RATE_MARGIN': 0.55,
        'EXAMPLE_POOL_SIZE': 100000,
        'CHECKPOINT_FREQUENCY': 1000,
    },
}

Config = collections.namedtuple(
    'Config',
    [
        'BOARD_SIZE',
        'KOMI',
        'NUM_ACTIONS',
        'PASS',
        'HISTORY_LENGTH',
        'INPUT_CHANNELS',
        'CONV_KERNEL_SIZE',
        'CONV_STRIDE',
        'CONV_PADDING',
        'RESIDUAL_FILTERS',
        'RESIDUAL_KERNEL_SIZE',
        'RESIDUAL_STRIDE',
        'RESIDUAL_PADDING',
        'RESIDUAL_BLOCKS',
        'POLICY_FILTERS',
        'POLICY_KERNEL_SIZE',
        'POLICY_STRIDE',
        'POLICY_PADDING',
        'VALUE_FILTERS',
        'VALUE_KERNEL_SIZE',
        'VALUE_STRIDE',
        'VALUE_PADDING',
        'VALUE_HIDDEN_LAYER_SIZE',
        'TOTAL_GAMES',
        'GAMES_PER_ITERATION',
        'NUM_ITERATIONS',
        'BATCH_SIZE',
        'L2_REG',
        'LR_SCHEDULE',
        'MAX_GAME_LENGTH',
        'EXPLORATION_TIME',
        'NUM_SIMULATIONS',
        'C_PUCT',
        'DIRICHLET_ALPHA',
        'DIRICHLET_EPSILON',
        'GAMES_PER_COMPARISON',
        'WIN_RATE_MARGIN',
        'EXAMPLE_POOL_SIZE',
        'CHECKPOINT_FREQUENCY',
    ]
)


def get_conf(conf_name):
    conf = CONFIGURATIONS[conf_name]

    # number of actions (must be BOARD_SIZE ** 2 + 1)
    conf['NUM_ACTIONS'] = conf['BOARD_SIZE'] ** 2 + 1

    # 0 to NUM_ACTIONS - 2 represent normal moves, NUM_ACTIONS - 1
    # represents pass
    conf['PASS'] = conf['NUM_ACTIONS'] - 1

    # number of feature planes in the input (must be
    # 2 * HISTORY_LENGTH + 1)
    conf['INPUT_CHANNELS'] = 2 * conf['HISTORY_LENGTH'] + 1

    # number of iterations
    conf['NUM_ITERATIONS'] = conf['TOTAL_GAMES'] // conf['GAMES_PER_ITERATION']

    # maximum game length for self-playing and comparison
    # notice that the rule of Go does not preclude the possibility of
    # more than 2 * BOARD_SIZE ** 2 moves, but this rarely happens in
    # practice
    conf['MAX_GAME_LENGTH'] = 2 * conf['BOARD_SIZE'] ** 2

    return Config(**conf)
