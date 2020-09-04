# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

from go import BLACK, WHITE


# an implementation of dihedral group of order 8 (D4)
def dihedral_trans(x, trans, axes):
    if trans == 0:
        # identity
        return np.ascontiguousarray(x)
    elif trans == 1:
        # rotate counterclockwise by 90 degrees
        return np.ascontiguousarray(np.rot90(x, k=1, axes=axes))
    elif trans == 2:
        # rotate counterclockwise by 180 degrees
        return np.ascontiguousarray(np.rot90(x, k=2, axes=axes))
    elif trans == 3:
        # rotate counterclockwise by 270 degrees
        return np.ascontiguousarray(np.rot90(x, k=3, axes=axes))
    elif trans == 4:
        # reflect over x-axis
        return np.ascontiguousarray(np.flip(x, axis=axes[0]))
    elif trans == 5:
        # reflect over y-axis
        return np.ascontiguousarray(np.flip(x, axis=axes[1]))
    elif trans == 6:
        # reflect over x-axis and then rotate counterclockwise by
        # 90 degrees
        return np.ascontiguousarray(
            np.rot90(np.flip(x, axis=axes[0]), k=1, axes=axes))
    elif trans == 7:
        # reflect over y-axis and then rotate counterclockwise by
        # 90 degrees
        return np.ascontiguousarray(
            np.rot90(np.flip(x, axis=axes[1]), k=1, axes=axes))


# inverse dihedral transformations
def inverse_dihedral_trans(x, trans, axes):
    if trans == 1 or trans == 3:
        return dihedral_trans(x, 4 - trans, axes=axes)
    else:
        return dihedral_trans(x, trans, axes=axes)


def extract_feature(node, conf):
    feature = np.ndarray(
        shape=(conf.INPUT_CHANNELS, conf.BOARD_SIZE, conf.BOARD_SIZE),
        dtype=np.float32,
    )
    for x in range(conf.BOARD_SIZE):
        for y in range(conf.BOARD_SIZE):
            feature[conf.INPUT_CHANNELS - 1][x][y] = \
                1.0 if node.go.turn == BLACK else 0.0
    for i in range(conf.HISTORY_LENGTH):
        for x in range(conf.BOARD_SIZE):
            for y in range(conf.BOARD_SIZE):
                if node is None:
                    feature[2 * i][x][y] = 0.0
                    feature[2 * i + 1][x][y] = 0.0
                else:
                    feature[2 * i][x][y] = \
                        1.0 if node.go.board.color(x, y) == BLACK else 0.0
                    feature[2 * i + 1][x][y] = \
                        1.0 if node.go.board.color(x, y) == WHITE else 0.0
        if node is not None:
            node = node.parent
    return feature


def predict(network, device, node, conf, random_trans=False):
    network.eval()
    with torch.no_grad():
        if random_trans:
            # uniform at random choose a Dihedral transformation and
            # apply it to the feature
            trans = np.random.randint(8)
            feature = dihedral_trans(
                extract_feature(node, conf), trans, axes=(1, 2))
            features = torch.stack([torch.from_numpy(feature)]).to(device)

            log_p, v = network(features)

            # transform the distribution back
            p = F.softmax(log_p, dim=1)
            p = p.cpu().numpy()[0]
            v = v.cpu().numpy()[0]

            p_move, p_pass = p[:conf.BOARD_SIZE**2], p[conf.PASS]
            p_move = inverse_dihedral_trans(
                np.reshape(p_move, (conf.BOARD_SIZE, conf.BOARD_SIZE)),
                trans,
                axes=(0, 1),
            )
            p_move = np.reshape(p_move, conf.BOARD_SIZE**2)
            p = np.append(p_move, p_pass)
        else:
            feature = extract_feature(node, conf)
            features = torch.stack([torch.from_numpy(feature)]).to(device)
            log_p, v = network(features)
            p = F.softmax(log_p, dim=1)
            p = p.cpu().numpy()[0]
            v = v.cpu().numpy()[0]
    return p, v
