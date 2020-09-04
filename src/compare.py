# -*- coding: utf-8 -*-

from go import BLACK
from play import play_against_network


def estimate_win_rate(network, opponent_network, device, conf):
    score = 0
    color = BLACK
    for i in range(conf.GAMES_PER_COMPARISON):
        if play_against_network(network, opponent_network, device, color, conf):
            score += 1
        color = -color
    return score / conf.GAMES_PER_COMPARISON
