# -*- coding: utf-8 -*-

from data_structure import Queue


class ResignManager:

    def __init__(self, conf):
        self._regret_frac = conf.RESIGN_REGRET_FRAC
        self._histories = Queue(conf.NUM_RESIGN_SAMPLES)
        self._threshold = -1.0

    def threshold(self):
        return -1.0 if not self._histories.is_full() else self._threshold

    def add(self, history, result):
        if len(history) == 0:
            return

        # suppose the resignation values of a game are
        # [v_0, v_1, v_2, ...], we say time t is resignation eligible
        # if it is possible to choose an appropriate threshold such
        # that the game resigns at t
        # for example, t = 0 is always resignation eligible
        # t = 1 is resignation eligible if and only if v_1 < v_0,
        # otherwise it is impossible to find a threshold such that the
        # game resigns at t = 1
        # it is not difficult to see that if t is resignation eligible,
        # the next resignation eligible time is the smallest t' such
        # that t' > t and v_t' < v_t

        # keep only the resignation eligible time and discard the rest
        history_ = [history[0]]
        for t in range(1, len(history)):
            if history[t][0] < history_[-1][0]:
                history_.append(history[t])

        # update the regret
        # regret = 1 means a false positive resignation, i.e., the game
        # could have been won if the player had not resigned
        # regret = 0 otherwise
        for i in range(len(history_)):
            # history_[i][1] originally stores the player
            # after regret is calculated, the player information is no
            # longer needed, we reuse the space to store regret
            history_[i][1] = 1 if history_[i][1] == result else 0

        # discard the earliest history and save the current history
        if self._histories.is_full():
            self._histories.dequeue()
        self._histories.enqueue(history_)

        # update the threshold when we get sufficient samples
        if self._histories.is_full():
            self._update_threshold()

    def _update_threshold(self):
        # sort all the resignation values in descending order
        resign_values = []
        for history in self._histories:
            resign_values += [x[0] for x in history]
        resign_values = sorted(resign_values, reverse=True)

        # search for the threshold
        pointers = [0] * len(self._histories)
        i = 0
        while i < len(resign_values):
            threshold = resign_values[i]

            # calculate the number of regretful resignations for this
            # candidate threshold
            regrets = 0
            for j in range(len(self._histories)):
                history = self._histories[j]
                while pointers[j] < len(history) and \
                        history[pointers[j]][0] > threshold:
                    pointers[j] += 1
                if pointers[j] < len(history):
                    regrets += history[pointers[j]][1]

            if regrets <= self._regret_frac * len(self._histories):
                self._threshold = threshold
                return

            # find the next candidate threshold
            j = i + 1
            while j < len(resign_values) and \
                    resign_values[j] == resign_values[i]:
                j += 1
            i = j

        self._threshold = -1.0
