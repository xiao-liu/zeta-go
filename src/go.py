# -*- coding: utf-8 -*-

from data_structure import BitSet, Queue, SmallSet

# color of intersection
BLACK = 1  # black stone
WHITE = -1  # white stone
EMPTY = 0  # empty intersection
_GRAY = 2  # internal use only, to mark the intermediate state in BFS

# the type of ownership for empty intersection chain
#   - an empty intersection chain is owned by black (white) if it is
#     adjacent to black (white) stones only
#   - an empty intersection chain's ownership is undecided if it is
#     adjacent to both black and white stones or it is not adjacent to
#     any stones (when the game starts)

# the empty intersection chain is adjacent to a single black stone
# chain only, therefore forms an eye
BLACK_EYE = 1

# the empty intersection chain is adjacent to at least two black stone
# chains
BLACK_OWNED = 2

# the empty intersection chain is adjacent to a single white stone
# chain only, therefore forms an eye
WHITE_EYE = -1

# the empty intersection chain is adjacent to at least two white stone
# chains
WHITE_OWNED = -2

# the ownership of the empty intersection is undecided
UNDECIDED = 0

# the four directions: left, up, right, down
_directions = ((1, 0), (0, 1), (-1, 0), (0, -1))


# a class of the Go board, maintaining the information of stone chains
# and empty intersection chains
class Board:

    def __init__(self, board_size=19, copy=None):
        if copy is None:
            # size of board
            self.board_size = board_size

            # number of intersections on board
            n = board_size * board_size

            # _color[x * board_size + y] stores the color of
            # intersection (x, y)
            # should be one of BLACK, WHITE, EMPTY
            self._color = [EMPTY] * n

            # adopt the idea of disjoint set to maintain the stone
            # chains and empty intersection chains
            self._parent = [0] * n

            # if (x, y) is the representative of the (stone or empty
            # intersection) chain it belongs to,
            # _chain_size[x * board_size + y] stores the number of
            # stones or empty intersections in the chain, otherwise the
            # value of _chain_size[x * board_size + y] is meaningless
            self._chain_size = [0] * n
            self._chain_size[0] = n

            # NOTICE: Unlike the stone chains, maintaining _parent and
            # _chain_size for empty intersection chains has relatively
            # high performance overhead. In the meanwhile, we do not
            # need empty intersection chains' _parent and _chain_size
            # either unless we are to calculate scores or eye
            # information. Therefore, we will not update empty
            # intersection chains' _parent and _chain_size in most of
            # the time. We will update them only before scoring by
            # calling update_empty_intersection().

            # if a stone is on (x, y) and it is the representative of
            # the stone chain it belongs to,
            # _liberties[x * board_size + y] stores all the liberties
            # of that chain, otherwise (i.e., if (x, y) is an empty
            # intersection or the stone on (x, y) is not the
            # representative), _liberties[x * board_size + y] = None
            self._liberties = [None] * n
        else:
            # create a deep copy
            self.board_size = copy.board_size
            self._color = copy._color[:]
            self._parent = copy._parent[:]
            self._chain_size = copy._chain_size[:]
            self._liberties = [
                None if x is None else BitSet(copy=x)
                for x in copy._liberties
            ]

    # return the color of (x, y)
    def color(self, x, y):
        return self._color[x * self.board_size + y]

    # return the size of the chain (x, y) belongs to
    def chain_size(self, x, y):
        return self._chain_size[self.find(x, y)]

    # return the size of the chain z belongs to, assuming z is an
    # representative
    def chain_size_(self, z):
        return self._chain_size[z]

    # return the liberties of the chain (x, y) belongs to
    def liberties(self, x, y):
        return self._liberties[self.find(x, y)]

    # decide if (x, y) is a legal coordinate on the board
    def on_board(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    # find the representative of the chain (x, y) belongs to
    def find(self, x, y):
        z = x * self.board_size + y
        while self._parent[z] != z:
            w = self._parent[z]
            self._parent[z] = self._parent[self._parent[z]]
            z = w
        return z

    # merge the chain (x1, y1) belongs to with the chain (x2, y2)
    # belongs to
    # assuming stones are on both (x1, y1) and (x2, y2), they are of
    # the same color and they are indeed connected
    def _union(self, x1, y1, x2, y2):
        z1 = self.find(x1, y1)
        z2 = self.find(x2, y2)
        if z1 == z2:
            return
        elif self._chain_size[z1] < self._chain_size[z2]:
            self._liberties[z2].union(self._liberties[z1])
            self._liberties[z1] = None
            self._parent[z1] = z2
            self._chain_size[z2] += self._chain_size[z1]
        else:
            self._liberties[z1].union(self._liberties[z2])
            self._liberties[z2] = None
            self._parent[z2] = z1
            self._chain_size[z1] += self._chain_size[z2]

    # remove the stone chain (x, y) belongs to
    # assuming a stone is on (x, y)
    def _remove(self, x, y):
        self._liberties[self.find(x, y)] = None

        n = self.board_size * self.board_size
        z = x * self.board_size + y
        color = self._color[z]

        # insert (x, y) into the queue, and mark it as "visited but not
        # processed" (color = _GRAY)
        # we abuse the _color array to store the BFS state here:
        #   - color = EMPTY: processed
        #   - color = BLACK: not visited (for black stones)
        #   - color = WHITE: not visited (for white stones)
        #   - color = _GRAY: visited (i.e., in queue) but not processed
        q = Queue(n)
        q.enqueue((x, y))
        self._color[z] = _GRAY
        adjacent_opponent_chains = SmallSet(4)
        while not q.is_empty():
            x, y = q.dequeue()
            adjacent_opponent_chains.clear()
            for dx, dy in _directions:
                if self.on_board(x + dx, y + dy):
                    # insert all the own adjacent stones that are not
                    # visited into the queue
                    if self.color(x + dx, y + dy) == color:
                        q.enqueue((x + dx, y + dy))
                        self._color[(x + dx) * self.board_size + y + dy] = _GRAY
                    # save opponent's stone chains that are adjacent to
                    # this new empty intersection
                    if self.color(x + dx, y + dy) == -color:
                        adjacent_opponent_chains.add(self.find(x + dx, y + dy))
            # the new empty intersection (x, y) provides liberty to its
            # adjacent stone chains
            z = x * self.board_size + y
            for c in adjacent_opponent_chains:
                if self._liberties[c] is None:
                    self._liberties[c] = BitSet(n)
                self._liberties[c].add(z)
            self._color[z] = EMPTY

    # update _parent and _chain_size for all the empty intersection
    # chains, and return an array ownership which contains the type of
    # ownership for each empty intersection chain:
    #   - if (x, y) is the representative of the empty intersection
    #     chain (x, y) belongs to, ownership[x * board_size + y] is the
    #     type of ownership for that chain, and its value must be one
    #     of BLACK_EYE, BLACK_OWNED, WHITE_EYE, WHITE_OWNED, UNDECIDED
    #   - otherwise (i.e., if a stone is on (x, y) or the empty
    #     intersection (x, y) is not the representative), the value of
    #     ownership[x * board_size + y] is meaningless
    def update_empty_intersection(self):
        n = self.board_size * self.board_size
        ownership = [UNDECIDED] * n
        visited = BitSet(n)
        adjacent_chains = BitSet(n)
        for x in range(self.board_size):
            for y in range(self.board_size):
                z = x * self.board_size + y
                if self._color[z] != EMPTY or visited.contains(z):
                    continue

                self._chain_size[z] = 0
                adjacent_to_black, adjacent_to_white = False, False
                adjacent_chains.clear()

                q = Queue(n)
                q.enqueue((x, y))
                visited.add(z)
                while not q.is_empty():
                    x, y = q.dequeue()
                    for dx, dy in _directions:
                        if self.on_board(x + dx, y + dy):
                            if self.color(x + dx, y + dy) == BLACK:
                                adjacent_chains.add(self.find(x + dx, y + dy))
                                adjacent_to_black = True
                            elif self.color(x + dx, y + dy) == WHITE:
                                adjacent_chains.add(self.find(x + dx, y + dy))
                                adjacent_to_white = True
                            elif not visited.contains(
                                    (x + dx) * self.board_size + y + dy):
                                q.enqueue((x + dx, y + dy))
                                visited.add((x + dx) * self.board_size + y + dy)
                    self._parent[x * self.board_size + y] = z
                    self._chain_size[z] += 1

                if (adjacent_to_black and adjacent_to_white) \
                        or len(adjacent_chains) == 0:
                    ownership[z] = UNDECIDED
                elif len(adjacent_chains) == 1:
                    if adjacent_to_black:
                        ownership[z] = BLACK_EYE
                    else:
                        ownership[z] = WHITE_EYE
                else:
                    if adjacent_to_black:
                        ownership[z] = BLACK_OWNED
                    else:
                        ownership[z] = WHITE_OWNED

        return ownership

    # place a stone on (x, y) with specified color
    # assuming (x, y) is an empty intersection
    def place(self, x, y, color):
        # create a stone chain of size 1 on (x, y)
        z = x * self.board_size + y
        self._color[z] = color
        self._parent[z] = z
        self._chain_size[z] = 1
        if self._liberties[z] is None:
            self._liberties[z] = BitSet(self.board_size * self.board_size)
        for dx, dy in _directions:
            if self.on_board(x + dx, y + dy) \
                    and self.color(x + dx, y + dy) == EMPTY:
                self._liberties[z].add((x + dx) * self.board_size + y + dy)

        # update the liberty of the chains adjacent to (x, y)
        adjacent_chains = SmallSet(4)
        for dx, dy in _directions:
            if self.on_board(x + dx, y + dy) \
                    and self.color(x + dx, y + dy) != EMPTY:
                adjacent_chains.add(self.find(x + dx, y + dy))
        for c in adjacent_chains:
            self._liberties[c].remove(z)

        # merge the stone on (x, y) with all the own stone chains that
        # are adjacent to (x, y)
        for dx, dy in _directions:
            if self.on_board(x + dx, y + dy) \
                    and self.color(x + dx, y + dy) == color:
                self._union(x, y, x + dx, y + dy)

        # remove all the opponent's stone chains that are captured
        for dx, dy in _directions:
            if self.on_board(x + dx, y + dy) \
                    and self.color(x + dx, y + dy) == -color \
                    and len(self.liberties(x + dx, y + dy)) == 0:
                self._remove(x + dx, y + dy)

    # calculate the scores for black and white
    def score(self, komi):
        ownership = self.update_empty_intersection()

        black_score, white_score = 0.0, komi
        for z in range(len(self._parent)):
            if self._parent[z] != z:
                continue
            if self._color[z] == BLACK:
                black_score += self._chain_size[z]
            elif self._color[z] == WHITE:
                white_score += self._chain_size[z]
            elif ownership[z] == BLACK_EYE or ownership[z] == BLACK_OWNED:
                black_score += self._chain_size[z]
            elif ownership[z] == WHITE_EYE or ownership[z] == WHITE_OWNED:
                white_score += self._chain_size[z]
        return black_score, white_score


# a class that maintains a game of Go
class Go:

    def __init__(self, board_size=19, komi=0, handicap=None, copy=None):
        if copy is None:
            # an instance of Board class
            self.board = Board(board_size=board_size)

            # komi
            self.komi = komi

            # turn is the player to make the next move
            self.turn = BLACK

            # necessary information about the last move to determine ko
            # notice that we only implement simple ko here and superko
            # is not supported
            # (previous_x, previous_y) is coordinate where the stone is
            # placed in the last move
            # previous_captured_size is the number of stones captured
            # in the last move
            self.previous_x = -1
            self.previous_y = -1
            self.previous_captured_size = 0

            if handicap is not None and len(handicap) > 0:
                # place handicap stones
                for x, y in handicap:
                    self.board.place(x, y, BLACK)
                # white first
                self.turn = WHITE
        else:
            self.board = Board(copy=copy.board)
            self.komi = copy.komi
            self.turn = copy.turn
            self.previous_x = copy.previous_x
            self.previous_y = copy.previous_y
            self.previous_captured_size = copy.previous_captured_size

    # determine opponent's stone chains that will be captured if
    # placing a stone on (x, y)
    # assuming (x, y) is an empty intersection on the board, and
    # ignoring the ko rule
    def captured_chains(self, x, y):
        captured_chains = SmallSet(4)
        b = self.board
        for dx, dy in _directions:
            if b.on_board(x + dx, y + dy) \
                    and b.color(x + dx, y + dy) == -self.turn \
                    and len(b.liberties(x + dx, y + dy)) == 1:
                captured_chains.add(b.find(x + dx, y + dy))
        return captured_chains

    # determine if placing a stone on (x, y) causes suicide
    # assuming (x, y) is an empty intersection on the board
    def suicide(self, x, y):
        b = self.board
        for dx, dy in _directions:
            if b.on_board(x + dx, y + dy) \
                    and (b.color(x + dx, y + dy) == EMPTY
                         or (b.color(x + dx, y + dy) == self.turn
                             and len(b.liberties(x + dx, y + dy)) > 1)):
                return False
        return True

    # determine if placing a stone on (x, y) is a legal play
    def legal_play(self, x, y, ignore_ko=False):
        b = self.board
        # (x, y) must be on the board
        if not b.on_board(x, y):
            return False
        # (x, y) must be an empty intersection
        if b.color(x, y) != EMPTY:
            return False
        # determine if (x, y) is in ko
        captured_chains = self.captured_chains(x, y)
        if not ignore_ko and len(captured_chains) == 1 \
                and b.chain_size_(captured_chains[0]) == 1 \
                and self.previous_captured_size == 1 \
                and b.find(self.previous_x, self.previous_y) == \
                captured_chains[0]:
            return False
        # the play is legal as long as we can capture opponent's
        # stones, or it does not cause suicide
        return len(captured_chains) != 0 or not self.suicide(x, y)

    # try to place a stone on (x, y)
    # return true if (x, y) is a legal play, return false otherwise
    def play(self, x, y):
        if not self.legal_play(x, y):
            return False

        # update previous_captured_size
        self.previous_captured_size = 0
        captured_chains = self.captured_chains(x, y)
        for c in captured_chains:
            self.previous_captured_size += self.board.chain_size_(c)

        # place a stone of own side on (x, y)
        self.board.place(x, y, self.turn)

        # update the other states
        self.previous_x = x
        self.previous_y = y
        self.turn = -self.turn

        return True

    # perform a pass move
    def pass_(self):
        self.previous_x = -1
        self.previous_y = -1
        self.previous_captured_size = 0
        self.turn = -self.turn

    # calculate the scores for black and white
    def score(self, komi=None):
        if komi is None:
            komi = self.komi
        return self.board.score(komi)

    # NOTICE: The functions below are about ladder capture and they
    # are quite compute-intensive. Ladder capture information is
    # important feature in AlphaGo but it is no longer used in
    # AlphaGo Zero. I list them here for completeness.

    # determine if placing a stone on (x, y) can ladder capture the
    # stone chain (target_x, target_y) belongs to
    # here we assume:
    #   - (x, y) is an empty intersection
    #   - an opponent's stone is on (target_x, target_y)
    #   - (x, y) provides liberty to the stone chain
    #     (target_x, target_y) belongs to
    def ladder_capture_(self, x, y, target_x, target_y):
        if not self.legal_play(x, y):
            return False

        if len(self.board.liberties(target_x, target_y)) == 1:
            # if the stone chain (target_x, target_y) belongs to has
            # only one liberty, then it must be (x, y)
            # notice that (x, y) is a legal play, hence the ladder
            # capture is successful
            return True
        elif len(self.board.liberties(target_x, target_y)) == 2:
            # if the stone chain (target_x, target_y) belongs to has
            # two liberties, then after we place a stone on (x, y), it
            # will have only one liberty
            # if the stone chain can escape the ladder capture by
            # placing a stone on its only liberty, the ladder capture
            # is failed, otherwise the ladder capture is successful
            go_ = Go(copy=self)
            go_.play(x, y)
            z = go_.board.liberties(target_x, target_y).arbitrary()
            x_, y_ = divmod(z, self.board.board_size)
            return not go_.ladder_escape_(x_, y_, target_x, target_y)
        else:
            # if the stone chain (target_x, target_y) belongs to has
            # more than two liberties, then cannot be ladder captured
            return False

    # determine if placing a stone (x, y) can let the stone chain
    # (target_x, target_y) belongs to escape from ladder capture
    # here we assume:
    #   - (x, y) is an empty intersection
    #   - an own stone is on (target_x, target_y)
    #   - (x, y) provides liberty to the stone chain
    #     (target_x, target_y) belongs to
    def ladder_escape_(self, x, y, target_x, target_y):
        if not self.legal_play(x, y):
            return False

        # only the stone chain in atari needs to consider the problem
        # of escaping from ladder capture
        if len(self.board.liberties(target_x, target_y)) != 1:
            return False

        # place a stone on (x, y)
        go_ = Go(copy=self)
        go_.play(x, y)

        if len(go_.board.liberties(target_x, target_y)) == 1:
            # if the stone chain (target_x, target_y) belongs to has
            # only one liberty, escaping is failed
            return False
        elif len(go_.board.liberties(target_x, target_y)) == 2:
            # if the stone chain (target_x, target_y) belongs to has
            # two liberties, then the escaping is failed if opponent
            # can ladder capture it by placing a stone on any of these
            # two liberties, otherwise the escaping is successful
            for z in go_.board.liberties(target_x, target_y).all():
                x_, y_ = divmod(z, self.board.board_size)
                if go_.ladder_capture_(x_, y_, target_x, target_y):
                    return False
            return True
        else:
            # if the stone chain (target_x, target_y) belongs to has
            # more than two liberties, then always be able to escape
            return True

    # determine if placing a stone on (x, y) can ladder capture any of
    # the opponent's stone chains that is adjacent to (x, y)
    # assuming (x, y) is an empty intersection
    def ladder_capture(self, x, y):
        adjacent_chains = SmallSet(4)
        b = self.board
        for dx, dy in _directions:
            if b.on_board(x + dx, y + dy) \
                    and b.color(x + dx, y + dy) == -self.turn \
                    and not adjacent_chains.contains(b.find(x + dx, y + dy)):
                if self.ladder_capture_(x, y, x + dx, y + dy):
                    return True
                adjacent_chains.add(b.find(x + dx, y + dy))
        return False

    # determine if placing a stone on (x, y) can let any of own stone
    # chains that is adjacent to (x, y) escape from ladder capture
    # assuming (x, y) is an empty intersection
    def ladder_escape(self, x, y):
        adjacent_chains = SmallSet(4)
        b = self.board
        for dx, dy in _directions:
            if b.on_board(x + dx, y + dy) \
                    and b.color(x + dx, y + dy) == self.turn \
                    and not adjacent_chains.contains(b.find(x + dx, y + dy)):
                if self.ladder_escape_(x, y, x + dx, y + dy):
                    return True
                adjacent_chains.add(b.find(x + dx, y + dy))
        return False
