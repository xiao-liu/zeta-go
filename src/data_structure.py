# -*- coding: utf-8 -*-


# a bitmap implementation of set
# high memory efficiency
class BitSet:

    # _masks[i] = 2 ** i
    _masks = (1, 2, 4, 8, 16, 32, 64, 128)

    # _table[i] is the number of 1's in the binary representation of i
    _table = (
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
    )

    def __init__(self, capacity=0, copy=None):
        if copy is None:
            self.capacity = capacity
            self.size = 0

            # n = ceiling(capacity / 8)
            n = capacity >> 3
            if capacity & 7 != 0:
                n += 1
            self.bitmap = bytearray(n)
        else:
            self.capacity = copy.capacity
            self.size = copy.size
            self.bitmap = copy.bitmap[:]

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        for i in range(len(self.bitmap)):
            self.bitmap[i] = 0

    def contains(self, i):
        return self.bitmap[i >> 3] & BitSet._masks[i & 7] != 0

    def add(self, i):
        if not self.contains(i):
            self.bitmap[i >> 3] |= BitSet._masks[i & 7]
            self.size += 1

    def remove(self, i):
        if self.contains(i):
            self.bitmap[i >> 3] &= ~BitSet._masks[i & 7]
            self.size -= 1

    def union(self, other):
        if other is None:
            return
        assert self.capacity == other.capacity
        self.size = 0
        for i in range(len(self.bitmap)):
            self.bitmap[i] |= other.bitmap[i]
            self.size += BitSet._table[self.bitmap[i]]

    def intersection(self, other):
        if other is None:
            self.clear()
            return
        assert self.capacity == other.capacity
        self.size = 0
        for i in range(len(self.bitmap)):
            self.bitmap[i] &= other.bitmap[i]
            self.size += BitSet._table[self.bitmap[i]]

    def subtraction(self, other):
        if other is None:
            return
        assert self.capacity == other.capacity
        self.size = 0
        for i in range(len(self.bitmap)):
            self.bitmap[i] &= ~other.bitmap[i]
            self.size += BitSet._table[self.bitmap[i]]

    def arbitrary(self):
        for i in range(len(self.bitmap)):
            if self.bitmap[i] != 0:
                for j in range(8):
                    if self.bitmap[i] & BitSet._masks[j] != 0:
                        return 8 * i + j
        return None

    def all(self):
        elements = []
        for i in range(len(self.bitmap)):
            if self.bitmap[i] != 0:
                for j in range(8):
                    if self.bitmap[i] & BitSet._masks[j] != 0:
                        elements.append(8 * i + j)
        return elements


# a list implementation of set
# suitable for small cardinality set
# in ZetaGo, we often use SmallSet with at most 4 elements
class SmallSet:

    def __init__(self, capacity):
        self._capacity = capacity
        self._size = 0
        self._current = 0
        self._element = [0] * capacity

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        if i < 0 or i >= self._size:
            return None
        else:
            return self._element[i]

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self._size:
            raise StopIteration
        else:
            x = self._element[self._current]
            self._current += 1
            return x

    def clear(self):
        self._size = 0
        self._current = 0

    def contains(self, x):
        i = 0
        while i < self._size:
            if self._element[i] == x:
                return True
            i += 1
        return False

    def add(self, x):
        if self._size >= self._capacity or self.contains(x):
            return
        self._element[self._size] = x
        self._size += 1


# an implementation of circular queue
class Queue:

    def __init__(self, capacity):
        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._current = 0
        self._element = [None] * capacity

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        if i < 0 or i >= self._size:
            return None
        else:
            return self._element[(self._head + i) % self._capacity]

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self._size:
            raise StopIteration
        else:
            x = self._element[(self._head + self._current) % self._capacity]
            self._current += 1
            return x

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == self._capacity

    def enqueue(self, x):
        if self.is_full():
            return False
        else:
            self._element[self._tail] = x
            self._tail = (self._tail + 1) % self._capacity
            self._size += 1
            return True

    def dequeue(self):
        if self.is_empty():
            return None
        else:
            x = self._element[self._head]
            self._head = (self._head + 1) % self._capacity
            self._size -= 1
            return x
