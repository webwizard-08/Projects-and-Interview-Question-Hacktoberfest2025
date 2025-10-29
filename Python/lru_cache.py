from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Iterator, Optional, Tuple, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    A simple O(1) LRU cache using OrderedDict.

    - get(key): returns value or -1 if missing; marks key as recently used
    - put(key, value): inserts/updates; evicts least-recently used item when capacity is full

    Example:
        cache = LRUCache(capacity=2)
        cache.put(1, "A")
        cache.put(2, "B")
        assert cache.get(1) == "A"  # 1 becomes most recently used
        cache.put(3, "C")            # evicts key 2 (least recently used)
        assert cache.get(2) == -1
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity: int = capacity
        self._store: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K):
        if key not in self._store:
            return -1
        self._store.move_to_end(key, last=True)
        return self._store[key]

    def put(self, key: K, value: V) -> None:
        if key in self._store:
            self._store[key] = value
            self._store.move_to_end(key, last=True)
            return
        if len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def items_mru_first(self) -> Iterator[Tuple[K, V]]:
        # Expose items in MRU -> LRU order for debugging/demo
        for k in reversed(self._store.keys()):
            yield k, self._store[k]


# A manual LRU (doubly-linked list + dict) for educational purposes
@dataclass
class _Node(Generic[K, V]):
    key: Optional[K]
    value: Optional[V]
    prev: Optional["_Node[K, V]"] = None
    next: Optional["_Node[K, V]"] = None


class LRUCacheManual(Generic[K, V]):
    """
    LRU with explicit doubly-linked list + dict achieving O(1) get/put.
    This mirrors classic interview solutions without relying on OrderedDict.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.key_to_node: Dict[K, _Node[K, V]] = {}
        # Sentinel nodes: head <-> ... <-> tail
        self.head: _Node[K, V] = _Node(None, None)
        self.tail: _Node[K, V] = _Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: _Node[K, V]) -> None:
        node.prev = self.head
        node.next = self.head.next
        assert node.next is not None
        self.head.next.prev = node  # type: ignore[union-attr]
        self.head.next = node

    def _remove(self, node: _Node[K, V]) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_front(self, node: _Node[K, V]) -> None:
        self._remove(node)
        self._add_to_front(node)

    def _evict_if_needed(self) -> None:
        if len(self.key_to_node) <= self.capacity:
            return
        # remove from tail (LRU is tail.prev)
        lru = self.tail.prev
        assert lru is not None and lru is not self.head
        self._remove(lru)
        assert lru.key is not None
        del self.key_to_node[lru.key]

    def get(self, key: K):
        node = self.key_to_node.get(key)
        if node is None:
            return -1
        self._move_to_front(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        node = self.key_to_node.get(key)
        if node is not None:
            node.value = value
            self._move_to_front(node)
            return
        new_node: _Node[K, V] = _Node(key, value)
        self.key_to_node[key] = new_node
        self._add_to_front(new_node)
        if len(self.key_to_node) > self.capacity:
            self._evict_if_needed()


def _run_basic_tests() -> None:
    # OrderedDict-backed
    c = LRUCache[int, str](2)
    c.put(1, "A")
    c.put(2, "B")
    assert c.get(1) == "A"  # MRU: 1
    c.put(3, "C")           # evict 2
    assert c.get(2) == -1
    assert c.get(3) == "C"
    c.put(4, "D")           # evict 1
    assert c.get(1) == -1
    assert c.get(3) == "C"
    assert c.get(4) == "D"

    # Manual implementation
    m = LRUCacheManual[int, str](2)
    m.put(1, "A")
    m.put(2, "B")
    assert m.get(1) == "A"
    m.put(3, "C")           # evict 2
    assert m.get(2) == -1
    assert m.get(3) == "C"
    m.put(4, "D")           # evict 1
    assert m.get(1) == -1
    assert m.get(3) == "C"
    assert m.get(4) == "D"


if __name__ == "__main__":
    _run_basic_tests()
    print("All LRU cache tests passed.")


