### LRU Cache (Least Recently Used)

An LRU Cache keeps the most recently accessed items quickly available and evicts the least recently used item when capacity is reached.

#### Operations and Complexity
- **get(key)**: Return value if present, otherwise -1. Marks key as most recently used. O(1)
- **put(key, value)**: Insert/update key. If capacity exceeded, evict least recently used key. O(1)

#### Approaches
- **Using OrderedDict (Python)**: Simplest practical solution leveraging `move_to_end` and `popitem(last=False)`.
- **Manual Doubly-Linked List + HashMap**: Classic interview approach to ensure O(1) operations without language-specific helpers.

#### Files
- `Python/lru_cache.py` implements both variants:
  - `LRUCache`: OrderedDict-backed, concise and production-friendly.
  - `LRUCacheManual`: Educational implementation using explicit linked list + dict.

Run the simple self-tests by executing:

```bash
python Python/lru_cache.py
```

Expected output:

```text
All LRU cache tests passed.
```

#### Common Pitfalls
- Forgetting to mark items as most-recent on `get`.
- Evicting the wrong end (ensure LRU is at the tail/left depending on structure).
- Handling capacity <= 0 (disallow or define semantics).

This problem appears frequently in interviews due to the combination of data structures and operational constraints.


