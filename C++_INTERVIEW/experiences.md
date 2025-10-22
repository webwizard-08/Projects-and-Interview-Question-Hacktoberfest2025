# How to Submit a New Interview Experience

Use this template when adding your interview story:

- **Company**: [Company name or "(undisclosed)"]
- **Role / Level**: [e.g. "Backend Engineer (mid)", "Systems Engineer (senior)", "Quant Dev", etc.]
- **When**: [Year or approximate date]
- **Interview Format**: [e.g. "2 coding rounds + 1 design + HR", "on-site with whiteboard", "pair programming", "take-home"]
- **Questions asked (2-4 sample)**:
  1. [Question A]
  2. [Question B]
  3. [optional C]
  4. [optional D]
- **Hints / Example answers / takeaways**: [short notes, tip, or how you answered]

---

# 20 Real C++ Interview Experiences (2024–2025)

> These are anonymized reports collected from engineers. Use them as reference for patterns, difficulty, question types, and answer styles.

---

### 1. Company: Google

**Role / Level**: Software Engineer (mid)  
**When**: 2025  
**Interview Format**: 3 rounds (coding + system design + behavioral)  
**Questions asked**:

1. Given a grid of 0/1, count number of distinct islands (shapes) — shape equality under translation.
2. Implement LRU cache with O(1) get/put.
3. Explain how C++ virtual dispatch works under the hood (vtable, pointers).
4. Given a function signature, convert recursion into tail recursion or iterative.  
   **Hints / Takeaways**:

- For distinct islands, canonical representation or hash of shape matters.
- Use `unordered_map + list` for LRU.
- Be precise about `virtual` in constructor/destructor context.
- Always clarify memory bounds, edges, performance.

---

### 2. Company: Microsoft

**Role / Level**: Senior Engineer  
**When**: 2024  
**Interview Format**: Onsite full day, 4 rounds  
**Questions asked**:

1. Design a thread-safe bounded queue (producer-consumer) in C++.
2. Given a histogram, find largest rectangle area.
3. Difference between `shared_ptr`, `unique_ptr`, `weak_ptr`.
4. Explain copy elision, move semantics, and RVO in modern C++.  
   **Hints / Takeaways**:

- Use `mutex` + `condition_variable` or lock-free ring buffer.
- Histogram is classic stack problem.
- Know edge cases of weak_ptr to avoid cycles.
- Understand when move is auto-used, and what `noexcept` means.

---

### 3. Company: Amazon

**Role / Level**: Backend / Core Developer  
**When**: 2025  
**Interview Format**: 2 coding rounds via virtual IDE + 1 system design  
**Questions asked**:

1. Two-sum variant but with multiple constraints (e.g. sum in range).
2. Merge k sorted lists (or streams).
3. Explain “Rule of Three / Five / Zero” in C++.  
   **Hints / Takeaways**:

- Start by brute, optimize using hash or two-pointer.
- Use min-heap for merging.
- Rule of Five: if you implement destructor / copy / move / assignment, do all appropriately (see CoderPad examples) :contentReference[oaicite:0]{index=0}

---

### 4. Company: Apple

**Role / Level**: Systems Engineer  
**When**: 2025  
**Interview Format**: 3 rounds (coding + system + low-level)  
**Questions asked**:

1. Given an IP packet buffer, parse header + options + payload.
2. Write a memory allocator (like `malloc`) for small allocations.
3. Explain aliasing rules, strict aliasing, `volatile`, and UB.
4. How `constexpr` and `consteval` differ and use cases.  
   **Hints / Takeaways**:

- Structure packing, alignment are key.
- For allocator, free list, splitting/coalescing.
- Violating aliasing can break optimizations.
- `consteval` enforces compile-time; `constexpr` can be runtime or compile-time.

---

### 5. Company: Meta (Facebook)

**Role / Level**: Performance Engineer  
**When**: 2024  
**Interview Format**: 4 rounds  
**Questions asked**:

1. Optimize a slow C++ loop (vector of structs) — memory layout / caching.
2. Given function calls stack trace, detect recursion or cycles.
3. Difference between polymorphism via inheritance vs templates.
4. Exception safety: strong / basic / no-throw guarantees.  
   **Hints / Takeaways**:

- Use structure-of-arrays (SoA) rather than array-of-structs for cache.
- Use tortoise-hare or detect cycles in function calls.
- Template (static) polymorphism vs runtime virtual.
- Use RAII, `noexcept`, careful in destructors.

---

### 6. Company: Bloomberg

**Role / Level**: Quant Developer  
**When**: 2025  
**Interview Format**: Coding + math + domain  
**Questions asked**:

1. Given prices, compute max profit with up to K transactions (DP).
2. Implement a sliding window median structure.
3. Use templates to write a generic solve for vector and list.  
   **Hints / Takeaways**:

- DP with O(nK) or optimize to O(n log K).
- Use two multisets or two heaps for median.
- Template specialization, SFINAE or concepts.

---

### 7. Company: NVIDIA

**Role / Level**: Graphics / GPU Engineer  
**When**: 2025  
**Interview Format**: Coding + domain + systems  
**Questions asked**:

1. Implement matrix multiply optimizing memory tiling.
2. Given pixel shader code, detect potential overflow or float precision issues.
3. Explain alignment, padding, vectorization, and `restrict`.  
   **Hints / Takeaways**:

- Cache tile sizes matter.
- Use fma, avoid branching.
- Understand how alignment directives help vector instructions.

---

### 8. Company: Uber

**Role / Level**: Backend / Infrastructure  
**When**: 2024  
**Interview Format**: Virtual coding + system design  
**Questions asked**:

1. Build rate limiter (token bucket / leaky bucket) in C++.
2. Given log streams, detect anomalies in sliding windows.
3. Differences between `new[]` and `malloc`, and pitfalls mixing them.  
   **Hints / Takeaways**:

- Token bucket implementation, thread safety, expiration.
- Use deque + window sum.
- `new` calls constructors; `delete` must match `new[]`.

---

### 9. Company: Intel

**Role / Level**: Low-level / Embedded Engineer  
**When**: 2025  
**Interview Format**: Onsite hardware + software  
**Questions asked**:

1. Write an interrupt handler stub in C++ (bare metal) without stdlib.
2. Implement circular buffer with lock-free concurrency.
3. Explain memory barriers, atomic ordering, `volatile` inefficacy.  
   **Hints / Takeaways**:

- Use ring buffer + atomic indexes.
- Use `std::atomic` with memory_order.
- Volatile doesn’t imply atomicity.

---

### 10. Company: Goldman Sachs

**Role / Level**: Quant / Backend  
**When**: 2024  
**Interview Format**: Coding + math + design  
**Questions asked**:

1. Given time series, compute rolling variance efficiently.
2. Implement custom hash map (open addressing) for speed.
3. Explain move semantics to avoid copies in numeric libraries.  
   **Hints / Takeaways**:

- Use Welford’s algorithm for variance.
- Probing strategies, rehashing.
- Perfect forwarding in templates.

---

### 11. Company: Stripe

**Role / Level**: Core Systems  
**When**: 2025  
**Interview Format**: 3 rounds  
**Questions asked**:

1. Build a memory pool for fixed-size allocations.
2. Given a parse tree, evaluate expression with operator precedence.
3. How does C++ RTTI work? `typeid`, `dynamic_cast`.  
   **Hints / Takeaways**:

- Use free-list, chunk pool.
- Use recursive descent or shunting yard.
- RTTI adds overhead; dynamic_cast safe only for polymorphic classes.

---

### 12. Company: Dropbox

**Role / Level**: Storage / Backend Engineer  
**When**: 2024  
**Interview Format**: Coding + design  
**Questions asked**:

1. Design a versioning file system snapshot.
2. Given file change events, compress diffs (delta).
3. Explain copy-on-write, memory mapping, and persistence.  
   **Hints / Takeaways**:

- Use persistent trees (e.g. copy-on-write B-tree).
- Binary diff (XOR/delta) logic.
- COW helps share pages.

---

### 13. Company: Oracle

**Role / Level**: Senior C++ Engineer  
**When**: 2025  
**Interview Format**: Coding + architecture  
**Questions asked**:

1. Design a plugin system (load/unload .so) with safe API.
2. Given ambiguous overloads, explain resolution rules.
3. Explain template instantiation, inline functions, ODR (One Definition Rule).  
   **Hints / Takeaways**:

- Use function pointers, registration, versioning.
- Know precedence of conversion, const, template deduction.
- Violating ODR causes undefined behavior.

---

### 14. Company: SAP

**Role / Level**: Backend Dev  
**When**: 2024  
**Interview Format**: 2 virtual + 1 on-prem  
**Questions asked**:

1. Given tree with weights, find max path sum.
2. Debug a small snippet having undefined behavior (dangling pointer).
3. Explain differences among `std::move`, `std::forward`.  
   **Hints / Takeaways**:

- DFS with recursion or DP.
- Dangling pointers appear when returning reference to local.
- `forward` preserves value category; `move` forces rvalue.

---

### 15. Company: Palantir

**Role / Level**: Software Engineer  
**When**: 2025  
**Interview Format**: Coding + system rounds  
**Questions asked**:

1. Given logs of events, detect sequence patterns (e.g. 3 events in order).
2. Implement a prefix tree (trie) with memory constraints.
3. Explain copy-on-write string implementations, COW vs `std::string_view`.  
   **Hints / Takeaways**:

- Sliding window + automaton.
- Use pointers, memory pooling.
- COW strings disallowed in C++17; use view + owning buffer.

---

### 16. Company: Zoom

**Role / Level**: Real-time Systems  
**When**: 2024  
**Interview Format**: 3 rounds  
**Questions asked**:

1. Build jitter buffer and packet reordering buffer.
2. Implement rate control or congestion control in C++.
3. Explain `asio` / `boost::asio` design and asynchronous model.  
   **Hints / Takeaways**:

- Circular buffer + timing logic.
- Use token buckets, EWMA.
- Understand callback, io_context, futures.

---

### 17. Company: Cisco

**Role / Level**: Networking / Systems  
**When**: 2025  
**Interview Format**: Onsite hardware + code  
**Questions asked**:

1. Implement shortest path (Dijkstra / Bellman-Ford) with constraints.
2. Given packet header fields, parse with bit-level operations.
3. Explain endianness, alignment, bit-fields.  
   **Hints / Takeaways**:

- Use priority queue + adjacency list.
- Use bitmask, shifts.
- Byte ordering matters; watch padding.

---

### 18. Company: Shopify

**Role / Level**: Core Backend Engineer  
**When**: 2025  
**Interview Format**: Virtual + onsite  
**Questions asked**:

1. Implement merge sort / quick sort in-place and stable variant.
2. Given many small requests, design sharding / caching.
3. Explain template metaprogramming, SFINAE, `enable_if`.  
   **Hints / Takeaways**:

- Watch recursion depth, choose pivot wisely.
- Use consistent hashing, LRU caches.
- Use `std::enable_if`, `std::void_t`, concepts (C++20).

---

### 19. Company: Adobe

**Role / Level**: Graphics Engine  
**When**: 2024  
**Interview Format**: Coding + design  
**Questions asked**:

1. Implement Bezier curve evaluation (cubic, de Casteljau).
2. Given transformations (scale, rotate, translate), compose and apply.
3. Explain operator overloading ((), [], +, <<), pitfalls.  
   **Hints / Takeaways**:

- Use barycentric or de Casteljau algorithm.
- Use homogeneous coordinates (4×4).
- Overload with correct const, reference, templates.

---

### 20. Company: Spotify

**Role / Level**: Backend / Real-time  
**When**: 2025  
**Interview Format**: Coding + architecture  
**Questions asked**:

1. Design event stream aggregator (windowed counts).
2. Given a heavy data stream, approximate top-K frequent elements.
3. Explain memory leak detection, `valgrind`, sanitizers.  
   **Hints / Takeaways**:

- Use sliding windows + maps or count-min-sketch.
- Use Misra-Gries or SpaceSaving algorithm.
- Use `ASAN`, `LSAN`, leak detection tools.

---

### 21. Company: Jane Street

**Role / Level**: C++ Developer (Low Latency)
**When**: 2025
**Interview Format**: Phone screen + multi-round onsite with pair programming and systems questions.
**Questions asked**:

1. Design a high-throughput, low-latency order matching engine for a stock exchange.
2. Given a hot loop processing market data, how would you minimize instruction cache misses and pipeline stalls?
3. Implement a data structure that can efficiently find the k-th smallest element in a stream of numbers.
   **Hints / Takeaways**:

- For the matching engine, focus on the data structures for the order book (e.g., `std::map` or custom balanced BSTs) and the trade execution logic.
- For the hot loop, discuss data layout (AoS vs. SoA), loop unrolling, SIMD intrinsics, and prefetching.
- For k-th smallest, an order-statistic tree or randomized quickselect on windows are good approaches. The focus is on extreme performance and mechanical sympathy.

---

### 22. Company: Epic Games

**Role / Level**: Game Engine Programmer (mid-level)
**When**: 2024
**Interview Format**: Take-home project + technical deep dive on the project + coding rounds.
**Questions asked**:

1. Write a C++ class for a 3D vector (`Vec3`) and implement dot product, cross product, and normalization.
2. Design a simple entity-component system (ECS). Discuss the data layout and how it benefits cache performance.
3. Explain object lifetimes in Unreal Engine (`UObject`, garbage collection, `TSharedPtr`).
   **Hints / Takeaways**:

- For the `Vec3` class, they look for clean code, operator overloading (`+`, `-`, `*`), and an understanding of the underlying math.
- For ECS, the key is explaining data-oriented design vs. object-oriented design and its performance implications.
- Domain-specific knowledge of a major engine's memory model is a huge plus. Know when to use different pointer types.

---

### 23. Company: Cloudflare

**Role / Level**: Systems Engineer
**When**: 2025
**Interview Format**: Coding challenge + systems design + deep C++ concepts.
**Questions asked**:

1. Parse a raw TCP packet from a byte buffer and extract the sequence number and payload without using a library.
2. Implement a performant read-write lock in C++.
3. Explain what `std::string_view` is, how it differs from `const std::string&`, and why it's crucial for high-performance parsing.
   **Hints / Takeaways**:

- Packet parsing requires careful attention to byte order (endianness), struct packing, and pointer arithmetic/casting.
- For the read-write lock, you can use `std::shared_mutex` (C++17) or build one from `std::mutex` and `std::condition_variable`. Discuss starvation.
- `string_view` is about avoiding allocations and copies by providing a non-owning view. It's a key tool for zero-copy parsing.

---

### 24. Company: Tesla

**Role / Level**: Autopilot / Robotics Engineer
**When**: 2024
**Interview Format**: Multiple coding rounds + project deep dive.
**Questions asked**:

1. Given a stream of 3D sensor data points (x, y, z, timestamp), find the N closest points to the origin that arrived within the last T seconds.
2. How would you design a software watchdog in C++ to ensure system responsiveness in a safety-critical environment?
3. Explain stack vs. heap allocation and why avoiding dynamic allocation is critical in real-time systems.
   **Hints / Takeaways**:

- The sensor data problem requires a combination of a sliding window (e.g., a `std::deque`) and an efficient spatial data structure (like a k-d tree) or a simple min-heap.
- A watchdog involves a separate thread that must be periodically "pet" (reset); if not, it triggers a safe state or system reset.
- Dynamic allocation is non-deterministic and can lead to memory fragmentation. Discuss alternatives like memory pools, stack allocation (`alloca`), and static arenas.

---

### 25. Company: LinkedIn

**Role / Level**: Senior Staff Engineer (Backend)
**When**: 2025
**Interview Format**: Coding + Large-scale system design.
**Questions asked**:

1. Design the backend for "People You May Know" (connection suggestions).
2. You have a performance-critical C++ service that is CPU-bound. How do you profile and identify optimization opportunities on a Linux server?
3. Explain how C++20 coroutines could simplify an asynchronous network service compared to a traditional callback-based model.
   **Hints / Takeaways**:

- PYMK involves offline graph processing (e.g., using MapReduce/Spark) to pre-calculate suggestions, which are then served from a low-latency key-value store.
- For profiling, mention tools like `perf`, `gprof`, Valgrind's `callgrind`, and Flame Graphs to find hotspots in the code.
- Coroutines allow you to write asynchronous code that looks synchronous, avoiding "callback hell" and making state management across async operations much cleaner.

---

### 26. Company: Citadel Securities

**Role / Level**: Software Engineer (HFT)
**When**: 2025
**Interview Format**: 4 rounds (coding + low-level C++ + systems).
**Questions asked**:

1. Implement a non-blocking, lock-free SPSC (single-producer, single-consumer) queue.
2. Given a stream of market data ticks, design a data structure to query the VWAP (Volume Weighted Average Price) over any time interval.
3. How would you optimize a matrix multiplication function for a specific CPU architecture, beyond just tiling?
   **Hints / Takeaways**:

- The lock-free queue requires careful use of `std::atomic` with correct memory ordering (`acquire`/`release`) to ensure visibility without mutexes.
- For VWAP, a prefix sum array (or two: one for `price*volume`, one for `volume`) allows O(1) queries after an O(n) pre-computation.
- For matrix optimization, they're looking for discussion of SIMD intrinsics (AVX), instruction-level parallelism, and cache line alignment.

---

### 27. Company: ARM

**Role / Level**: Compiler Engineer
**When**: 2024
**Interview Format**: Technical interviews focused on compilers, architecture, and C++.
**Questions asked**:

1. For a simple `for` loop, explain how a compiler might perform loop unrolling and strength reduction optimizations.
2. What is the One Definition Rule (ODR) in C++, and what kind of subtle bugs can occur if you violate it (e.g., with inline functions in headers)?
3. Implement a simple AST (Abstract Syntax Tree) for arithmetic expressions (`+`, `-`, `*`, `/`, integers) and write a function to evaluate it.
   **Hints / Takeaways**:

- Be ready to discuss the trade-offs of optimizations: loop unrolling increases code size but reduces branching.
- ODR violations can lead to mysterious crashes or incorrect behavior that differs between builds, often related to mismatched type definitions.
- For the AST, using `std::unique_ptr` for tree nodes is a good modern C++ approach. An evaluation function can be a simple recursive visitor.

---

### 28. Company: MongoDB

**Role / Level**: Database Kernel Engineer
**When**: 2025
**Interview Format**: Coding + systems design with a focus on database internals.
**Questions asked**:

1. Design a buffer manager for a database. It should handle fetching pages from disk, caching them, and implementing an eviction policy (like CLOCK or LRU).
2. Explain what write-ahead logging (WAL) is and why it's crucial for database durability and atomicity (the 'A' and 'D' in ACID).
3. In C++, how does `dynamic_cast` work internally, and what is the performance overhead compared to `static_cast`?
   **Hints / Takeaways**:

- The buffer manager requires a hash map to look up pages and a separate data structure (like a queue or list) to manage the eviction order.
- For WAL, explain that changes are written to a sequential log file _before_ being applied to the data files, ensuring recovery after a crash.
- `dynamic_cast` relies on RTTI (Run-Time Type Information), which often involves a virtual table lookup and string comparison of type names, making it significantly slower than a compile-time `static_cast`.

---

### 29. Company: Riot Games

**Role / Level**: Backend Services Engineer (C++)
**When**: 2024
**Interview Format**: Virtual coding rounds + large-scale systems design.
**Questions asked**:

1. Design a matchmaking system for a 5v5 game that needs to balance player skill (MMR), low latency, and queue time.
2. Write C++ code to serialize a struct (containing `int`, `std::string`, `std::vector`) into a compact binary format.
3. Explain the "Rule of Zero" in modern C++ and how it relates to resource management.
   **Hints / Takeaways**:

- For matchmaking, discuss using a priority queue or bucketing system to find suitable matches and the trade-offs involved (e.g., widening the MMR search range over time).
- For serialization, focus on handling endianness, variable-length data (like strings and vectors) by prefixing them with their size.
- The Rule of Zero states that if your class doesn't manage a raw resource itself (and instead uses smart pointers/containers), you often don't need to write any custom destructor, copy/move constructors, or copy/move assignment operators.

---

### 30. Company: Cruise

**Role / Level**: Senior C++ Engineer (Perception)
**When**: 2025
**Interview Format**: 4-5 rounds of coding and system design focused on robotics.
**Questions asked**:

1. Implement a Kalman filter in C++ to track a moving object given noisy sensor measurements.
2. Given data from multiple sensors (e.g., LiDAR, camera) that arrive at slightly different times, design a system to synchronize and fuse them.
3. How do C++20 Concepts improve on SFINAE for writing generic code? Provide a simple example.
   **Hints / Takeaways**:

- The Kalman filter question is more about understanding the algorithm's state update and prediction steps rather than complex math. A simple 1D implementation is often sufficient.
- For sensor fusion, discuss timestamping, interpolation, and using a buffer or queue for each sensor stream to find the best temporal alignment.
- Concepts provide much clearer syntax and vastly improved compiler error messages compared to the template acrobatics required for SFINAE. `template<typename T> requires std::integral<T>` is much cleaner than the `std::enable_if` equivalent.

---

### 31. Company: OpenAI

**Role / Level**: Software Engineer (C++)  
**When**: 2025  
**Interview Format**: 3 rounds of optimization and code
**Questions asked**:

1. Implement a thread-safe singleton pattern in C++.
2. Given a huge log file, efficiently find the top 10 most frequent IP addresses.
3. Explain how C++17 `std::optional` works and how it compares to pointers or nullable types.  

**Hints / Takeaways**:

- Singleton: lazy init + `std::call_once` or `std::atomic`.
- Use streaming frequency count (heap or hashmap).
- `std::optional` is a lightweight wrapper — avoids null pointers but doesn’t allocate.
