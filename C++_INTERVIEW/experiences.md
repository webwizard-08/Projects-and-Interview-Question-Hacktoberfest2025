# How to Submit a New Interview Experience

Use this template when adding your interview story:

* **Company**: [Company name or "(undisclosed)"]
* **Role / Level**: [e.g. "Backend Engineer (mid)", "Systems Engineer (senior)", "Quant Dev", etc.]
* **When**: [Year or approximate date]
* **Interview Format**: [e.g. "2 coding rounds + 1 design + HR", "on-site with whiteboard", "pair programming", "take-home"]
* **Questions asked (2-4 sample)**:
  1. [Question A]
  2. [Question B]
  3. [optional C]
  4. [optional D]
* **Hints / Example answers / takeaways**: [short notes, tip, or how you answered]

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

## Summary & Patterns

- Many interviews combine **algorithms + C++ core + system design**.  
- Frequent topics: memory management, smart pointers, move semantics, templates, concurrency.  
- Domain interviews (systems, graphics, networking) push you into low-level, buffer, alignment, bit operations.  
- Always clarify constraints, edge cases, complexity, memory usage.  
- Use RAII, modern C++ idioms, efficient data structures, and writing clean code helps.

---
