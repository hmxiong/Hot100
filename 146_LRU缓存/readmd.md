请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

 

示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
 

提示：

1 <= capacity <= 3000
0 <= key <= 10000
0 <= value <= 105
最多调用 2 * 105 次 get 和 put

解题思路：
- LRU 的规则是：每次 get/put 访问一个 key，都把它变成“最近使用”；当容量满了要淘汰“最久未使用”的 key。
- 要求 get/put 平均 O(1) ，所以需要两种结构配合：
  - 哈希表： key -> 节点位置 ，用于 O(1) 找到某个 key 是否存在、以及它在哪
  - 双向链表（C++ 用 list ）：维护“使用顺序”，表头是最近使用，表尾是最久未使用
- 操作：
  - get(key) ：哈希表找不到返回 -1；找到就把对应节点移动到链表头（变最近）
  - put(key,value) ：
    - 若已存在：更新 value，并移动到链表头
    - 若不存在：容量满则删链表尾 + 哈希表对应项；再把新节点插到链表头，并在哈希表登记
时间复杂度：平均 O(1) ；空间 O(capacity) 。