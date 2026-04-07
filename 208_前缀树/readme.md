Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
 

示例：

输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
 

提示：

1 <= word.length, prefix.length <= 2000
word 和 prefix 仅由小写英文字母组成
insert、search 和 startsWith 调用次数 总计 不超过 3 * 104 次

解题思路：
- 题目要支持三种操作：插入单词、查完整单词、查是否存在前缀，并且总调用次数较多，所以需要每次操作都接近“按字符走一遍”的复杂度。
- Trie（前缀树）的核心想法：把单词看成从根出发的一条路径，每个字符走一条边；路径上的公共部分（前缀）天然共享。
  - insert(word) ：沿字符路径走，不存在的节点就新建，最后节点打上“单词结束”标记。
  - search(word) ：沿路径走完后，必须“走得通”且最后节点 isEnd==true 。
  - startsWith(prefix) ：只要前缀路径“走得通”即可，不要求 isEnd 。
时间复杂度：每次 O(L) （L 为字符串长度），空间是节点总数（所有字符总量级）。