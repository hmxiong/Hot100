"""
常用运行指令: python3 acm_template.py
快速测试指令 (从文件读入): python3 acm_template.py < input.txt
"""
import sys
import collections
import heapq
import math
from typing import List, Optional

class Solution:
    def solve(self):
        """
        在这里实现你的核心逻辑
        """
        # TODO: 补充具体的解题代码
        pass

def main():
    # ==========================================
    # 模式1：sys.stdin.read().split() 
    # 优势：一次性把所有的输入全部读进来并按空格/换行切成列表，
    # 提取数字特别方便，适合大部分纯数字或无空格字符串题
    # ==========================================
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # 用一个迭代器方便提取数据
    it = iter(input_data)
    
    # 举例：读取 t 组数据
    try:
        t = int(next(it))
        for _ in range(t):
            # n = int(next(it))
            # arr = [int(next(it)) for _ in range(n)]
            
            sol = Solution()
            sol.solve()
    except StopIteration:
        pass
    """

    # ==========================================
    # 模式2：sys.stdin 按行读取 (默认打开)
    # 优势：适合一行一行有严格结构的数据，或者带有空格的字符串题
    # ==========================================
    for line in sys.stdin:
        # 去掉换行符和首尾空格
        line = line.strip()
        if not line:
            continue
        
        # TODO: 在这里解析当前行数据，然后传递给 solution
        # 比如：a, b = map(int, line.split())
        
        sol = Solution()
        sol.solve()

if __name__ == "__main__":
    # 为了避免递归深度超过限制，有些题需要解开这个注释
    # sys.setrecursionlimit(200000)
    main()
