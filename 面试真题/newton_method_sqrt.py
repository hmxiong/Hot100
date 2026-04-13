import sys

def newton_sqrt(x, epsilon=1e-7):
    """
    使用牛顿迭代法求平方根的 Python 版本
    :param x: 被开方数
    :param epsilon: 迭代停止精度
    :return: 平方根结果
    """
    if x < 0:
        return None
    if x == 0:
        return 0.0
    
    # 初始猜测值
    val = float(x)
    
    while True:
        # 牛顿迭代核心公式：x_{k+1} = (x_k + x / x_k) / 2
        next_val = 0.5 * (val + x / val)
        
        # 判断两次迭代的差异是否小于精度要求
        if abs(next_val - val) < epsilon:
            break
        val = next_val
        
    return val

def main():
    # ACM 模式读取输入
    # sys.stdin.read().split() 会读取所有输入并按空格/换行切分
    inputs = sys.stdin.read().split()
    if not inputs:
        return
        
    for item in inputs:
        try:
            n = float(item)
            if n < 0:
                print("Invalid input")
                continue
                
            result = newton_sqrt(n)
            # 输出保留6位小数，与 C++ 版本对齐
            print(f"{result:.6f}")
        except ValueError:
            # 忽略非数字输入
            continue

if __name__ == "__main__":
    main()
