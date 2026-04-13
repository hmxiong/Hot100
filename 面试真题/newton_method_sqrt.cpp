#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;



// 使用牛顿迭代法求平方根
double newtonSqrt(double x, double epsilon = 1e-7) {
    if (x < 0) return -1.0; // 负数没有实数平方根
    if (x == 0) return 0.0;
    
    // 初始猜测值，可以选择 x 或 x/2
    double val = x; 
    double last_val;
    
    do {
        last_val = val;
        // 牛顿迭代公式：x_{k+1} = (x_k + x / x_k) / 2
        val = 0.5 * (val + x / val);
    } while (abs(val - last_val) > epsilon); // 直到两次迭代的差值小于给定精度
    
    return val;
}

int main() {
    // 优化输入输出
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    double n;
    // 循环读取输入，支持ACM模式下的多组测试用例
    while (cin >> n) {
        if (n < 0) {
            cout << "Invalid input\n";
            continue;
        }
        double result = newtonSqrt(n);
        // 输出保留6位小数
        cout << fixed << setprecision(6) << result << "\n";
    }
    
    return 0;
}
