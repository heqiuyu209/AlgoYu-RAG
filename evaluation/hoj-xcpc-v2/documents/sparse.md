# Sparse Table 静态区间查询

## 重叠查询

st[i][j] 保存从 i 开始长度 2^j 的区间最小值。查询 [l,r]，令 k=floor(log2(r-l+1))，取 min(st[l][k],st[r-2^k+1][k])。两段允许重叠，因为 min 是幂等运算。预处理 O(n log n)，查询 O(1)，不支持快速单点更新。

## 求和反例

普通 Sparse Table 的两段重叠公式不能直接用于 sum，重叠元素会被重复累加。区间和可以用静态前缀和相减；非幂等运算可考虑 Disjoint Sparse Table，本讲义不展开其构造。
