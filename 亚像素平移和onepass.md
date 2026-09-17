1. 相位相关从 TS 与 HasDef 的边缘图中估计亚像素平移；
2. 一次性面积重采样把这个平移直接写入高分辨率标签的下采样积分区间；
3. 获得对齐像素后，使用这些像素对拟合单调 LUT。

---

# 一、相位相关的具体原理

## 1. 傅里叶平移定理

设目标边缘图为：

$$
f(x,y)
$$

TS 边缘图是它平移 $(\Delta x,\Delta y)$ 后的结果：

$$
g(x,y)=f(x-\Delta x,y-\Delta y)
$$

两张图的二维傅里叶变换分别为：

$$
F(u,v)=\mathcal{F}\{f(x,y)\}
$$

$$
G(u,v)=\mathcal{F}\{g(x,y)\}
$$

根据傅里叶平移定理：

$$
G(u,v)
=
F(u,v)
\exp\left[
-j2\pi
\left(
\frac{u\Delta x}{W}
+
\frac{v\Delta y}{H}
\right)
\right]
$$

其中：

- $W,H$：图像宽度和高度
- $u,v$：频率坐标
- $j=\sqrt{-1}$

因此，图像在空间域发生平移后，频域幅值基本不变，变化主要体现在相位上。

---

## 2. 构造归一化互功率谱

一种常见定义是：

$$
C(u,v)
=
\frac{
G(u,v)F^*(u,v)
}{
|G(u,v)F^*(u,v)|+\varepsilon
}
$$

其中 $F^*$ 表示复共轭，$\varepsilon$ 防止分母为零。

代入平移关系：

$$
G(u,v)F^*(u,v)
=
|F(u,v)|^2
\exp\left[
-j2\pi
\left(
\frac{u\Delta x}{W}
+
\frac{v\Delta y}{H}
\right)
\right]
$$

归一化后，幅值信息被消除：

$$
C(u,v)
\approx
\exp\left[
-j2\pi
\left(
\frac{u\Delta x}{W}
+
\frac{v\Delta y}{H}
\right)
\right]
$$

所以互功率谱只保留两张图之间的相位差，也就是平移信息。

如果使用相反的共轭顺序，最后得到的位移符号也会相反。实际正负方向由函数参数顺序决定。

---

## 3. 逆傅里叶变换得到相关峰

对互功率谱做逆傅里叶变换：

$$
r(x,y)=\mathcal{F}^{-1}\{C(u,v)\}
$$

理想情况下：

$$
r(x,y)
=
\delta(x-\Delta x,y-\Delta y)
$$

也就是说，相关图在真实位移位置出现一个脉冲峰：

$$
(\hat{\Delta x},\hat{\Delta y})
=
\arg\max_{x,y}r(x,y)
$$

如果最大峰出现在：

$$
(1,-2)
$$

就说明两张图之间大约存在横向 1 像素、纵向 −2 像素的平移。

---

## 4. 亚像素位移是怎么得到的

整数位移可以直接读取相关图峰值坐标，但真实峰值往往位于像素之间。

例如局部相关值为：

| 位置 | 相关强度 |
|---|---:|
| $x=4$ | 0.50 |
| $x=5$ | 0.90 |
| $x=6$ | 0.70 |

真实峰值可能在 $x=5.2$，而不是整数 5。

OpenCV 的 `phaseCorrelate` 会在峰值附近做局部质心估计，从而返回浮点位移：

$$
\hat{x}
=
\frac{\sum_{(x,y)\in\Omega}x\,r(x,y)}
{\sum_{(x,y)\in\Omega}r(x,y)}
$$

$$
\hat{y}
=
\frac{\sum_{(x,y)\in\Omega}y\,r(x,y)}
{\sum_{(x,y)\in\Omega}r(x,y)}
$$

其中 $\Omega$ 是相关峰附近的局部区域。

因此可能得到：

$$
\Delta x=0.43,\qquad
\Delta y=-0.27
$$

而不只是整数位移。

---

## 5. `response` 表示什么

`cv2.phaseCorrelate` 同时返回：

```python
(dx, dy), response
```

`response` 可以理解为相关峰的集中程度或可信程度：

- 峰值尖锐、唯一：response 较高
- 多个相似峰：response 较低
- 图像缺少边缘：response 较低
- 周期结构产生多个候选位移：response 可能降低

项目设置：

$$
response\geq0.8
$$

并要求：

$$
\sqrt{dx^2+dy^2}\leq3
$$

否则将位移置为零。

---

# 二、一次性面积重采样如何融合亚像素平移

这里不是先缩小图像，再对缩小结果进行双线性平移。

它直接改变每个 TS 输出像素在高分辨率原图上的积分区域。

## 1. 没有平移时

假设高分辨率图像宽度为 $W_{\mathrm{in}}$，TS 宽度为 $W_{\mathrm{out}}$。

横向缩放比为：

$$
s_x=\frac{W_{\mathrm{in}}}{W_{\mathrm{out}}}
$$

输出图第 $j$ 个像素对应原图中的区间：

$$
X_j=
[js_x,(j+1)s_x)
$$

例如：

$$
s_x=4
$$

那么输出像素 $j=2$ 对应原图：

$$
X_2=[8,12)
$$

它会对原图第 8 到 11 个像素覆盖的区域做面积平均。

---

## 2. 加入亚像素平移

如果位移为 $\Delta x$，代码将积分区间修改为：

$$
X_j(\Delta x)
=
[(j-\Delta x)s_x,\,(j+1-\Delta x)s_x)
$$

纵向同理：

$$
Y_i(\Delta y)
=
[(i-\Delta y)s_y,\,(i+1-\Delta y)s_y)
$$

其中：

$$
s_y=\frac{H_{\mathrm{in}}}{H_{\mathrm{out}}}
$$

这正对应代码：

```python
left = (out_index - shift) * scale
right = left + scale
```

见 [data.py](/home/zjzhang/Desktop/灰度图像恢复/灰度图像恢复/different_ways/DNCNN/data.py:186)。

---

## 3. 为什么这就实现了平移

假设：

$$
s_x=4,\qquad \Delta x=0.25
$$

原本输出像素 $j=2$ 对应：

$$
[8,12)
$$

加入平移后：

$$
[(2-0.25)\times4,\,(3-0.25)\times4)
=
[7,11)
$$

积分区域在高分辨率原图中向左移动了 1 个高分辨率像素。

因为：

$$
0.25\text{ 个 TS 像素}
\times
4
=
1\text{ 个高分辨率像素}
$$

最终在输出 TS 网格上，图像内容就相当于向右移动了 0.25 个像素。

---

## 4. 面积权重的具体公式

设原图第 $n$ 个横向像素覆盖区间：

$$
P_n=[n,n+1)
$$

输出像素 $j$ 对应的平移后区间是：

$$
X_j=[l_j,r_j)
$$

其中：

$$
l_j=(j-\Delta x)s_x
$$

$$
r_j=l_j+s_x
$$

原图像素 $n$ 对输出像素 $j$ 的横向权重为：

$$
w^x_{j,n}
=
\frac{
|X_j\cap P_n|
}{
s_x
}
$$

展开为：

$$
w^x_{j,n}
=
\frac{
\max
\left(
0,
\min(r_j,n+1)-\max(l_j,n)
\right)
}{
s_x
}
$$

纵向权重同理：

$$
w^y_{i,m}
=
\frac{
\max
\left(
0,
\min(b_i,m+1)-\max(t_i,m)
\right)
}{
s_y
}
$$

其中：

$$
t_i=(i-\Delta y)s_y
$$

$$
b_i=t_i+s_y
$$

最终二维输出像素为：

$$
O(i,j)
=
\sum_m\sum_n
w^y_{i,m}
w^x_{j,n}
I(m,n)
$$

因为横向和纵向权重可以分开计算，所以代码使用可分离实现：

1. 先做横向面积加权；
2. 再做纵向面积加权。

对应 [data.py](/home/zjzhang/Desktop/灰度图像恢复/灰度图像恢复/different_ways/DNCNN/data.py:238)。

---

## 5. 为什么它叫“一次性”

传统 two-pass 是：

$$
I_{\mathrm{small}}
=
\operatorname{AreaResize}(I)
$$

然后：

$$
O
=
\operatorname{BilinearShift}
(I_{\mathrm{small}},\Delta x,\Delta y)
$$

数据被插值两次。

当前 one-pass 直接计算：

$$
O(i,j)
=
\frac{1}{s_xs_y}
\iint_{R_{ij}(\Delta x,\Delta y)}
I(x,y)\,dx\,dy
$$

其中 $R_{ij}$ 就是已经包含亚像素位移的原图积分区域。

因此下采样和平移在同一次面积积分中完成，避免第二次双线性插值造成边缘模糊。

---

# 三、LUT 使用已配准像素对的具体公式

## 1. 训练像素对

对于每个训练样本，先得到：

$$
u_s(p)
=
P(TS_s(p))
$$

其中：

- $s$：样本编号
- $p$：TS 网格上的像素位置
- $P$：极性规范化函数
- $u_s(p)$：极性规范化后的 TS

同时得到已配准目标：

$$
x_s(p)
=
\operatorname{Register}
(HasDef_s)(p)
$$

因为两者处于同一个 TS 网格，所以：

$$
\bigl(u_s(p),x_s(p)\bigr)
$$

构成一个像素级训练对。

---

## 2. 将 TS 灰度划分为 1024 个区间

设 LUT 分箱数为：

$$
K=1024
$$

对于一个 TS 灰度 $u_i\in[0,1]$，分箱编号为：

$$
b_i
=
\min
\left(
\lfloor Ku_i\rfloor,
K-1
\right)
$$

其中 $i$ 可以理解为遍历全部训练样本和全部像素后的统一索引。

例如：

$$
u_i=0.5
$$

则：

$$
b_i=\lfloor1024\times0.5\rfloor=512
$$

---

## 3. 统计每个灰度档对应的目标均值

第 $k$ 个灰度档的像素数量为：

$$
N_k
=
\sum_i\mathbf{1}(b_i=k)
$$

对应 HasDef 灰度总和为：

$$
S_k
=
\sum_{i:b_i=k}x_i
$$

于是该灰度档的原始映射值为：

$$
m_k
=
\frac{S_k}{N_k}
$$

也就是：

$$
m_k
\approx
E[X\mid U\in B_k]
$$

它回答的问题是：

> 当极性规范化后的 TS 灰度落在第 $k$ 档时，同一位置的 HasDef 平均灰度是多少？

---

## 4. 空分箱插值

如果某些灰度档没有训练像素：

$$
N_k=0
$$

则无法直接计算 $m_k$。

代码使用相邻已有分箱线性插值。例如：

$$
m_{100}=0.20,\qquad m_{104}=0.28
$$

那么中间可以补成：

$$
m_{101}=0.22
$$

$$
m_{102}=0.24
$$

$$
m_{103}=0.26
$$

---

## 5. 局部平滑

默认平滑半径：

$$
r=4
$$

意味着使用当前分箱左右各 4 档，总共最多 9 档进行加权平滑：

$$
\widetilde{m}_k
=
\frac{
\sum_{q=k-r}^{k+r}
w_qm_q
}{
\sum_{q=k-r}^{k+r}w_q
}
$$

项目中的权重近似为：

$$
w_q=\max(N_q,1)
$$

样本多的灰度档权重更大，样本少的灰度档对曲线影响较小。

---

## 6. 单调回归

经过统计和平滑后，曲线仍可能出现：

$$
\widetilde{m}_{k+1}
<
\widetilde{m}_k
$$

也就是 TS 更亮，LUT 输出反而更暗。

项目使用加权 PAVA，求解：

$$
\min_{z_0,\ldots,z_{K-1}}
\sum_{k=0}^{K-1}
w_k
(z_k-\widetilde{m}_k)^2
$$

约束为：

$$
0\leq z_0\leq z_1\leq\cdots\leq z_{K-1}\leq1
$$

最终：

$$
L_k=z_k
$$

就是保存到 manifest 中的 1024 个 LUT 值。

---

## 7. 推理时线性插值

第 $k$ 个 LUT 采样点的中心位置为：

$$
c_k=\frac{k+0.5}{K}
$$

如果输入灰度 $u$ 位于：

$$
c_k\leq u\leq c_{k+1}
$$

则输出为：

$$
L(u)
=
L_k+
\frac{u-c_k}{c_{k+1}-c_k}
(L_{k+1}-L_k)
$$

如果输入位于 LUT 范围两端之外，则使用首尾值：

$$
L(u)=
\begin{cases}
L_0,&u<c_0\\
L_{K-1},&u>c_{K-1}
\end{cases}
$$

---

# 四、三个环节之间的完整数学关系

对训练样本 $s$：

### 位移估计

$$
(\Delta x_s,\Delta y_s)
=
\operatorname{PhaseCorrelate}
\left(
E(HasDef_s^\downarrow),
E(TS_s)
\right)
$$

### 一次性配准

$$
x_s
=
\operatorname{OnePassArea}
\left(
HasDef_s,
\Delta x_s,
\Delta y_s
\right)
$$

### 极性规范化

$$
u_s=P(TS_s)
$$

### LUT 拟合

$$
L
=
\arg\min_{L\text{ 单调}}
\sum_s\sum_p
\left[
L(u_s(p))-x_s(p)
\right]^2
$$

项目没有直接对所有像素求一个任意函数，而是通过：

```text
1024 分箱
每箱目标均值
空箱插值
局部平滑
PAVA 单调回归
```

近似求解这个问题。

最终网络输入为：

$$
y_s(p)=L(u_s(p))
$$

恢复网络学习：

$$
\hat{x}_s
=
\operatorname{Restore}(y_s)
$$

所以三者的分工是：

- 相位相关：估计几何位移；
- 一次性面积重采样：生成与 TS 对齐的监督目标；
- LUT：在对齐像素基础上学习灰度响应关系。