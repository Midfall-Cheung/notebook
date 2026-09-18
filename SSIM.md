# SSIM（Structural Similarity Index）详细推导

## 1. SSIM 是什么

SSIM，全称为 **Structural Similarity Index Measure，结构相似性指数**。

它和 MSE、PSNR 的思路不同。

MSE 和 PSNR 本质上主要比较：

> 两幅图像对应位置的像素值相差多少。

而 SSIM 更关注：

> 两幅图像在亮度、对比度和局部结构方面是否相似。

因此，SSIM 将图像相似性拆分为三个部分：

1. **Luminance：亮度相似性**
2. **Contrast：对比度相似性**
3. **Structure：结构相似性**

最终：

$$
\mathrm{SSIM}(x,y)
=
l(x,y)^{\alpha}
c(x,y)^{\beta}
s(x,y)^{\gamma}
$$

其中：

* \(x\) 表示参考图像的一个局部窗口；
* \(y\) 表示待评价图像的对应局部窗口；
* \(l(x,y)\) 表示亮度相似性；
* \(c(x,y)\) 表示对比度相似性；
* \(s(x,y)\) 表示结构相似性；
* \(\alpha,\beta,\gamma\) 用于控制三个部分的重要程度。

通常取：

$$
\alpha=\beta=\gamma=1
$$

于是：

$$
\mathrm{SSIM}(x,y)
=
l(x,y)c(x,y)s(x,y)
$$

---

# 2. 为什么 SSIM 不直接比较像素？

假设原图为：

$$
x=
\begin{bmatrix}
100 & 100\\
100 & 100
\end{bmatrix}
$$

恢复图为：

$$
y=
\begin{bmatrix}
110 & 110\\
110 & 110
\end{bmatrix}
$$

从像素值上来看，每个像素都有误差：

$$
|100-110|=10
$$

因此 MSE 不为零。

但是，两幅图的结构其实完全相同，只是整体亮了一点。

再例如：

$$
x=
\begin{bmatrix}
0 & 0\\
255 & 255
\end{bmatrix}
$$

与：

$$
y=
\begin{bmatrix}
10 & 10\\
245 & 245
\end{bmatrix}
$$

虽然像素值不同，但它们都保留了相同的水平边缘结构。

所以 SSIM 的思想是：

> 不仅比较像素绝对值，还要比较图像的局部统计结构。

---

# 3. SSIM 的基本统计量

假设局部窗口中有 \(N\) 个像素。

参考图像块为：

$$
x=
\{x_1,x_2,\dots,x_N\}
$$

恢复图像块为：

$$
y=
\{y_1,y_2,\dots,y_N\}
$$

我们首先计算三个基本统计量：

* 均值；
* 方差 / 标准差；
* 协方差。

---

# 4. 均值：描述亮度

参考图像窗口的均值为：

$$
\mu_x
=
\frac{1}{N}
\sum_{i=1}^{N}x_i
$$

恢复图像窗口的均值为：

$$
\mu_y
=
\frac{1}{N}
\sum_{i=1}^{N}y_i
$$

均值实际上表示这个局部区域的平均灰度。

例如：

$$
x=
[100,102,98,100]
$$

那么：

$$
\mu_x
=
\frac{100+102+98+100}{4}
=
100
$$

因此，可以把：

$$
\mu_x
$$

理解成参考区域的局部亮度。

---

# 5. 第一个组成部分：亮度相似性

如果两幅图的局部亮度相近，那么：

$$
\mu_x
\approx
\mu_y
$$

我们需要构造一个函数，使得：

* 两者相等时，相似性等于 1；
* 差异越大，相似性越低。

SSIM 使用：

$$
l(x,y)
=
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
$$

先暂时忽略常数 \(C_1\)：

$$
l(x,y)
=
\frac{
2\mu_x\mu_y
}{
\mu_x^2+\mu_y^2
}
$$

为什么这样设计？

因为根据：

$$
(\mu_x-\mu_y)^2\geq0
$$

展开：

$$
\mu_x^2
-2\mu_x\mu_y
+\mu_y^2
\geq0
$$

于是：

$$
2\mu_x\mu_y
\leq
\mu_x^2+\mu_y^2
$$

所以：

$$
\frac{
2\mu_x\mu_y
}{
\mu_x^2+\mu_y^2
}
\leq1
$$

当且仅当：

$$
\mu_x=\mu_y
$$

时：

$$
l(x,y)=1
$$

因此，这个形式非常适合衡量两个均值的相似程度。

---

# 6. 为什么要加常数 \(C_1\)？

如果：

$$
\mu_x\approx0
$$

并且：

$$
\mu_y\approx0
$$

那么分母：

$$
\mu_x^2+\mu_y^2
$$

可能非常接近零。

这样数值计算会非常不稳定。

所以引入：

$$
C_1>0
$$

最终得到：

$$
\boxed{
l(x,y)
=
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
}
$$

---

# 7. 标准差：描述对比度

接下来需要描述图像区域的灰度变化程度。

参考图像的方差为：

$$
\sigma_x^2
=
\frac{1}{N-1}
\sum_{i=1}^{N}
(x_i-\mu_x)^2
$$

标准差为：

$$
\sigma_x
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N}
(x_i-\mu_x)^2
}
$$

类似地：

$$
\sigma_y
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N}
(y_i-\mu_y)^2
}
$$

---

# 8. 为什么标准差表示对比度？

例如两个 patch：

第一幅：

$$
x=
[100,100,100,100]
$$

其均值为：

$$
\mu_x=100
$$

所有像素都一样，因此：

$$
\sigma_x=0
$$

这说明图像完全平坦，没有对比度。

第二幅：

$$
y=
[50,100,150,100]
$$

虽然平均值同样可能接近：

$$
100
$$

但是像素变化很大，因此：

$$
\sigma_y
$$

比较大。

所以：

$$
\boxed{
\sigma
\text{ 可以表示局部对比度强弱}
}
$$

---

# 9. 第二个组成部分：对比度相似性

对比度相似性定义为：

$$
\boxed{
c(x,y)
=
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
}
$$

如果先忽略 \(C_2\)：

$$
c(x,y)
=
\frac{
2\sigma_x\sigma_y
}{
\sigma_x^2+\sigma_y^2
}
$$

和前面的亮度公式完全类似。

根据：

$$
(\sigma_x-\sigma_y)^2\geq0
$$

得到：

$$
2\sigma_x\sigma_y
\leq
\sigma_x^2+\sigma_y^2
$$

因此：

$$
c(x,y)\leq1
$$

当：

$$
\sigma_x=\sigma_y
$$

时：

$$
c(x,y)=1
$$

所以，如果两幅图局部灰度变化程度一致，那么它们的对比度相似性最高。

---

# 10. 去掉亮度和对比度后比较结构

现在已经比较了：

$$
\mu_x
\quad\text{和}\quad
\mu_y
$$

以及：

$$
\sigma_x
\quad\text{和}\quad
\sigma_y
$$

接下来需要比较真正的：

> 图像结构是否一致。

先从图像中去掉平均亮度：

$$
x-\mu_x
$$

以及：

$$
y-\mu_y
$$

然后再除以标准差：

$$
\frac{x-\mu_x}{\sigma_x}
$$

和：

$$
\frac{y-\mu_y}{\sigma_y}
$$

因此得到两个标准化信号：

$$
\hat{x}
=
\frac{x-\mu_x}{\sigma_x}
$$

$$
\hat{y}
=
\frac{y-\mu_y}{\sigma_y}
$$

---

# 11. 为什么这样就能表示“结构”？

操作：

$$
x-\mu_x
$$

去除了平均亮度。

操作：

$$
\frac{x-\mu_x}{\sigma_x}
$$

又把整体对比度进行了归一化。

所以剩下的信息主要描述：

> 像素相对于周围像素是如何变化的。

也就是局部结构。

因此 SSIM 中所谓的 structure，可以理解为：

$$
\boxed{
\text{Structure}
=
\frac{
\text{图像}-\text{亮度}
}{
\text{对比度}
}
}
$$

---

# 12. 协方差：描述两幅图变化是否一致

两幅图之间的协方差定义为：

$$
\sigma_{xy}
=
\frac{1}{N-1}
\sum_{i=1}^{N}
(x_i-\mu_x)
(y_i-\mu_y)
$$

协方差的意义是：

如果：

$$
x_i-\mu_x>0
$$

同时：

$$
y_i-\mu_y>0
$$

二者乘积为正。

如果：

$$
x_i-\mu_x<0
$$

同时：

$$
y_i-\mu_y<0
$$

乘积也为正。

也就是说：

> 如果两幅图在相同位置上的灰度变化方向一致，协方差会比较大。

---

# 13. 第三个组成部分：结构相似性

结构相似性定义为：

$$
\boxed{
s(x,y)
=
\frac{
\sigma_{xy}+C_3
}{
\sigma_x\sigma_y+C_3
}
}
$$

如果暂时忽略常数：

$$
s(x,y)
=
\frac{
\sigma_{xy}
}{
\sigma_x\sigma_y
}
$$

你会发现，这实际上就是非常熟悉的：

$$
\boxed{
\rho_{xy}
=
\frac{
\sigma_{xy}
}{
\sigma_x\sigma_y
}
}
$$

即相关系数。

因此：

$$
\boxed{
\text{SSIM 中的结构比较，本质上和相关性有关}
}
$$

---

# 14. 为什么相关性可以表示结构相似？

假设：

$$
y=x
$$

那么：

$$
\sigma_{xy}
=
\sigma_x^2
$$

同时：

$$
\sigma_x=\sigma_y
$$

因此：

$$
s(x,y)
=
\frac{
\sigma_x^2
}{
\sigma_x\sigma_x
}
=
1
$$

说明结构完全一致。

---

如果：

$$
y=-x
$$

在去均值之后，两幅图结构完全相反，则：

$$
\sigma_{xy}<0
$$

所以结构相似度甚至可能变成负数。

因此理论上 SSIM 并不严格只在 0 到 1 之间。

---

# 15. 三个部分组合起来

现在已经得到：

## 亮度

$$
l(x,y)
=
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
$$

## 对比度

$$
c(x,y)
=
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
$$

## 结构

$$
s(x,y)
=
\frac{
\sigma_{xy}+C_3
}{
\sigma_x\sigma_y+C_3
}
$$

于是 SSIM 定义为：

$$
\boxed{
\mathrm{SSIM}(x,y)
=
l(x,y)^\alpha
c(x,y)^\beta
s(x,y)^\gamma
}
$$

---

# 16. 最常用的参数设置

通常设置：

$$
\alpha=\beta=\gamma=1
$$

所以：

$$
\mathrm{SSIM}(x,y)
=
l(x,y)c(x,y)s(x,y)
$$

同时选择：

$$
C_3=\frac{C_2}{2}
$$

于是：

$$
\mathrm{SSIM}(x,y)
=
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
\cdot
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
\cdot
\frac{
\sigma_{xy}+\frac{C_2}{2}
}{
\sigma_x\sigma_y+\frac{C_2}{2}
}
$$

---

# 17. 为什么最后可以化简？

观察后面两个部分：

$$
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
$$

乘：

$$
\frac{
\sigma_{xy}+\frac{C_2}{2}
}{
\sigma_x\sigma_y+\frac{C_2}{2}
}
$$

注意：

$$
2\sigma_x\sigma_y+C_2
=
2
\left(
\sigma_x\sigma_y+\frac{C_2}{2}
\right)
$$

因此：

$$
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
\cdot
\frac{
\sigma_{xy}+\frac{C_2}{2}
}{
\sigma_x\sigma_y+\frac{C_2}{2}
}
$$

可以约去：

$$
\sigma_x\sigma_y+\frac{C_2}{2}
$$

最终得到：

$$
\boxed{
\frac{
2\sigma_{xy}+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
}
$$

所以标准 SSIM 最终变成：

$$
\boxed{
\mathrm{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)
(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}
}
$$

这就是我们实际代码中最常见的 SSIM 公式。

---

# 18. 最终公式怎么理解？

最终公式：

$$
\mathrm{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)
(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}
$$

可以分成两个部分理解：

$$
\boxed{
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
}
$$

主要负责比较：

> 亮度。

而：

$$
\boxed{
\frac{
2\sigma_{xy}+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
}
$$

综合比较：

> 对比度和结构。

---

# 19. 常数 \(C_1\) 和 \(C_2\) 是怎么来的？

通常定义：

$$
C_1=(K_1L)^2
$$

$$
C_2=(K_2L)^2
$$

常用参数为：

$$
K_1=0.01
$$

$$
K_2=0.03
$$

其中：

$$
L
$$

表示图像的动态范围。

---

## 8-bit 图像

如果图像范围为：

$$
[0,255]
$$

则：

$$
L=255
$$

所以：

$$
C_1
=
(0.01\times255)^2
$$

即：

$$
C_1
=
6.5025
$$

而：

$$
C_2
=
(0.03\times255)^2
$$

即：

$$
C_2
=
58.5225
$$

---

## 归一化图像

如果图像范围为：

$$
[0,1]
$$

那么：

$$
L=1
$$

于是：

$$
C_1
=
(0.01)^2
=
0.0001
$$

$$
C_2
=
(0.03)^2
=
0.0009
$$

这也是为什么代码计算 SSIM 时，必须正确设置：

```python
data_range=1.0
```

或者：

```python
data_range=255
```

---

# 20. SSIM 不是整张图片直接算一次

实际 SSIM 通常不是直接对整张图计算：

$$
\mu_x,\sigma_x,\sigma_{xy}
$$

然后得到一个数字。

而是在一个局部窗口中计算。

例如：

$$
11\times11
$$

窗口。

过程类似：

```text
整幅图像
   ↓
取 11×11 局部窗口
   ↓
计算局部均值
   ↓
计算局部方差
   ↓
计算局部协方差
   ↓
计算局部 SSIM
   ↓
移动窗口
   ↓
得到整张 SSIM map
```

---

# 21. 局部 SSIM

对于图像中的位置 \(p\)，可以得到：

$$
\mathrm{SSIM}_p
$$

所以整幅图实际上对应一张：

$$
\boxed{\mathrm{SSIM\ Map}}
$$

例如：

```text
原图结构正确的位置：

SSIM ≈ 0.98


边缘恢复错误的位置：

SSIM ≈ 0.60


严重失真的位置：

SSIM ≈ 0.20
```

---

# 22. 最后的 SSIM 数值怎么得到？

将所有局部 SSIM 做平均：

$$
\boxed{
\mathrm{MSSIM}(X,Y)
=
\frac{1}{M}
\sum_{j=1}^{M}
\mathrm{SSIM}(x_j,y_j)
}
$$

其中：

* \(x_j\) 为参考图像第 \(j\) 个局部窗口；
* \(y_j\) 为恢复图像对应窗口；
* \(M\) 为窗口数量。

通常我们平时说的：

> “这张图片 SSIM 是 0.92”

实际上很多实现输出的是这种平均局部 SSIM。

---

# 23. SSIM 和 MSE 的本质区别

MSE：

$$
\mathrm{MSE}
=
\frac{1}{N}
\sum_i
(x_i-y_i)^2
$$

它只问：

> 对应像素差多少？

而 SSIM 问：

### 第一层

$$
\mu_x
\stackrel{?}{\approx}
\mu_y
$$

也就是：

> 亮度像不像？

### 第二层

$$
\sigma_x
\stackrel{?}{\approx}
\sigma_y
$$

也就是：

> 对比度像不像？

### 第三层

$$
\frac{
\sigma_{xy}
}{
\sigma_x\sigma_y
}
$$

也就是：

> 局部变化结构像不像？

因此：

$$
\boxed{
\mathrm{SSIM}
=
\mathrm{Brightness}
+
\mathrm{Contrast}
+
\mathrm{Structure}
}
$$

更准确来说，是三者的乘积关系。

---

# 24. 一个简单例子

假设一个局部窗口：

$$
x=
[100,110,120]
$$

恢复结果：

$$
y=
[102,112,122]
$$

可以看到：

$$
y=x+2
$$

也就是说整体只是亮度偏移了 2。

---

## 均值

$$
\mu_x
=
110
$$

$$
\mu_y
=
112
$$

亮度有轻微变化。

---

## 标准差

因为两个序列变化幅度完全一样：

$$
\sigma_x
\approx
\sigma_y
$$

所以：

$$
c(x,y)
\approx1
$$

---

## 结构

因为：

$$
y-\mu_y
=
x-\mu_x
$$

所以两者结构完全相同：

$$
s(x,y)\approx1
$$

因此整体：

$$
\mathrm{SSIM}
$$

仍然会非常接近 1。

这正体现了 SSIM：

> 对结构保持更加敏感，而不会因为小的整体亮度偏移就认为图像完全错误。

---

# 25. 对你的 HasDef 重建任务尤其意味着什么？

你现在研究的是：

$$
TD
\rightarrow
\widehat{HasDef}
$$

而 HasDef 里面非常重要的信息包括：

* 边缘；
* 线宽；
* 缺陷形状；
* bridge；
* break；
* hole；
* edge intrusion；
* edge extension。

这些都属于：

$$
\boxed{\text{结构信息}}
$$

因此 SSIM 对你的任务会比单独使用 MSE 更有意义。

例如：

### 情况 A

恢复结果：

* 整体灰度有一点偏差；
* 但是缺陷轮廓准确；
* 边缘位置准确。

可能：

$$
\mathrm{PSNR}
$$

不是特别高，

但：

$$
\mathrm{SSIM}
$$

仍然较高。

---

### 情况 B

恢复结果整体灰度很接近，但边缘被严重模糊。

可能：

$$
\mathrm{MSE}
$$

并没有特别大，

但是：

$$
\mathrm{SSIM}
$$

可能明显下降。

这也是你后续做 NAFNet、Wiener、BM3D 比较时，不能只看 PSNR 的原因。

---

# 26. SSIM 完整推导总结

首先计算：

$$
\mu_x
=
\frac{1}{N}
\sum_i x_i
$$

$$
\mu_y
=
\frac{1}{N}
\sum_i y_i
$$

然后：

$$
\sigma_x^2
=
\frac{1}{N-1}
\sum_i
(x_i-\mu_x)^2
$$

$$
\sigma_y^2
=
\frac{1}{N-1}
\sum_i
(y_i-\mu_y)^2
$$

协方差：

$$
\sigma_{xy}
=
\frac{1}{N-1}
\sum_i
(x_i-\mu_x)
(y_i-\mu_y)
$$

---

亮度：

$$
\boxed{
l(x,y)
=
\frac{
2\mu_x\mu_y+C_1
}{
\mu_x^2+\mu_y^2+C_1
}
}
$$

对比度：

$$
\boxed{
c(x,y)
=
\frac{
2\sigma_x\sigma_y+C_2
}{
\sigma_x^2+\sigma_y^2+C_2
}
}
$$

结构：

$$
\boxed{
s(x,y)
=
\frac{
\sigma_{xy}+C_3
}{
\sigma_x\sigma_y+C_3
}
}
$$

组合：

$$
\boxed{
\mathrm{SSIM}
=
l^\alpha
c^\beta
s^\gamma
}
$$

通常：

$$
\alpha=\beta=\gamma=1
$$

以及：

$$
C_3=\frac{C_2}{2}
$$

最终得到：

$$
\boxed{
\mathrm{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)
(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}
}
$$

---

# 27. 一句话理解 SSIM

可以把 SSIM 记成：

$$
\boxed{
\mathrm{SSIM}
=
\text{亮度相似性}
\times
\text{对比度相似性}
\times
\text{结构相似性}
}
$$

其中最关键的三个统计量分别是：

$$
\boxed{
\mu
\rightarrow
\text{亮度}
}
$$

$$
\boxed{
\sigma
\rightarrow
\text{对比度}
}
$$

$$
\boxed{
\sigma_{xy}
\rightarrow
\text{结构相关性}
}
$$

因此 SSIM 最核心的思想不是单纯判断：

> “两个像素是不是一样？”

而是判断：

> **“两幅图的局部亮度、变化幅度以及像素之间的结构关系是不是一样？”**

对于你的灰度图像恢复和缺陷结构重建任务，这正是 SSIM 相比单纯 MSE/PSNR 更有价值的原因。
