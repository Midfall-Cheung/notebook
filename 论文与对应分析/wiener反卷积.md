# Wiener 反卷积

Wiener 反卷积是一种常用的**非盲线性图像恢复算法**。

## 1. 非盲

盲反卷积指事先不知道模糊核或 PSF，非盲指事先知道模糊核或 PSF。

## 2. 线性

假设一个系统：

$$
y = h * x + n
$$

其中：

- $y$ 是模糊图像；
- $x$ 是清晰图像；
- $h$ 是模糊核；
- $n$ 是噪声。

> PS：$n$ 与 $x$ 相互独立。

我们可以使用一个 Wiener 卷积核 $g$，使得：

$$
x = g * y
$$

---

# 如何求 Wiener 卷积核 $g$？

在空域中进行卷积：

$$
\hat{x} = g * y
$$

变换到频域：

$$
\hat{X} = GY
$$

我们希望：

$$
\hat{X} = X
$$

因此需要优化：

$$
\min \epsilon = E\left[|X-\hat{X}|^2\right]
$$

这是一个二次优化问题，属于凸优化，极值点出现在导数为 $0$ 的位置。

---

## 1. 展开均方误差

由于：

$$
Y = HX + N
$$

所以：

$$
\epsilon
=
E|X-GY|^2
=
E|(1-GH)X-GN|^2
$$

将上式展开：

$$
\begin{aligned}
\epsilon
={}&
(1-GH)(1-GH)^*E|X|^2 \\
&-(1-GH)G^*E(XN^*) \\
&-G(1-GH)^*E(NX^*) \\
&+GG^*E|N|^2
\end{aligned}
$$

---

## 2. 利用信号与噪声相互独立

由于假设噪声与原始信号相互独立，因此有：

$$
E(XN^*) = E(NX^*) = 0
$$

同时规定功率密度：

$$
E|X|^2 = S
$$

$$
E|N|^2 = M
$$

定义噪信比：

$$
\frac{M}{S}=NSR
$$

其中 $NSR$ 表示 Noise-to-Signal Ratio（噪声信号功率比）。

因此误差可以简化为：

$$
\epsilon
=
(1-GH)(1-GH)^*S
+
GG^*M
$$

---

## 3. 求最优 Wiener 滤波器

令：

$$
\frac{d\epsilon}{dG}
=
G^*M-H(1-GH)^*S
=
0
$$

最终得到：

$$
G
=
\frac{H^*S}
{|H|^2S+M}
$$

因为：

$$
NSR=\frac{M}{S}
$$

所以：

$$
G
=
\frac{H^*}
{|H|^2+NSR}
$$

也可以写成：

$$
G
=
\frac{1}{H}
\frac{|H|^2}
{|H|^2+NSR}
$$

因此，Wiener 反卷积滤波器为：

$$
\boxed{
G(u,v)
=
\frac{H^*(u,v)}
{|H(u,v)|^2+NSR(u,v)}
}
$$