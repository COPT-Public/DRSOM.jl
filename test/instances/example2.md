


Consider:


$$
\begin{aligned}
f_\epsilon(x,y) &= \sin^2(x) + y^2\left(\sin^2(x) - \epsilon\right); \epsilon > 0\\
f_x &=  (y^2 + 1) \left(2\sin(x)\cos(x)\right) \\
f_y & = 2y(\sin^2(x) - \epsilon)
\end{aligned}
$$

we write Hessian

$$
H(x,y) = \begin{bmatrix}
-2 y^2 \sin ^2(x)+2 y^2 \cos ^2(x)-2 \sin ^2(x)+2 \cos ^2(x) & 4 y \sin (x) \cos (x) \\
4 y \sin (x) \cos (x) & 2\left(\sin ^2(x) - \epsilon\right)
\end{bmatrix}
$$

We know $(x, y)=(k\cdot\pi, 0)$ are first-order points, so we have,
$$\epsilon > 0 \Rightarrow H_{22}(k\cdot\pi, 0) < 0,
$$
which implies even though we have infinite-number of first-order points, **all** of them are saddle.
