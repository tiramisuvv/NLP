## 4 Feed-Forward Neural Networks and Backpropagation

Page 7-11
$$
\Delta W_{[u, v]} = - \eta \frac{\partial E}{\partial W_{[u, v]}}
$$


1. 从output 到 hidden layer

$$
\begin{align}
\Delta W_{[h, o]} &= - \eta \frac{\partial E}{\partial W_{[h, o]}}\\
&= -\eta \frac{\partial E}{\partial y_o} \cdot\frac{\partial y_o}{\partial s_o} \cdot \frac{\partial s_o}{\partial W_{[h, o]}}
\end{align}
$$

这里从后向前推，分别是error $\rightarrow$ $y_o=\sigma(s_o)$  $\rightarrow$ $s_o = W\cdot y_v$  


下面分别计算每一项：

$$
\begin{align}
\frac{\partial E}{\partial y_o} &= \frac{\frac{1}{2} \sum_{k\in O}e_k^2}{\partial y_o} = \frac{\frac{1}{2}e_o^2}{\partial y_o} = e_o \frac{e_o}{\partial y_o} = -(d_o-y_o)\\
\frac{\partial y_o}{\partial s_o} &= y_o(1-y_o), \quad \because y_o = \sigma(s_o), \sigma'(x) = \sigma(x)(1-\sigma(x))\\
\frac{\partial s_o}{\partial W_{[h, o]}} &= \frac{\partial (\sum_v W_{[v, o]}y_v)}{\partial W_{[h,o]}} = y_h
\end{align}
$$


对output unit o 定义 error signal :

$$
\vartheta_o :=-\frac{\partial E}{\partial y_o}\frac{\partial y_o}{\partial s_o} = (d_o-y_o)y_o(1-y_o)
$$


因此，可以基于下面公式更新hidden unit h 和 output unit o 之间的weight W

$$
\Delta W_{[h, o]} = \eta \vartheta_o y_h
$$



2. 从hidden  到 hidden layer

$$
\Delta W_{[i, h]} = - \eta \sum_o \frac{\partial E}{\partial y_o} \frac{\partial y_o}{\partial s_o} \cdot \frac{\partial s_o}{\partial y_h} \cdot \frac{\partial y_h}{\partial s_h} \cdot \frac{\partial s_h}{\partial W_{[i,h]}} \quad \text{with} \ o \in \text{Suc}(h)
$$

逐项计算：
$$
\begin{align}
&- \frac{\partial E}{\partial y_o} \frac{\partial y_o}{\partial s_o}  = \vartheta_o \\
&\frac{\partial s_o}{\partial y_h} = \frac{\partial (\sum_v W_{[v, o]}y_v)}{\partial y_h} = W_{[h,o]} \\
& \frac{\partial y_h}{\partial s_h} =  y_h(1-y_h), \quad \because y_h = \sigma(s_h)\\
&\frac{\partial s_h}{\partial W_{[i,h]}} = \frac{\partial (\sum_v W_{[v, h]}y_v)}{\partial W_{[i,h]}} = y_i
\end{align}
$$
有
$$
\Delta W_{[i, h]} = \eta \sum_o \vartheta_o \cdot W_{[h,o]} \cdot y_h(1-y_h)y_i
$$
对hidden unit o 定义 error signal :
$$
\vartheta_h =\sum_o \vartheta_o \cdot W_{[h,o]} \cdot y_h(1-y_h)  \quad \text{with} \ o \in \text{Suc}(h)
$$
有统一的表达：
$$
\Delta W_{[v,u]} = \eta \vartheta_u y_v
$$

## RNN

### section 6.1

$$
\begin{align}
\Delta W_{[u,v]} = - \eta\frac{\partial E_{total}(t',t)}{\partial W_{[u,v]}}
\end{align}
$$

其中


$$
\begin{align}
\frac{\partial E_{total}(t',t)}{\partial W_{[u,v]}} &=\frac{\partial \sum_{\tau = t'}^t E(\tau)} {\partial W_{[u,v]}}\\
&=\sum_{\tau = t'}^t\frac{\partial  E(\tau)} {\partial W_{[u,v]}}\\
\end{align}
$$

$$
\begin{align}
\frac{\partial  E(\tau)} {\partial W_{[u,v]}} & =\frac{\partial  E(\tau)} {\partial z_u}\cdot \frac{\partial  z_u} {\partial W_{[u,v]}}\\
&=\vartheta_u \cdot \frac{\partial  z_u} {\partial W_{[u,v]}}\\
&=\vartheta_u \cdot X_{[u,v]}(\tau)\\

\end{align}
$$

最后一个等式成立因为：
$$
z_u(\tau) = \sum_{l} W_{[u, l]}X_{[l,u]}(\tau), \quad \text{with } l\in \text{Pre}(u)
$$







$$
\begin{align}
\vartheta_u &= \frac{\partial  E(\tau)} {\partial z_u(\tau)}\\
&= \frac{1}{2}\sum_{k\in U}\frac{\partial[ (e_k(\tau))^2]} {\partial z_u(\tau)} \\
&= \frac{1}{2}\sum_{k\in U}2(e_k(\tau))\frac{\partial(e_k(\tau))}{\partial y_u(\tau)}\cdot \frac{{\partial y_u(\tau)}}{\partial z_u(\tau)} \\
&=\sum_{k\in U}e_k(\tau)\cdot \frac{\partial(e_k(\tau))}{\partial y_u(\tau)} \cdot f'_u(z_u(\tau))
\end{align}
$$


