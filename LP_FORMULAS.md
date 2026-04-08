# Problem Formulation

## Variable Definitions

$$
n:\ \text{number of users},\quad i\in\{1,\dots,n\}
$$

$$
\mu_{1i}:\ \Pr(Y_i=1\mid T_i=1),\quad \mu_{0i}:\ \Pr(Y_i=1\mid T_i=0)
$$

$$
\text{Treatment Effect}: u_i=\mu_{1i}-\mu_{0i}
$$

$$
x_i\in\{0,1\},\quad x=(x_1,\dots,x_n)
$$

$$
f\in[0,1],\quad \text{Number of users (k)}: k=\operatorname{round}(nf),\quad \sum_{i=1}^{n}x_i=k
$$

$$
p:\ \text{price per conversion},\quad c_u:\ \text{unit cost per conversion},\quad c_m:\ \text{marketing cost per treated user}
$$

$$
B:\ \text{marketing budget cap}
$$

$$
E_{\mathrm{conv}}(x)=\sum_{i=1}^{n}\left[x_i\mu_{1i}+(1-x_i)\mu_{0i}\right]
$$

$$
R(x)=p\,E_{\mathrm{conv}}(x),\quad
\Pi(x)=p\,E_{\mathrm{conv}}(x)-c_u\,E_{\mathrm{conv}}(x)-c_m\sum_{i=1}^{n}x_i
$$

$$
V(x)=\sum_{i=1}^{n}\left[x_i\mu_{1i}+(1-x_i)\mu_{0i}\right]
$$

## Common Constraints

$$
x_i\in\{0,1\}\ \forall i,\quad \sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

## Profit Optimization

$$
\max_{x\in\{0,1\}^{n}}\;\Pi(x)
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

$$
\max_{x\in\{0,1\}^{n}}\;\sum_{i=1}^{n}x_i\left((p-c_u)(\mu_{1i}-\mu_{0i})-c_m\right)
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

## Revenue Optimization

$$
\max_{x\in\{0,1\}^{n}}\;R(x)
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

$$
\max_{x\in\{0,1\}^{n}}\;\sum_{i=1}^{n}\left[x_i\mu_{1i}+(1-x_i)\mu_{0i}\right]
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

## Visit Optimization

$$
\max_{x\in\{0,1\}^{n}}\;V(x)
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

## Outer Split Optimization

$$
f\in\{0,0.01,\dots,1.00\},\quad k=\operatorname{round}(nf),\quad c_m k\le B
$$

$$
x^{\star}(f)=\arg\max_{x\in\{0,1\}^{n}}\;J_f(x)
\quad\text{s.t.}\quad
\sum_{i=1}^{n}x_i=k,\quad c_m\sum_{i=1}^{n}x_i\le B
$$

$$
f^{\star}=\arg\max_{f}J_f\!\left(x^{\star}(f)\right)
$$

$$
	\text{Equivalent cap used in implementation: }\quad
k\le \left\lfloor \frac{B}{c_m} \right\rfloor\ (c_m>0),\quad
f\le \frac{1}{n}\left\lfloor \frac{B}{c_m} \right\rfloor
$$
