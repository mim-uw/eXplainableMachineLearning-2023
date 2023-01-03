## Task 1

Consider a following model:

f(x1, x2) = (x1 + x2)^2

Assume that x1, x2 ~ U[-1,1] and x1=x2 (full dependency)

Calculate PD profile for variable x1 in this model.

g_pd_1(z) = E_X[f(z, X)] = E_X[z^2 + 2 * z * X + X^2] = z^2 + 2* z * E_X[X] + E_X[X^2] = z^2 + 1/3