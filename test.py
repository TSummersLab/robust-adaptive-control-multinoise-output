# combined estimator and controller for multiplicative noise LQG


import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt

from numpy import zeros, ones, diag
from numpy.random import randn
from numpy.linalg import eig


from utility.matrixmath import specrad


seed = 1
npr.seed(seed)

# problem data
n = 8
m = 4
p = 2

n2 = n**2

# nominal system matrices
A = randn(n, n)
A *= 0.8*specrad(A)
B = randn(n, m)
C = randn(p, n)



# # multiplicative noise variances
# SigmaA = randn(n2, n2)
# SigmaA = SigmaA*SigmaA.T
# SigmaA = 0.1*SigmaA/max(eig(SigmaA))
# [Va, Ea] = eig(SigmaA)
# a = diag(Ea)
# Aa = zeros(n, n, n2)
# for i=1:n2
# Aa(:,:, i) = reshape(Va(:, i), n, n)
# end
#
# SigmaB = randn(n*m, n*m)
# SigmaB = SigmaB*SigmaB.T
# SigmaB = 0.1*SigmaB/max(eig(SigmaB))
# [Vb, Eb] = eig(SigmaB)
# b = diag(Eb)
# Bb = zeros(n, m, n*m)
# for i=1:n*m
# Bb(:,:, i) = reshape(Vb(:, i), n, m)
# end
#
# SigmaC = randn(n*p, n*p)
# SigmaC = SigmaC*SigmaC
# '; SigmaC = 0.1*SigmaC/max(eig(SigmaC));
# [Vc, Ec] = eig(SigmaC)
# c = diag(Ec)
# Cc = zeros(p, n, n*p)
# for i=1:n*p
# Cc(:,:, i) = reshape(Vc(:, i), p, n)
# end
#
# # cost matrices
# Q = eye(n)
# R = eye(m)
#
# # noise variances (note: following de Koning, V is process noise variance,
# # W is measurement noise variance)
# V = 1*eye(n)
# W = 1*eye(p)
#
# # de Koning example 2
# # n = 2; m = 1; p = 1;
# # A = [0.7092 0.3017; 0.1814 0.9525];
# # B = [0.7001; 0.1593];
# # C = [0.3088 0.5735];
# #
# # beta = 0.2;
# # a = beta; Aa = A;
# # b = beta; Bb = B;
# # c = beta; Cc = C;
# #
# # Q = diag([0.7350 0.9820]);
# # R = 0.6644;
# # V = diag([0.5627 0.7357]);
# # W = 0.2588;
#
#
# # recursion (from De Koning, TAC 1992)
# # initialize
# X1 = zeros(n)
# X2 = zeros(n)
# X3 = zeros(n)
# X4 = zeros(n)
#
# # note: following de Koning, K is estimator gain, L is controller gain
# K = A*X3*C'*pinv(W + C*X3*C' + MultSum2(X3, c, Cc) + MultSum2(X4, c, Cc));
# L = pinv(R + B'*X1*B + MultSum1(X1, b, Bb) + MultSum1(X2, b, Bb))*B'*X1*A;
# F = A - B*L - K*C;
#
# Xprev = randn(n, 4*n);
#
# while (norm([X1 X2 X3 X4] - Xprev, 'fro') > 1e-8)
#     Xprev = [X1 X2 X3 X4];
#
#     X1 = A
#     '*X1*A + MultSum1(X1, a, Aa) - L'*(
#     R + B'*X1*B + MultSum1(X1, b, Bb) + MultSum1(X2, b, Bb))*L + Q + MultSum1(X2, a, Aa) + MultSum1(K'*X2*K, c, Cc);
#     X2 = (A - K*C)
#     '*X2*(A - K*C) + L'*(R + B'*X1*B + MultSum1(X1, b, Bb) + MultSum1(X2, b, Bb))*L;
#                          X3 = A*X3*A' + MultSum2(X3, a, Aa) - K*(W + C*X3*C' + MultSum2(X3, c, Cc) + MultSum2(X4, c,
#                                                                                                               Cc))*K
#     ' + V + MultSum2(X4, a, Aa) + MultSum2(L*X4*L', b, Bb);
#     X4 = (A - B*L)*X4*(A - B*L)
#     ' + K*(W + C*X3*C' + MultSum2(X3, c, Cc) + MultSum2(X4, c, Cc))*K
#     ';
#
#     K = A*X3*C'*pinv(W + C*X3*C' + MultSum2(X3, c, Cc) + MultSum2(X4, c, Cc));
#     L = pinv(R + B'*X1*B + MultSum1(X1, b, Bb) + MultSum1(X2, b, Bb))*B'*X1*A;
#     F = A - B*L - K*C;
#
#     #     norm([X1 X2 X3 X4] - Xprev, 'fro')
#     end
#
#     Jstar1 = trace(Q*X3 + (Q + L'*R*L)*X4);
# Jstar2 = trace(V*X1 + (V + K*W*K')*X2);
#
#
# # TODO: check solution (also compare with special case of Bernstein + Haddad IJC
# # 1987)
#
# # TODO: compare to separated solution, find an example where separated
# # solution is unstable?
#
# ## simulate optimal controller
# x0 = 50*randn(n, 1)
# xhat0 = 50*randn(n, 1)
# T = 50
# x = zeros(n, T)
# xhat = zeros(n, T)
# u = zeros(m, T)
# y = zeros(p, T)
# x(:, 1) = x0
# xhat(:, 1) = xhat0
#
# for t = 1:T
# Ar = reshape(mvnrnd(zeros(n ^ 2, 1), SigmaA, 1)
# ', n, n);
# Br = reshape(mvnrnd(zeros(n*m, 1), SigmaB, 1)
# ', n, m);
# Cr = reshape(mvnrnd(zeros(n*p, 1), SigmaC, 1)
# ', p, n);
# v = mvnrnd(zeros(n, 1), V, 1)
# ';
# w = mvnrnd(zeros(p, 1), W, 1)
# ';
#
# u(:, t) = -L*xhat(:, t);
# x(:, t + 1) = (A + Ar)*x(:, t) + (B + Br)*u(:, t) + v
# y(:, t) = (C + Cr)*x(:, t) + w
# xhat(:, t + 1) = F*xhat(:, t) + K*y(:, t)
# end
#
# stairs(x
# ');
# xlabel('time')
# ylabel('states')
#
# function
# Z = MultSum1(X, z, Zz)
# Z = zeros(size(Zz, 2))
# for i = 1:length(z)
# Z = Z + z(i)*Zz(:,:, i)'*X*Zz(:,:,i)
# end
# end
#
# function
# Z = MultSum2(X, z, Zz)
# Z = zeros(size(Zz, 1))
# for i = 1:length(z)
# Z = Z + z(i)*Zz(:,:, i)*X*Zz(:,:, i)'
# end
# end