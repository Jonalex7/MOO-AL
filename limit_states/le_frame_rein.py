# le_frame.py
#
# This is an implementation of the linear elastic frame structure also analysed in Blatman and
# Sudret (2010). It is a 3-bay 5-storey building subjected to horizontal loads. The main function
# le_frame accepts a list of 21 random variables. If the parameter std_normal is set to True, then a
# list of 21 standard normally distributed random variables is accepted. In the latter case the list
# may be truncated to a minimum of 13 values (exploiting a useful eigendecomposition property). The
# return value of the function is a list of 5 values with the displacements of the left-most node of
# each storey. A limit state function may be expressed using one (or multiple) of the displacements
# and a certain threshold value. Two of the cases considered in Blatman and Sudret (2010) are:
#
#  Case  Limit state function  P_f                beta
#  ----  --------------------  -----------------  -----------
#  1     5.0*0.0328084 - u[4]  1.35e-3 (1.54e-3)  3.00 (2.96)
#  2     7.0*0.0328084 - u[4]  1.21e-4 (3.75e-5)  3.67 (3.96)
#
# Values in parentheses are taken from Blatman and Sudret (2010), the other values have been
# calculated using this implementation. The problem appears to be very sensitive to the way the
# truncated distributions are dealt with -- which may have caused the different outcomes. A special
# thanks goes to Jean-Marc Bourinet for providing the MATLAB source code on which this one is based.
#
# Blatman, G. and Sudret, B. (2010). An adaptive algorithm to build up sparse polynomial chaos
#     expansions for stochastic finite element analysis. Probabilistic Engineering Mechanics,
#     Volume 25, pp. 183-197.
#
# This file is part of the Reliability Problems repository (https://rprepo.readthedocs.io). It is
# not meant to be used in any other context. The MIT License applies, see below.
#
# Author: Rein de Vries
# Date: 11 March 2019
#
#
# The MIT License (MIT)
#
# Copyright (c) 2019 TNO, Delft, The Netherlands
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from math import exp, sqrt, log
import numpy as np
from scipy.stats import norm


####################################################################################################
# Generic functions

def sq(x):
    return x * x


def ln(x):
    return log(x)


def ln_ms(mean, stddev):
    x = sqrt(sq(mean) + sq(stddev))
    mu = ln(sq(mean) / x)
    sigma = sqrt(2.0 * ln(x / mean))
    return mu, sigma


def index_from_id(arr, id):
    for i in range(0, len(arr)):
        if arr[i].id == id:
            return i
    return -1


####################################################################################################
# Global variables

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class Element:
    def __init__(self, id, i_E, i_IA, node_id_1, node_id_2, i_node_1=-1, i_node_2=-1):
        self.id = id
        self.i_E = i_E
        self.i_IA = i_IA
        self.node_id_1 = node_id_1
        self.node_id_2 = node_id_2
        self.i_node_1 = i_node_1
        self.i_node_2 = i_node_2


nodes = []
elements = []

mu_lnP = [0.0] * 3;
sigma_lnP = [0.0] * 3
mu_E = [0.0] * 2;
sigma_E = [0.0] * 2;
F_min_E = [0.0] * 2
mu_I = [0.0] * 8;
sigma_I = [0.0] * 8;
F_min_I = [0.0] * 8
mu_A = [0.0] * 8;
sigma_A = [0.0] * 8;
F_min_A = [0.0] * 8


####################################################################################################
# Initialization

def init():
    global nodes, elements
    global mu_lnP, sigma_lnP
    global mu_E, sigma_E, F_min_E
    global mu_I, sigma_I, F_min_I
    global mu_A, sigma_A, F_min_A
    global L

    # Input parameters
    nodes = [
        Node(1, 0.0, 0.0),
        Node(2, 25.0, 0.0),
        Node(3, 55.0, 0.0),
        Node(4, 80.0, 0.0),
        Node(11, 0.0, 16.0),
        Node(12, 25.0, 16.0),
        Node(13, 55.0, 16.0),
        Node(14, 80.0, 16.0),
        Node(21, 0.0, 28.0),
        Node(22, 25.0, 28.0),
        Node(23, 55.0, 28.0),
        Node(24, 80.0, 28.0),
        Node(31, 0.0, 40.0),
        Node(32, 25.0, 40.0),
        Node(33, 55.0, 40.0),
        Node(34, 80.0, 40.0),
        Node(41, 0.0, 52.0),
        Node(42, 25.0, 52.0),
        Node(43, 55.0, 52.0),
        Node(44, 80.0, 52.0),
        Node(51, 0.0, 64.0),
        Node(52, 25.0, 64.0),
        Node(53, 55.0, 64.0),
        Node(54, 80.0, 64.0)
    ]

    elements = [
        Element(1, 1, 2, 1, 11),
        Element(2, 1, 3, 2, 12),
        Element(3, 1, 3, 3, 13),
        Element(4, 1, 2, 4, 14),
        Element(11, 1, 1, 11, 21),
        Element(12, 1, 2, 12, 22),
        Element(13, 1, 2, 13, 23),
        Element(14, 1, 1, 14, 24),
        Element(21, 1, 1, 21, 31),
        Element(22, 1, 2, 22, 32),
        Element(23, 1, 2, 23, 33),
        Element(24, 1, 1, 24, 34),
        Element(31, 1, 0, 31, 41),
        Element(32, 1, 1, 32, 42),
        Element(33, 1, 1, 33, 43),
        Element(34, 1, 0, 34, 44),
        Element(41, 1, 0, 41, 51),
        Element(42, 1, 1, 42, 52),
        Element(43, 1, 1, 43, 53),
        Element(44, 1, 0, 44, 54),
        Element(15, 0, 6, 11, 12),
        Element(16, 0, 7, 12, 13),
        Element(17, 0, 6, 13, 14),
        Element(25, 0, 5, 21, 22),
        Element(26, 0, 6, 22, 23),
        Element(27, 0, 5, 23, 24),
        Element(35, 0, 5, 31, 32),
        Element(36, 0, 6, 32, 33),
        Element(37, 0, 5, 33, 34),
        Element(45, 0, 4, 41, 42),
        Element(46, 0, 5, 42, 43),
        Element(47, 0, 4, 43, 44),
        Element(55, 0, 4, 51, 52),
        Element(56, 0, 5, 52, 53),
        Element(57, 0, 4, 53, 54)
    ]

    for element in elements:
        element.i_node_1 = index_from_id(nodes, element.node_id_1)
        element.i_node_2 = index_from_id(nodes, element.node_id_2)

    mu_lnP[0], sigma_lnP[0] = ln_ms(30.0, 9.0)  # P_1
    mu_lnP[1], sigma_lnP[1] = ln_ms(20.0, 8.0)  # P_2
    mu_lnP[2], sigma_lnP[2] = ln_ms(16.0, 6.4)  # P_3

    mu_E[0] = 454e3;
    sigma_E[0] = 40e3  # E_4
    mu_E[1] = 497e3;
    sigma_E[1] = 40e3  # E_5

    mu_I[0] = 0.94;
    sigma_I[0] = 0.12  # I_6
    mu_I[1] = 1.33;
    sigma_I[1] = 0.15  # I_7
    mu_I[2] = 2.47;
    sigma_I[2] = 0.30  # I_8
    mu_I[3] = 3.00;
    sigma_I[3] = 0.35  # I_9
    mu_I[4] = 1.25;
    sigma_I[4] = 0.30  # I_10
    mu_I[5] = 1.63;
    sigma_I[5] = 0.40  # I_11
    mu_I[6] = 2.69;
    sigma_I[6] = 0.65  # I_12
    mu_I[7] = 3.00;
    sigma_I[7] = 0.75  # I_13

    mu_A[0] = 3.36;
    sigma_A[0] = 0.60  # A_14
    mu_A[1] = 4.00;
    sigma_A[1] = 0.80  # A_15
    mu_A[2] = 5.44;
    sigma_A[2] = 1.00  # A_16
    mu_A[3] = 6.00;
    sigma_A[3] = 1.20  # A_17
    mu_A[4] = 2.72;
    sigma_A[4] = 1.00  # A_18
    mu_A[5] = 3.13;
    sigma_A[5] = 1.10  # A_19
    mu_A[6] = 4.01;
    sigma_A[6] = 1.30  # A_20
    mu_A[7] = 4.50;
    sigma_A[7] = 1.50  # A_21

    # Calculate minimum quantiles for x = 0 truncation

    F_min_E = [norm.cdf(0.0, mu_E[i], sigma_E[i]) for i in range(0, 2)]
    F_min_I = [norm.cdf(0.0, mu_I[i], sigma_I[i]) for i in range(0, 8)]
    F_min_A = [norm.cdf(0.0, mu_A[i], sigma_A[i]) for i in range(0, 8)]

    # Correlation matrix

    R = np.zeros((21, 21))

    for i in range(3, 5):
        for j in range(3, 5):
            R[i][j] = 0.9  # E_4 and E_5

    for i in range(5, 13):
        for j in range(5, 13):
            R[i][j] = 0.13  # I_6 to I_13

    for i in range(13, 21):
        for j in range(13, 21):
            R[i][j] = 0.13  # A_14 to A_21

    for i in range(5, 13):
        for j in range(13, 21):
            R[i][j] = R[j][i] = 0.13  # A_6 to A_13 with I_14 to I_21

    for i in range(5, 13):
        R[i + 8][i] = R[i][i + 8] = 0.95  # A_6 to A_13 with I_6 to I_13

    for i in range(0, 21):
        R[i][i] = 1.0

    # Eigendecomposition
    # mod for Hermitian (symmetric if real-valued) matrices
    # Instead of np.linalg.eig, which is general-purpose: np.linalg.eigh
    w, V = np.linalg.eigh(R)

    L = np.matmul(V, np.diag(np.sqrt(w)))
    
    # Put independent RVs in front again
    indexes = w.argsort()[::-1]

    for i in range(0, 3):
        indexes = np.delete(indexes, np.argwhere(indexes == i))
        indexes = np.insert(indexes, i, i)

    L = L[:, indexes]


####################################################################################################
# Calculate displacements given a random vector

def le_frame(x, std_normal=False):
    global nodes, elements
    global mu_lnP, sigma_lnP
    global mu_E, sigma_E, F_min_E
    global mu_I, sigma_I, F_min_I
    global mu_A, sigma_A, F_min_A
    global L

    # Initialization of global variables (done once)

    if len(nodes) == 0:
        init()

    # Realizations of parameters; fill vectors P, E, I and A

    if std_normal:
        # prepare vector xi_corr containing the correlated std normals

        if len(x) < 13 or len(x) > 21:
            return []

        xi = np.zeros(21)
        for i in range(0, len(x)):
            xi[i] = x[i]

        xi_corr = np.matmul(L, xi)

        # apply transformation

        P = [exp(mu_lnP[i] + sigma_lnP[i] * xi_corr[i]) for i in range(0, 3)]
        # print(P)
        E = [0.0] * 2
        
        for i in range(0, len(E)):
            F = norm.cdf(xi_corr[3 + i])
            # F = np.real(F)
            # print(type(F_min_E[i]), type(F), type(mu_E[i]), type(sigma_E[i]))
            if F == 1.0:
                E[i] = mu_E[i] + 10.0 * sigma_E[i]
            else:
                E[i] = norm.ppf(F_min_E[i] + F * (1.0 - F_min_E[i]), mu_E[i], sigma_E[i])

        I = [0.0] * 8
        for i in range(0, len(I)):
            F = norm.cdf(xi_corr[5 + i])
            # F = np.real(F)
            if F == 1.0:
                I[i] = mu_I[i] + 10.0 * sigma_I[i]
            else:
                I[i] = norm.ppf(F_min_I[i] + F * (1.0 - F_min_I[i]), mu_I[i], sigma_I[i])

        A = [0.0] * 8
        for i in range(0, len(A)):
            F = norm.cdf(xi_corr[13 + i])
            # F = np.real(F)
            if F == 1.0:
                A[i] = mu_A[i] + 10.0 * sigma_A[i]
            else:
                A[i] = norm.ppf(F_min_A[i] + F * (1.0 - F_min_A[i]), mu_A[i], sigma_A[i])

    else:
        # assign directly

        P = [x[i] for i in range(0, 3)]
        E = [x[i + 3] for i in range(0, 2)]
        I = [x[i + 5] for i in range(0, 8)]
        A = [x[i + 13] for i in range(0, 8)]

        # check bounds

        for i in range(0, 3):
            if P[i] < 0.0:
                return []

        for i in range(0, 2):
            if E[i] < 0.0:
                return []

        for i in range(0, 8):
            if I[i] < 0.0:
                return []
            if A[i] < 0.0:
                return []

    # Stiffness matrix

    node_count = len(nodes)
    dof = 3

    K = np.zeros((node_count * dof, node_count * dof))

    K_e = np.zeros((2 * dof, 2 * dof))
    K_et = np.zeros((2 * dof, 2 * dof))
    T = np.zeros((2 * dof, 2 * dof))

    for element in elements:
        node_1 = nodes[element.i_node_1]
        node_2 = nodes[element.i_node_2]

        L_ = sqrt(sq(node_2.x - node_1.x) + sq(node_2.y - node_1.y))
        A_ = A[element.i_IA]
        I_ = I[element.i_IA]
        E_ = E[element.i_E]

        # element stiffness matrix
        #
        # [N_1]         [u_x_1  ]
        # [V_1]         [u_y_1  ]
        # [M_1]         [theta_1]
        # [N_2] = [K_e] [u_x_2  ]
        # [V_2]         [u_y_2  ]
        # [M_2]         [theta_2]

        K_e[0][0] = K_e[3][3] = E_ * A_ / L_
        K_e[3][0] = K_e[0][3] = -K_e[0][0]

        K_e[1][1] = K_e[4][4] = 12.0 * E_ * I_ / pow(L_, 3)
        K_e[1][4] = K_e[4][1] = -K_e[1][1]

        K_e[2][2] = K_e[5][5] = 4.0 * E_ * I_ / L_

        K_e[1][2] = K_e[2][1] = K_e[1][5] = K_e[5][1] = 6.0 * E_ * I_ / sq(L_)
        K_e[2][4] = K_e[4][2] = K_e[4][5] = K_e[5][4] = -K_e[1][2]

        K_e[2][5] = K_e[5][2] = 2.0 * E_ * I_ / L_

        # transformation matrix

        c = (node_2.x - node_1.x) / L_;
        s = (node_2.y - node_1.y) / L_;

        T[0][0] = c;
        T[0][1] = s
        T[1][0] = -s;
        T[1][1] = c
        T[2][2] = 1.0

        T[3][3] = c;
        T[3][4] = s
        T[4][3] = -s;
        T[4][4] = c
        T[5][5] = 1.0

        K_et = np.matmul(np.matmul(T.transpose(), K_e), T)

        # assemble into global stiffness matrix

        node_indexes = [element.i_node_1, element.i_node_2]

        for i in range(0, len(node_indexes)):
            src_row = i * dof
            dst_row = node_indexes[i] * dof

            for j in range(0, len(node_indexes)):
                src_col = j * dof;
                dst_col = node_indexes[j] * dof

                for k in range(0, dof):
                    for l in range(0, dof):
                        K[dst_row + k][dst_col + l] += K_et[src_row + k][src_col + l]

    # Force vector

    f = np.zeros(node_count * dof)

    f[4 * dof] = P[2];  # node 11, x
    f[8 * dof] = P[1];  # node 21, x
    f[12 * dof] = P[0];  # node 31, x
    f[16 * dof] = P[0];  # node 41, x
    f[20 * dof] = P[0];  # node 51, x

    # Constrain bottom nodes in all dof

    for i in range(0, 4 * dof):
        K[:, i] = 0.0
        K[i, :] = 0.0
        K[i][i] = 1.0

    # solve for displacements

    u = np.linalg.solve(K, f)

    return [u[4 * dof],  # node 11, x
            u[8 * dof],  # node 21, x
            u[12 * dof],  # node 31, x
            u[16 * dof],  # node 41, x
            u[20 * dof]]  # node 51, x
