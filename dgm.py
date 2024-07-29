import pandas as pd
import os
import numpy as np
from scipy import linalg


def auxfunc(sample, data, nrecord, min_dis_sample, w_record): 
    iact = np.zeros((nrecord, 1))
    dist1 = np.array([np.linalg.norm(sample - tmp2) ** 2 for tmp2 in data])
    f_auxfunc = 0
    for i in range(0, nrecord):
        if dist1[i] > min_dis_sample[i]:
            iact[i] = 1
            f_auxfunc += min_dis_sample[i]*w_record[i]
        else:
            iact[i] = 2
            f_auxfunc += dist1[i]*w_record[i]

    return f_auxfunc, iact


def func(sample, data, nrecord, nc, w_record): 
    mf = np.size(data, 1)
    lact3 = np.zeros((nrecord, 1))
    dist2 = np.zeros((nrecord, nc))
    for i in range(0, nrecord):
        for j in range(0, nc):
            dist2[i, j] = np.linalg.norm(sample[j*mf:(j+1)*mf].T - data[i, :])**2
    f = 0
    for i in range(0, nrecord):
        index, f4 = np.argmin(dist2[i, :]), min(dist2[i, :])
        lact3[i] = index
        f = f + f4*w_record[i]

    return f, lact3


def fv(ns, sample, data, min_dis_sample, nrecord, nc, w_record):  
    iact = np.zeros((nrecord, 1))
    lact3 = np.zeros((nrecord, 1))
    f2 = 0
    if ns == 1:
        f2, iact = auxfunc(sample, data, nrecord, min_dis_sample, w_record)
    if ns == 2:
        f2, lact3 = func(sample, data, nrecord, nc, w_record)

    return f2, iact, lact3


def auxgrad(sample, nrecord, iact, data, w_record):
    mf = np.size(data, 1)
    grad = np.zeros((mf, ))
    for i in range(0, nrecord):
        if iact[i] == 2:
            for j in range(0, mf):
                grad[j] = grad[j] + 2.0e+00 * (sample[j]-data[i, j]) * w_record[i]

    return grad

def funcgrad(sample, nrecord, data, lact3, m, w_record): 
    #print('ns=2时funcgrad')
    mf = np.size(data, 1)
    grad = np.zeros((m, ))
    for i in range(nrecord):
        j1 = lact3[i]
        for j in range(mf):
            k1 = int(j1 * mf + j)
            grad[k1] = grad[k1] + 2.00e+00 * (sample[k1] - data[i, j]) * w_record[i]

    return grad


def dgrad(x, sl, g, iact, lact3, data, nrecord, ns, m, w_record): 
    x1 = np.empty_like(x)
    for k in range(0, m):
        x1[k] = x[k] + sl * g[k]
    grad = np.zeros((m, ))
    if ns == 1:
        grad = auxgrad(x1, nrecord, iact, data, w_record)
    if ns == 2:
        grad = funcgrad(x1, nrecord, data, lact3, m, w_record)

    return grad


def armijo(sample, data, min_dis_sample, nrecord, g, f1, f4, sl, ns, r, nc, m, w_record):  
    step = sl
    f5 = f4
    x1 = np.zeros((m,))
    while True:
        step = 2.0e+00 * step
        for i in range(0, m):
            x1[i] = sample[i] + step*g[i]
        f6, iact, lact3 = fv(ns, x1, data, min_dis_sample, nrecord, nc, w_record)
        f3 = f6 - f1 + 1.0e-02 * step * r
        if f3 > 0.0e+00:
            # with open('./results/test.txt', 'a') as file:
            #     file.write(f'\t armijo_return_f6,f5\t sample\t {f6}\t{f5}\n')
            step = step/2.0e+00
            break
        f5 = f6

    return f5, step, iact, lact3


def equations(n, a):
    b = a[0:n, 0:n]
    e = np.ones((n, 1))
    z1 = linalg.solve(b, e)
    z1 = z1/(z1.sum())

    return z1


def wolfe(ndg, prod, kmin, ns): 
    ij = np.zeros((10000, 1), dtype=int)
    ij[0] = kmin
    z = np.zeros((10000, 1))
    j9 = 0
    jmax = 500*ndg
    jvertex = 1
    z[0] = 1.0e+00

    while True:
        r = 0.0e+00
        for i in range(0, jvertex):
            for j in range(0, jvertex):
                r = r + z[i] * z[j] * prod[ij[i]-1, ij[j]-1]

        outer_break_flag = False
        if ndg == 1:
            break
# ================================================================
        t0 = 1.0e+12
        for i in range(0, ndg):
            #t1 = np.dot(z[0:jvertex], prod[ij[0:jvertex], k])
            t1 = 0.0e+00
            for j in range(0, jvertex):
                t1 = t1 + z[j]*prod[ij[j]-1, i]
            if t1 < t0:
                t0 = t1
                kmax = i
# ================================================================
# First stopping criterion
# ================================================================
        rm = prod[kmax, kmax]
        #print('rm:', rm)
        for i in range(0, jvertex):
            rm = max(rm, prod[ij[i]-1, ij[i]-1])
        r2 = r-1.0e-12 * rm
        if t0 > r2:
            break
# ================================================================
# Second stopping criterion
# ================================================================
        if (kmax in ij[0:jvertex]-1):
            break

# ================================================================
# Step 1(e) from Wolfe's algorithm
# ================================================================
        jvertex = jvertex + 1
        ij[jvertex-1] = kmax + 1
        z[jvertex-1] = 0.0e00
# ================================================================
        break_flag = False
        a = np.empty_like(prod)
        while True:
            for i in range(0, jvertex):
                for j in range(0, jvertex):
                    a[i, j] = 1.0e00 + prod[ij[i]-1, ij[j]-1]
            j9 = j9 + 1
            if j9 > jmax:
                outer_break_flag = True
                break
            z1 = equations(jvertex, a)
            while True:
                if (z1[0:jvertex] <= 1.0e-10).any():
                    break
                else:
                    z[0:jvertex] = z1[0:jvertex]
                    break_flag = True
                    break
            if break_flag:
                break

            teta = 1.0e+00
            for i in range(0, jvertex):
                z5 = z[i] - z1[i]
                if z5 > 1.0e-10:
                    teta = min(teta, z[i]/z5)
            for i in range(0, jvertex):
                z[i] = (1.0e+00 - teta) * z[i] + teta*z1[i]
                if z[i] <= 1.0e-10:
                    z[i] = 0.0e+00
                    kzero = i
            j2 = 0
            for i in range(0, jvertex):
                if i != kzero:
                    j2 = j2 + 1
                    ij[j2-1] = ij[i]
                    z[j2-1] = z[i]
            jvertex = j2

        if outer_break_flag:
            break
    return z, jvertex, ij


def dgm(data, m, sample, min_dis_sample, ns, nc, w_record): 
    nrecord = np.size(data, 0)
    div = 1.0e-1
    slinit = 1.0e0
    slmin = 1.0e-5 * slinit
    nbundle = min(m+3, 40)
    g = np.zeros((m, 1))
    niter = 0
    maxiter = 5000
    fvalues = np.empty(maxiter)
    mturn = 4
    sdif = 1.0e-5
    eps0 = 1.0e-07
    dist1 = 1.0e-07
    prod = np.zeros((1000, 1000))
    x1 = np.empty_like(sample)
    step0 = -5.0e-02
    ww = np.zeros((1000, m))
    sl = slinit/div

    f2, iact, lact3 = fv(ns, sample, data, min_dis_sample, nrecord, nc, w_record)

    while True:
        sl = div * sl
        if sl < slmin:
            break
        for i in range(m):
            g[i] = 1.0e+00 / np.sqrt(m)
        nnew = 0

        outer_break_flag = False
        while True:
            niter = niter + 1
            if (niter > maxiter):
                outer_break_flag = True
                break
            nnew = nnew + 1
            f1 = f2
            fvalues[niter-1] = f1
# ---------------------------------------------------------------
            if (nnew > mturn):
                mturn2 = niter-mturn+1
                ratio1 = (fvalues[mturn2 - 1]-f1)/(abs(f1)+1.0e+00)
                if (ratio1 < sdif):
                    break
            if (nnew >= (2*mturn)):
                mturn2 = niter-2*mturn+1
                ratio1 = (fvalues[mturn2-1]-f1)/(abs(f1)+1.0e+00)
                if (ratio1 < (1.0e-01*sdif)):
                    break
# ---------------------------------------------------------------
            break_flag = False
            for ndg in range(1, nbundle+1):
                v = dgrad(sample, sl, g, iact, lact3, data, nrecord, ns, m, w_record)
                dotprod = 0.0e+00
                for i in range(0, m):
                    dotprod = dotprod + v[i] * v[i]
                r = np.sqrt(dotprod)
                if (r < eps0):
                    break_flag = True
                    break
                if ndg == 1:
                    rmean = r
                    kmin = 1
                    rmin = r
                if ndg > 1:
                    rmin = min(rmin, r)
                    if r == rmin:
                        kmin = ndg
                    rmean = ((ndg-1)*rmean + r)/ndg
                toler = max(eps0, dist1*rmean)

                if ndg > 1:
                    for i in range(0, ndg-1):
                        prod[ndg-1, i] = np.dot(ww[i, :], v)
                        prod[i, ndg-1] = prod[ndg-1, i]
                prod[ndg-1, ndg-1] = dotprod

# ================================================================
                for i in range(0, m):
                    ww[ndg-1, i] = v[i]

                z, jvertex, ij = wolfe(ndg, prod, kmin, ns)

                for i in range(0, m):
                    v[i] = 0.0
                    for j in range(0, jvertex):
                        v[i] = v[i] + ww[ij[j]-1, i] * z[j]
                r = 0
                for i in range(0, m):
                    r = r + v[i] * v[i]
                r = np.sqrt(r)
                if r < toler:
                    break_flag = True
                    break

                for i in range(0, m):
                    g[i] = -v[i]/r
                    x1[i] = sample[i] + sl * g[i]

                f4, iact, lact3 = fv(ns, x1, data, min_dis_sample, nrecord, nc, w_record)
                f3 = (f4 - f1) / sl
                decreas = step0 * r
                if f3 < decreas:
                    f5, step, iact, lact3 = armijo(sample, data, min_dis_sample, nrecord, g, f1, f4, sl, ns, r, nc, m, w_record)
                    f2 = f5
                    for i in range(0, m):
                        sample[i] = sample[i] + step * g[i]
                    if ndg <= 2:
                        sl = 1.1e+00 * sl
                    break

                if ndg == nbundle:
                    break_flag = True
                    break

            if break_flag:
                break
        if outer_break_flag:
            break

    return sample, f2



