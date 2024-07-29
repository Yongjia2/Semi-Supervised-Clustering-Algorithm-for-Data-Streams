import pandas as pd
import os
import numpy as np
from scipy import linalg


def auxfunc(sample, data, nrecord, min_dis_sample, w_record): # 已加入权重w
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
    #iact_ff = iact.T
    # with open('./results/test.txt', 'a') as file:
    #     file.write(f'\t auxfunc_f,\t sample\t {f_auxfunc}\t{sample[0]}\t {sample[1]}\t {sample[2]}\t {sample[3]}\n')
    # with open('./results/test.txt', 'a') as file:
    #     file.write(
    #             f'\t auxfunc_mindis\t {min_dis_sample[0]}\t {min_dis_sample[1]}\t {min_dis_sample[2]}\t '
    #             f'{min_dis_sample[3]}\n')
    # with open('./results/test.txt', 'a') as file:
    #     file.write(
    #             f'\t auxfunc_dist1\t {dist1[0]}\t {dist1[1]}\t {dist1[2]}\t {dist1[3]}\t\n')
    # with open('./results/test.txt', 'a') as file:
    #     file.write(f'\t auxfunc_iact\t sample\t {iact_ff[0,0:40]}\n')
    return f_auxfunc, iact


def func(sample, data, nrecord, nc, w_record): # 已加入权重w
    #print('ns=2时func')
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


def fv(ns, sample, data, min_dis_sample, nrecord, nc, w_record):  # 已加入权重w
    iact = np.zeros((nrecord, 1))
    lact3 = np.zeros((nrecord, 1))
    f2 = 0
    if ns == 1:
        f2, iact = auxfunc(sample, data, nrecord, min_dis_sample, w_record)
    if ns == 2:
        f2, lact3 = func(sample, data, nrecord, nc, w_record)

    return f2, iact, lact3


def auxgrad(sample, nrecord, iact, data, w_record): # 已加入权重w
    mf = np.size(data, 1)
    grad = np.zeros((mf, ))
    # with open('./results/test.txt', 'a') as file:
    #     file.write(f'\t grad\tsample\t {grad[0]}\t {grad[1]}\t {grad[2]}\t {sample[0]}\t {sample[1]}\t {sample[2]}\n')
    # iact_f = iact.T
    # with open('./results/test.txt', 'a') as file:
    #     file.write(f'\t auxgrad_iact\t {iact_f[0,0:80]}\t\n')
    # with open('./results/test.txt', 'a') as file:
    #     file.write(f'\t auxgrad_iact\t {iact_f[0,120:180]}\t\n')
    for i in range(0, nrecord):
        if iact[i] == 2:
            for j in range(0, mf):
                grad[j] = grad[j] + 2.0e+00 * (sample[j]-data[i, j]) * w_record[i]

    return grad


def funcgrad(sample, nrecord, data, lact3, m, w_record): # 已加入权重w
    #print('ns=2时funcgrad')
    mf = np.size(data, 1)
    grad = np.zeros((m, ))
    for i in range(nrecord):
        j1 = lact3[i]
        for j in range(mf):
            k1 = int(j1 * mf + j)
            grad[k1] = grad[k1] + 2.00e+00 * (sample[k1] - data[i, j]) * w_record[i]

    return grad


def dgrad(x, sl, g, iact, lact3, data, nrecord, ns, m, w_record): # 已加入权重w
    x1 = np.empty_like(x)
    for k in range(0, m):
        x1[k] = x[k] + sl * g[k]
    grad = np.zeros((m, ))
    if ns == 1:
        grad = auxgrad(x1, nrecord, iact, data, w_record)
    if ns == 2:
        grad = funcgrad(x1, nrecord, data, lact3, m, w_record)

    return grad


def armijo(sample, data, min_dis_sample, nrecord, g, f1, f4, sl, ns, r, nc, m, w_record):  # 已加入权重w
    #print('armijo')
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
    # with open('./results/test.txt', 'a') as file:
    #    file.write(f'\t armijo_f5\t sample\t {f5}\n')

    return f5, step, iact, lact3


def equations(n, a): # 不用加入权重w
    b = a[0:n, 0:n]
    e = np.ones((n, 1))
    z1 = linalg.solve(b, e)
    z1 = z1/(z1.sum())

    return z1


def wolfe(ndg, prod, kmin, ns):  # 不用加入权重w
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
            # if ns == 2:
            #     with open('./results/test.txt', 'a') as file:
            #         file.write(f'\t wolfer_j9\t {j9}\t\n')
            if j9 > jmax:
                outer_break_flag = True
                break
            z1 = equations(jvertex, a)
            # if ns == 2:
            #     with open('./results/test.txt', 'a') as file:
            #         file.write(f'\t wolfer_z1\t {z1[0]}\t\n')
            while True:
                if (z1[0:jvertex] <= 1.0e-10).any():
                    # if ns == 2:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'\t wolfer_break1\t \n')
                    break
                else:
                    z[0:jvertex] = z1[0:jvertex]
                    # if ns == 2:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'\t wolfer_break_z1\t {z[0]}\t\n')
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
            # if ns == 2:
            #     with open('./results/test.txt', 'a') as file:
            #         file.write(f'\t wolfer_jvertex\t {jvertex}\t\n')

        if outer_break_flag:
            break
    return z, jvertex, ij


def dgm(data, m, sample, min_dis_sample, ns, nc, w_record, nblocks): # 已加入权重w
    # if nblocks == 2:
    #     with open('./results/test.txt', 'a') as file:
    #         file.write(f' dgm_sample_input: {sample} \t \r\n')
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

    # if ns == 2 and nc == 3:
    #     with open('./results/test.txt', 'a') as file:
    #         file.write(f'\t dem_sample_input\tg\t {sample} \n')

    f2, iact, lact3 = fv(ns, sample, data, min_dis_sample, nrecord, nc, w_record)
    # if nblocks == 2:
    #     with open('./results/test.txt', 'a') as file:
    #         file.write(f' dgm_f2: {f2} \t \r\n')

    while True:
        sl = div * sl
        if sl < slmin:
            break
        for i in range(m):
            g[i] = 1.0e+00 / np.sqrt(m)
        nnew = 0
        #print('flag')

        outer_break_flag = False
        while True:
            niter = niter + 1
            #print('niter:', niter)
            if (niter > maxiter):
                outer_break_flag = True
                break
            nnew = nnew + 1
            f1 = f2
            #print('f1:', f1)
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
                # if nblocks == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f'\t dem_ndg\tg\t {ndg}\t {sl}\t {sample[0]}\t {sample[1]}\t {g[0]}\t {g[1]}\t {g[2]} \n')
                # with open('./results/test.txt', 'a') as file:
                #     file.write(f'\t dem_ndg\tsl\tx\t {ndg}\t {sl}\t {sample[0]}\t {sample[1]}\t {sample[2]} '
                #                f'{sample[3]}\n')
                v = dgrad(sample, sl, g, iact, lact3, data, nrecord, ns, m, w_record)
                # if nblocks == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f' dgm_v: {v} \t \r\n')

                dotprod = 0.0e+00
                for i in range(0, m):
                    dotprod = dotprod + v[i] * v[i]
                r = np.sqrt(dotprod)
                # if nblocks == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f' dgm_r: {r} \t \r\n')

                #if ndg == 2:
                    #print('v', v)
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
                # if nblocks == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f' dgm_toler: {toler} \t \r\n')

                # if ns == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f'\t dem_toler\t {toler}\n')

                if ndg > 1:
                    for i in range(0, ndg-1):
                        prod[ndg-1, i] = np.dot(ww[i, :], v)
                        prod[i, ndg-1] = prod[ndg-1, i]
                prod[ndg-1, ndg-1] = dotprod

# ================================================================
                for i in range(0, m):
                    ww[ndg-1, i] = v[i]

                z, jvertex, ij = wolfe(ndg, prod, kmin, ns)

                # if ns == 1 and nc == 5:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f'\t dem_z\t {z[0]}\t {z[1]}\t {z[2]}\t\n')
                for i in range(0, m):
                    v[i] = 0.0
                    for j in range(0, jvertex):
                        v[i] = v[i] + ww[ij[j]-1, i] * z[j]
                #print('v', v)
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
                #if ndg == 1:
                    #print('testfv')
                f3 = (f4 - f1) / sl
                decreas = step0 * r
                if f3 < decreas:
                    # if ns == 1 and nc == 5:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'\t run armijo\t f3=\t {f3} \n')
                    f5, step, iact, lact3 = armijo(sample, data, min_dis_sample, nrecord, g, f1, f4, sl, ns, r, nc, m, w_record)
                    # if ns == 1 and nc == 5:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'\t dgm_f5\t {f5}\n')
                    f2 = f5
                    #print('f2:', f2, ndg, jvertex)
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

    #print('dgm', sample, f2)
    return sample, f2



