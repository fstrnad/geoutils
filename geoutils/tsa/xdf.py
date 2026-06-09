''' File description
Transformation script to reduce the effect of Auto-Correlations in correlation based analysis.
@Author  :   Moritz Haas
'''
import numpy as np
import scipy.stats as sp
import warnings
# warnings.filterwarnings('error')


def coords_to_upperidx(i, j, N):
    # convert coords of NxN square matrix to idx in flattened triu(...,k=1)
    return int((N-1)*i - i*(i+1)//2 + j - 1)


def autocorr(x, t=1):
    """ dumb autocorrelation on a 1D array,
    almost identical to matlab autocorr()"""
    x = x.copy()
    x = x - np.tile(np.mean(x), np.size(x))
    AC = np.zeros(t)
    for l in np.arange(t):
        AC[l] = np.corrcoef(np.array([x[0:len(x)-l], x[l:len(x)]]))[0, 1]
    return AC


def AC_fft(Y, T, copy=True):
    # computes the AC via the FFT. Works when zero-padding to twice the original length
    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print('AC_fft::: Input should be in IxT form, the matrix was transposed.')
        Y = np.transpose(Y)
    Y = np.array(Y)
    print("AC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y-np.transpose(np.tile(mY2, (T, 1)))

    nfft = int(nextpow2(2*T-1))  # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)  # be careful with the dimensions
    ACOV = np.real(np.fft.ifft(yfft*np.conj(yfft), axis=1))
    ACOV = ACOV[:, 0:T]

    Norm = np.sum(np.abs(Y)**2, axis=1)
    Norm = np.transpose(np.tile(Norm, (T, 1)))
    xAC = ACOV/Norm  # normalise the COVs

    bnd = (np.sqrt(2)*1.3859)/np.sqrt(T)  # assumes normality for AC
    CI = [-bnd, bnd]

    return xAC, CI


def xC_fft_flat(Y, T, is_short, shortlag, maxlag, copy=True):
    # is_short is in flat index format

    # ***********************************
    # This should be checked! There shouldn't be any complex numbers!!
    # __main__:74: ComplexWarning: Casting complex values to real discards the imaginary part
    # This is because Python, in contrast to Matlab, produce highly prcise imaginary parts
    # by defualt, when you wanna do ifft, just use np.real()
    # ***********************************
    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print('xC_fft::: Input should be in IxT form, the matrix was transposed.')
        Y = np.transpose(Y)
    Y = np.array(Y)

    I = np.shape(Y)[0]

    print("xC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y-np.transpose(np.tile(mY2, (T, 1)))

    nfft = nextpow2(2*T-1)  # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)  # be careful with the dimensions

    shortlagcc = int((shortlag-1)*2+1)
    xCshort = np.zeros([np.sum(is_short), shortlagcc])

    maxlagcc = int((maxlag-1)*2+1)
    xClong = np.zeros([len(is_short)-np.sum(is_short), maxlagcc])

    short_idx, long_idx = 0, 0

    XX = np.triu_indices(I, 1)[0]
    YY = np.triu_indices(I, 1)[1]

    for idx in np.arange(np.size(XX)):  # loop around edges.
        i, j = XX[idx], YY[idx]
        # idx = coords_to_upperidx(XX[idx],YY[idx],I)
        xC0 = np.fft.ifft(yfft[i, :]*np.conj(yfft[j, :]), axis=0)
        xC0 = np.real(xC0)
        if is_short[idx]:
            xC0 = np.concatenate((xC0[-shortlag+1:], xC0[0:shortlag]))
        else:
            xC0 = np.concatenate((xC0[-maxlag+1:], xC0[0:maxlag]))

        xC0 = np.fliplr([xC0])[0]
        Norm = np.sqrt(np.sum(np.abs(Y[i, :])**2)*np.sum(np.abs(Y[j, :])**2))

        xC0 = xC0/Norm
        if is_short[idx]:
            xCshort[short_idx, :] = xC0
            short_idx += 1
        else:
            xClong[long_idx, :] = xC0
            long_idx += 1
        del xC0

    if short_idx != np.sum(is_short):
        raise ValueError(
            f'Expected {np.sum(is_short)} short idcs, but got {short_idx}.')
    return xCshort, xClong


def xC_fft(Y, T, mxL=[], copy=True):

    # ***********************************
    # This should be checked! There shouldn't be any complex numbers!!
    # __main__:74: ComplexWarning: Casting complex values to real discards the imaginary part
    # This is because Python, in contrast to Matlab, produce highly prcise imaginary parts
    # by defualt, when you wanna do ifft, just use np.real()
    # ***********************************
    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print('xC_fft::: Input should be in IxT form, the matrix was transposed.')
        Y = np.transpose(Y)
    Y = np.array(Y)
    if not np.size(mxL):
        mxL = T

    I = np.shape(Y)[0]

    print("xC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y-np.transpose(np.tile(mY2, (T, 1)))

    nfft = nextpow2(2*T-1)  # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)  # be careful with the dimensions

    mxLcc = int((mxL-1)*2+1)
    xC = np.zeros([I, I, mxLcc])

    XX = np.triu_indices(I, 1)[0]
    YY = np.triu_indices(I, 1)[1]

    for i in np.arange(np.size(XX)):  # loop around edges.

        xC0 = np.fft.ifft(yfft[XX[i], :]*np.conj(yfft[YY[i], :]), axis=0)
        xC0 = np.real(xC0)
        xC0 = np.concatenate((xC0[-mxL+1:], xC0[0:mxL]))

        xC0 = np.fliplr([xC0])[0]
        Norm = np.sqrt(np.sum(np.abs(Y[XX[i], :])**2)
                       * np.sum(np.abs(Y[YY[i], :])**2))

        xC0 = xC0/Norm
        xC[XX[i], YY[i], :] = xC0
        del xC0

    xC = xC + np.transpose(xC, (1, 0, 2))
    lidx = np.arange(-(mxL-1), mxL)

    return xC, lidx


def nextpow2(x):
    """
    nextpow2 Next higher power of 2.
    nextpow2(N) returns the first P such that P >= abs(N).  It is
    often useful for finding the nearest power of two sequence
    length for FFT operations.
    """
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))


def ACL(ts, T):
    """
    Calculates Autocorrelation Length of time series ts
    SA, Ox, 2019
    """
    return(np.sum(AC_fft(ts, T)**2, axis=1))

    ######################## AC Reg Functions #################################


def tukeytaperme(ac, T, M, verbose=True):
    """
    performs single Tukey tapering for given length of window, M, and initial
    value, intv. intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    ac = ac.copy()
    # ----Checks:
    if not T in np.shape(ac):
        raise ValueError('tukeytaperme::: There is something wrong, mate!')
        # print('Oi')
    # ----

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M))/2
    tt_ts = np.zeros(np.shape(ac))

    if len(np.shape(ac)) == 2:
        if np.shape(ac)[1] != T:
            ac = ac.T
        if verbose:
            print('tukeytaperme::: The input is 2D.')
        N = np.shape(ac)[0]
        tt_ts[:, 0:M-1] = np.tile(tukeymultiplier, [N, 1])*ac[:, 0:M-1]

    elif len(np.shape(ac)) == 3:
        if verbose:
            print('tukeytaperme::: The input is 3D.')
        N = np.shape(ac)[0]
        tt_ts[:, :, 0:M-1] = np.tile(tukeymultiplier,
                                     [N, N, 1])*ac[:, :, 0:M-1]

    elif len(np.shape(ac)) == 1:
        if verbose:
            print('tukeytaperme::: The input is 1D.')
        tt_ts[0:M-1] = tukeymultiplier*ac[0:M-1]

    return(tt_ts)


def curbtaperme(ac, T, M, verbose=True):
    """
    Curb the autocorrelations, according to Anderson 1984
    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(np.shape(ac))
    if len(np.shape(ac)) == 2:
        if verbose:
            print('curbtaperme::: The input is 2D.')
        msk[:, 0:M] = 1

    elif len(np.shape(ac)) == 3:
        if verbose:
            print('curbtaperme::: The input is 3D.')
        msk[:, :, 0:M] = 1

    elif len(np.shape(ac)) == 1:
        if verbose:
            print('curbtaperme::: The input is 1D.')
        msk[0:M] = 1

    ct_ts = msk*ac

    return ct_ts


def shrinkme(ac, T):
    """
    Shrinks the *early* bucnhes of autocorr coefficients beyond the CI.
    Yo! this should be transformed to the matrix form, those fors at the top
    are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if np.shape(ac)[1] != T:
        ac = ac.T

    bnd = (np.sqrt(2)*1.3859)/np.sqrt(T)  # assumes normality for AC

    N = np.shape(ac)[0]
    msk = np.zeros(np.shape(ac))
    BreakPoint = np.zeros(N)
    for i in np.arange(N):
        # finds the break point -- intercept
        TheFirstFalse = np.where(np.abs(ac[i, :]) < bnd)
        # if you coulnd't find a break point, then continue = the row will remain zero
        if np.size(TheFirstFalse) == 0:
            continue
        else:
            BreakPoint_tmp = TheFirstFalse[0][0]
        msk[i, :BreakPoint_tmp] = 1
        BreakPoint[i] = BreakPoint_tmp
    return ac*msk, BreakPoint


def SumMat(Y0, T, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise summation of each row with other
         rows.

    SA, Ox, 2019
    """
    Y0 = Y0.T
    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print('SumMat::: Input should be in TxN form, the matrix was transposed.')
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)
    #F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0])-1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = (Y0[:, xx]+Y0[:, yy])
        SM[yy, xx, :] = (Y0[:, yy]+Y0[:, xx])

    return SM


def ProdMat(Y0, T, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise multiplication of each row with
         other rows.

    SA, Ox, 2019
    """
    Y0 = Y0.T
    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print('ProdMat::: Input should be in TxN form, the matrix was transposed.')
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)
    #F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0])-1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = (Y0[:, xx]*Y0[:, yy])
        SM[yy, xx, :] = (Y0[:, yy]*Y0[:, xx])

    return SM


def CorrMat(ts, T, method='rho', copy=True):
    """
    Produce sample correlation matrices
    or Naively corrected z maps.
    """

    if copy:
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        print('xDF::: Input should be in IxT form, the matrix was transposed.')
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    R = np.corrcoef(ts)

    Z = np.zeros_like(R)  # Initialize Z with zeros

    # Indices for non-diagonal elements
    non_diag_indices = np.where(~np.eye(N, dtype=bool))
    R_non_diag = R[non_diag_indices]  # Non-diagonal elements of R

    valid_indices = np.where((R_non_diag != 1) & (
        R_non_diag != -1))  # Indices where R is not 1 or -1
    Z[non_diag_indices] = np.arctanh(
        R_non_diag[valid_indices]) * np.sqrt(T - 3)

    R[range(N), range(N)] = 0
    Z[range(N), range(N)] = 0

    return R, Z


# %%

def xDF_Calc(ts, T,
             method='truncate',
             methodparam='adaptive',
             verbose=True,
             TV=True,
             copy=True):

    # -------------------------------------------------------------------------
    # READ AND CHECK 0---------------------------------------------------------

    #if not verbose: blockPrint()

    if copy:  # Make sure you are not messing around with the original time series
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        if verbose:
            print('xDF::: Input should be in IxT form, the matrix was transposed.')
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    # print(ts)
    ts_std = np.std(ts, axis=1, ddof=1)
    ts = ts/np.transpose(np.tile(ts_std, (T, 1)))  # standardise
    print('xDF_Calc::: Time series standardised by their standard deviations.')
    # ts = np.asmatrix(ts)
    # print(np.var(ts[0]))
    # Corr----------------------------------------------------------------------
    print("T: {}".format(T))
    rho, znaive = CorrMat(ts, T)
    # print(f"rho:\n{rho}")
    # print(f"znaive:\n{znaive}")
    rho = np.round(rho, 7)
    znaive = np.round(znaive, 7)

    [ac, CI] = AC_fft(ts, T)
    # The last element of ACF is rubbish, the first one is 1, so why bother?!
    ac = ac[:, 1:T-1]

    nLg = T-2

    # print("NLG {}".format(nLg))
    # Cross-corr----------------------------------------------------------------
    [xcf, lid] = xC_fft(ts, T)
    # [xcf_short,lid] = xC_fft(ts[shorts,:],T,mxL=shortlag); # TODO: is mxL use correct? instead input shorts and longs to xC_fft!!!
    # [xcf_long,lid] = xC_fft(ts[longs,:],T,mxL=maxlag);

    xc_p = xcf[:, :, 1:T-1]
    xc_p = np.flip(xc_p, axis=2)  # positive-lag xcorrs
    xc_n = xcf[:, :, T:-1]  # negative-lag xcorrs
    print('xcf shape ', xcf.shape)

    # -------------------------------------------------------------------------
    # Start of Regularisation--------------------------------------------------
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(T)
        else:
            M = methodparam
        if verbose:
            print('xDF_Calc::: AC Regularisation: Tukey tapering of M = ' +
                  str(int(np.round(M))))
        ac = tukeytaperme(ac, nLg, M)
        xc_p = tukeytaperme(xc_p, nLg, M)
        xc_n = tukeytaperme(xc_n, nLg, M)
        # print(np.round(ac[0,0:50],4))

    elif method.lower() == 'truncate':
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError(
                    'What?! Choose adaptive as the option, or pass an integer for truncation')
            if verbose:
                print('xDF_Calc::: AC Regularisation: Adaptive Truncation')
            [ac, bp] = shrinkme(ac, nLg)
            # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)
            for i in np.arange(N):
                for j in np.arange(N):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(
                        xc_p[i, j, :], nLg, maxBP, verbose=False)
                    xc_n[i, j, :] = curbtaperme(
                        xc_n[i, j, :], nLg, maxBP, verbose=False)
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    'xDF_Calc::: AC Regularisation: Non-adaptive Truncation on M = ' + str(methodparam))
            ac = curbtaperme(ac, nLg, methodparam)
            xc_p = curbtaperme(xc_p, nLg, methodparam)
            xc_n = curbtaperme(xc_n, nLg, methodparam)
            bp = methodparam*np.ones_like(ac[:, 0])

        else:
            raise ValueError(
                'xDF_Calc::: methodparam for truncation method should be either str or int.')

    # -------------------------------------------------------------------------
    # Start of the Monster Equation--------------------------------------------
    # -------------------------------------------------------------------------
    # '''
    wgt = np.arange(nLg, 0, -1)
    wgtm2 = np.tile((np.tile(wgt, [N, 1])), [N, 1])
    # this is shit, eats all the memory!
    wgtm3 = np.reshape(wgtm2, [N, N, np.size(wgt)])
    Tp = T-1

    """
    VarHatRho = (Tp*(1-rho.^2).^2 ...
    +   rho.^2 .* sum(wgtm3 .* (SumMat(ac.^2,nLg)  +  xc_p.^2 + xc_n.^2),3)...         %1 2 4
    -   2.*rho .* sum(wgtm3 .* (SumMat(ac,nLg)    .* (xc_p    + xc_n))  ,3)...         % 5 6 7 8
    +   2      .* sum(wgtm3 .* (ProdMat(ac,nLg)    + (xc_p   .* xc_n))  ,3))./(T^2);   % 3 9
    """

    # Da Equation!--------------------

    VarHatRho = (Tp*(1-rho**2)**2
                 + rho**2 *
                 np.sum(wgtm3 * (SumMat(ac**2, nLg) + xc_p**2 + xc_n**2), axis=2)
                 - 2*rho * np.sum(wgtm3 * (SumMat(ac, nLg)
                                  * (xc_p + xc_n)), axis=2)
                 + 2 * np.sum(wgtm3 * (ProdMat(ac, nLg) + (xc_p * xc_n)), axis=2))/(T**2)

    # Truncate to Theoritical Variance --------------------------------------
    TV_val = (1-rho**2)**2/T
    TV_val[range(N), range(N)] = 0

    idx_ex = np.where(VarHatRho < TV_val)
    NumTVEx = (np.shape(idx_ex)[1])/2
    # print(NumTVEx)

    if NumTVEx > 0 and TV:
        if verbose:
            print('Variance truncation is ON.')
        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        VarHatRho[idx_ex] = TV_val[idx_ex]
        # print(N)
        # print(np.shape(idx_ex)[1])
        FGE = N*(N-1)/2
        if verbose:
            print('xDF_Calc::: ' + str(NumTVEx) + ' (' + str(round((NumTVEx/FGE)*100, 3)
                                                             ) + '%) edges had variance smaller than the textbook variance!')
    else:
        if verbose:
            print('xDF_Calc::: NO truncation to the theoritical variance.')
    # Sanity Check:
    #        for ii in np.arange(NumTVEx):
    #            print( str( idx_ex[0][ii]+1 ) + '  ' + str( idx_ex[1][ii]+1 ) )

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    # delta method; make sure the N is correct! So they cancel out.
    sf = VarHatRho/((1-rho**2)**2)
    sf[range(N), range(N)] = 1  # diagonal is zeroed out anyways
    num_neg = np.sum(sf < 0)
    if num_neg > 0:
        # why???
        print(
            f'xDF_Calc::: Bad sign: Encountered {num_neg} negative VarHatRho entries ({np.round(100*num_neg/(sf.shape[0]*(sf.shape[0]-1)),2)} % of edges)')
    # sometimes VarHatRho can have negative entries??
    sf = np.maximum(1e-8, sf)

    try:
        rzf = rf/np.sqrt(sf)  # TODO: Moritz added 1e-8
    except Warning:
        print(rf, sf, sf[sf < 1e-8])
        rzf = rf/np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    VarHatRho[range(N), range(N)] = 0
    # NaN screws up everything, so get rid of the diag, but becareful here.
    f_pval[range(N), range(N)] = 0
    rzf[range(N), range(N)] = 0
    #result = np.tanh(rzf)

    # print(f"Zcc:\n{rzf}")
    # print(f"TanhZcc:\n{result}") # try to get everything in the interval [-1,1]
    xDFOut = {'p': f_pval,
              'z': rzf,
              'znaive': znaive,
              'v': VarHatRho,
              'TV': TV_val,
              'TVExIdx': idx_ex,
              'bp': bp}

    return xDFOut


def eff_xDF_old(ts, T,
                method='truncate',
                methodparam='adaptive',
                verbose=True,
                TV=True,
                copy=True,
                alpha=1):

    # -------------------------------------------------------------------------
    # READ AND CHECK 0---------------------------------------------------------

    #if not verbose: blockPrint()

    if copy:  # Make sure you are not messing around with the original time series
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        if verbose:
            print('xDF::: Input should be in IxT form, the matrix was transposed.')
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    # print(ts)
    ts_std = np.std(ts, axis=1, ddof=1)
    ts = ts/np.transpose(np.tile(ts_std, (T, 1)))  # standardise
    print('xDF_Calc::: Time series standardised by their standard deviations.')
    # ts = np.asmatrix(ts)
    # print(np.var(ts[0]))
    # Corr----------------------------------------------------------------------
    print("T: {}".format(T))
    rho, znaive = CorrMat(ts, T)
    # print(f"rho:\n{rho}")
    # print(f"znaive:\n{znaive}")
    rho = np.round(rho, 7)
    znaive = np.round(znaive, 7)

    [ac, CI] = AC_fft(ts, T)
    # The last element of ACF is rubbish, the first one is 1, so why bother?!
    ac = ac[:, 1:T-1]
    nLg = T-2
    if method.lower() == 'truncate':
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError(
                    'What?! Choose adaptive as the option, or pass an integer for truncation')
            if verbose:
                print('xDF_Calc::: AC Regularisation: Adaptive Truncation')
            [ac, bp] = shrinkme(ac, nLg)
    # truncate here and save tons of space and time below!
    # take 80% as smaller array. Find criterion to automize the quantile selection?
    maxlag = int(np.maximum(2, np.max(bp)))
    shortlag = int(np.percentile(bp, 80))
    shorts = (bp <= shortlag)

    print(
        f'Maxlag, shortlag =  {maxlag, np.max(bp), shortlag}, frac shorts: {np.mean(shorts)}')

    # print("NLG {}".format(nLg))
    # Cross-corr----------------------------------------------------------------
    # TODO: is mxL use correct? instead input shorts and longs to xC_fft!!!
    [xcf, lid] = xC_fft(ts, T, mxL=maxlag)
    # [xcf_short,lid] = xC_fft(ts[shorts,:],T,mxL=shortlag); # TODO: is mxL use correct? instead input shorts and longs to xC_fft!!!

    # [xcf_long,lid] = xC_fft(ts[longs,:],T,mxL=maxlag);

    # TODO: continue here:
    # xc_p      = xcf[:,:,1:T-1];
    # xc_p      = np.flip(xc_p,axis=2); #positive-lag xcorrs
    # xc_n      = xcf[:,:,T:-1];        #negative-lag xcorrs
    print('xcf shape ', xcf.shape)
    xc_p = np.flip(xcf[:, :, 1:maxlag], axis=2)
    xc_n = xcf[:, :, maxlag:]

    print('xc shapes: ', xc_p.shape, xc_n.shape)

    # xcs_p      = xcl_p[shorts, TODO: need short to long idcs!!]
    # xcs_n      = xcf_short[:,:,shortlag:-1];        #negative-lag xcorrs

    # -------------------------------------------------------------------------
    # Start of Regularisation--------------------------------------------------
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(T)
        else:
            M = methodparam
        if verbose:
            print('xDF_Calc::: AC Regularisation: Tukey tapering of M = ' +
                  str(int(np.round(M))))
        ac = tukeytaperme(ac, nLg, M)
        xc_p = tukeytaperme(xc_p, nLg, M)
        xc_n = tukeytaperme(xc_n, nLg, M)

        # print(np.round(ac[0,0:50],4))

    elif method.lower() == 'truncate':
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError(
                    'What?! Choose adaptive as the option, or pass an integer for truncation')
        #     if verbose: print('xDF_Calc::: AC Regularisation: Adaptive Truncation')
        #     [ac,bp] = shrinkme(ac,nLg)
        # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)
            for i in np.arange(N):
                for j in np.arange(N):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(
                        xc_p[i, j, :], nLg, maxBP, verbose=False)
                    xc_n[i, j, :] = curbtaperme(
                        xc_n[i, j, :], nLg, maxBP, verbose=False)
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    'xDF_Calc::: AC Regularisation: Non-adaptive Truncation on M = ' + str(methodparam))
            ac = curbtaperme(ac, nLg, methodparam)
            xc_p = curbtaperme(xc_p, nLg, methodparam)
            xc_n = curbtaperme(xc_n, nLg, methodparam)
            bp = methodparam*np.ones_like(ac[:, 0])

        else:
            raise ValueError(
                'xDF_Calc::: methodparam for truncation method should be either str or int.')

    # -------------------------------------------------------------------------
    # Start of the Monster Equation--------------------------------------------
    # -------------------------------------------------------------------------

    # MORITZ: To set b_n properly, instead of using weights w_k^xDF = N-2-k we want to use w_k^alpha = max(0,N (1-k/(N**alpha)))= max(0, N - k * N**(1-alpha))
    # TODO: find the best alpha!
    # let's compute the monster equation more memory-efficiently, but less time efficiently?!:
    VarHatRho = np.zeros((N, N))
    print(ac.shape)
    print(f'Mean truncation lag: {bp.mean()}')
    if alpha < 1:
        # faster decay replaces truncation
        weights = np.maximum(0, T - np.arange(0, nLg) * T**(1-alpha))
    else:
        weights = np.arange(nLg, 0, -1)/T  # original weights
    # TODO: use my weights and remove 1-1/T
    # print(f'weights {weights.shape}, ac {ac.shape}, xc_p {xc_p.shape}, xc_n {xc_n.shape}')
    # print(ac)
    # TODO: only sum to bp! probably much less compute time and space
    num_wrongrho = np.sum(np.abs(rho) > 1)
    if num_wrongrho > 0:
        print(f'{num_wrongrho} rho entries greater than 1')
    VarHatRho = (1-rho**2)**2 / T  # * (1-1/T) / T # lag-0 term
    # maxBP=int(np.max(bp))
    len3 = xc_p.shape[2]
    weights, ac, xc_p, xc_n = weights[:len3], ac[:,
                                                 :len3], xc_p[:, :, :len3], xc_n[:, :, :len3]
    for i in range(N):
        for j in range(i):  # TODO: is xc symmetric? seems like
            # if method.lower()=='truncate':
            #     maxBP        = int(np.max([bp[i],bp[j]]))
            # else:
            #     maxBP=nLg
            # VarHatRho[i,j] += (rho[i,j]**2 * np.sum(weights[:maxBP] * (ac[i,:maxBP]**2+ac[j,:maxBP]**2+xc_p[i,j,:maxBP]**2+xc_n[i,j,:maxBP]**2))\
            #     - 2 * rho[i,j] * np.sum(weights[:maxBP] * ((ac[i,:maxBP]+ac[j,:maxBP])*(xc_p[i,j,:maxBP]+xc_n[i,j,:maxBP])))\
            #     + 2* np.sum(weights[:maxBP] * (ac[i,:maxBP]*ac[j,:maxBP] + xc_p[i,j,:maxBP]*xc_n[i,j,:maxBP])))/T

            VarHatRho[i, j] += (rho[i, j]**2 * np.sum(weights * (ac[i, :]**2+ac[j, :]**2+xc_p[i, j, :]**2+xc_n[i, j, :]**2))
                                - 2 *
                                rho[i, j] * np.sum(weights * ((ac[i, :]+ac[j, :])
                                                   * (xc_p[i, j, :]+xc_n[i, j, :])))
                                + 2 * np.sum(weights * (ac[i, :]*ac[j, :] + xc_p[i, j, :]*xc_n[i, j, :])))/T
            VarHatRho[j, i] = VarHatRho[i, j]
    # print(f"VarHatRho:\n {VarHatRho}")

    # Truncate to Theoritical Variance --------------------------------------
    TV_val = (1-rho**2)**2/T
    TV_val[range(N), range(N)] = 0

    idx_ex = np.where(VarHatRho < TV_val)
    NumTVEx = (np.shape(idx_ex)[1])/2
    # print(NumTVEx)

    if NumTVEx > 0 and TV:
        if verbose:
            print('Variance truncation is ON.')
        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        VarHatRho[idx_ex] = TV_val[idx_ex]
        # print(N)
        # print(np.shape(idx_ex)[1])
        FGE = N*(N-1)/2
        if verbose:
            print('xDF_Calc::: ' + str(NumTVEx) + ' (' + str(round((NumTVEx/FGE)*100, 3)
                                                             ) + '%) edges had variance smaller than the textbook variance!')
    else:
        if verbose:
            print('xDF_Calc::: NO truncation to the theoritical variance.')
    # Sanity Check:
    #        for ii in np.arange(NumTVEx):
    #            print( str( idx_ex[0][ii]+1 ) + '  ' + str( idx_ex[1][ii]+1 ) )

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    # delta method; make sure the N is correct! So they cancel out.
    sf = VarHatRho/((1-rho**2)**2)
    sf[range(N), range(N)] = 1  # diagonal is zeroed out anyways
    num_neg = np.sum(sf < 0)
    if num_neg > 0:
        # why???
        print(
            f'xDF_Calc::: Bad sign: Encountered {num_neg} negative VarHatRho entries ({np.round(100*num_neg/(sf.shape[0]*(sf.shape[0]-1)),2)} % of edges)')
    # sometimes VarHatRho can have negative entries??
    sf = np.maximum(1e-8, sf)

    try:
        rzf = rf/np.sqrt(sf)  # TODO: Moritz added 1e-8
    except Warning:
        print(rf, sf, sf[sf < 1e-8])
        rzf = rf/np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    VarHatRho[range(N), range(N)] = 0
    # NaN screws up everything, so get rid of the diag, but becareful here.
    f_pval[range(N), range(N)] = 0
    rzf[range(N), range(N)] = 0
    #result = np.tanh(rzf)

    # print(f"Zcc:\n{rzf}")
    # print(f"TanhZcc:\n{result}") # try to get everything in the interval [-1,1]
    xDFOut = {'p': f_pval,
              'z': rzf,
              'znaive': znaive,
              'v': VarHatRho,
              'TV': TV_val,
              'TVExIdx': idx_ex,
              'bp': bp}

    return xDFOut


def eff_xDF(ts, T,
            method='truncate',
            methodparam='adaptive',
            verbose=True,
            TV=True,
            copy=True,
            alpha=1,
            percentile=80):

    # -------------------------------------------------------------------------
    # READ AND CHECK 0---------------------------------------------------------

    #if not verbose: blockPrint()

    if copy:  # Make sure you are not messing around with the original time series
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        if verbose:
            print('xDF::: Input should be in IxT form, the matrix was transposed.')
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    # print(ts)
    ts_std = np.std(ts, axis=1, ddof=1)
    ts = ts/np.transpose(np.tile(ts_std, (T, 1)))  # standardise
    print('xDF_Calc::: Time series standardised by their standard deviations.')
    # ts = np.asmatrix(ts)
    # print(np.var(ts[0]))
    # Corr----------------------------------------------------------------------
    print("T: {}".format(T))
    rho, znaive = CorrMat(ts, T)
    # print(f"rho:\n{rho}")
    # print(f"znaive:\n{znaive}")
    rho = np.round(rho, 7)
    znaive = np.round(znaive, 7)

    [ac, CI] = AC_fft(ts, T)
    # The last element of ACF is rubbish, the first one is 1, so why bother?!
    ac = ac[:, 1:T-1]
    nLg = T-2
    if method.lower() == 'truncate':
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError(
                    'What?! Choose adaptive as the option, or pass an integer for truncation')
            if verbose:
                print('xDF_Calc::: AC Regularisation: Adaptive Truncation')
            [ac, bp] = shrinkme(ac, nLg)
    # TODO: truncate here and save tons of space and time below!!
    # TODO: calc maxbp, but take 80% as smaller array. Find criterion to automize the quantile selection?
    maxlag = int(np.maximum(2, np.max(bp)))
    shortlag = int(np.maximum(2, np.percentile(bp, percentile)))
    shorts = (bp <= shortlag)
    # upper_to_matrix(uparray,diag=1), coords_to_idx(i, j, num_cols)
    print(
        f'Maxlag, shortlag =  {maxlag, np.max(bp), shortlag}, frac shorts: {np.mean(shorts)}')

    # print("NLG {}".format(nLg))
    # Cross-corr----------------------------------------------------------------
    is_short = np.array([[(shorts[i] and shorts[j]) for j in range(N)]
                        for i in range(N)])[np.triu_indices(N, 1)]
    xcf_short, xcf_long = xC_fft_flat(ts, T, is_short, shortlag, maxlag)
    # [xcf_short,lid] = xC_fft(ts[shorts,:],T,mxL=shortlag); # TODO: is mxL use correct? instead input shorts and longs to xC_fft!!!
    # [xcf_long,lid] = xC_fft(ts[longs,:],T,mxL=maxlag);

    # TODO: continue here:
    # xc_p      = xcf[:,:,1:T-1];
    # xc_p      = np.flip(xc_p,axis=2); #positive-lag xcorrs
    # xc_n      = xcf[:,:,T:-1];        #negative-lag xcorrs
    print('xcf shape ', xcf_short.shape, xcf_long.shape)
    xcl_p = np.flip(xcf_long[:, 1:maxlag], axis=1)
    xcl_n = xcf_long[:, maxlag:]
    xcs_p = np.flip(xcf_short[:, 1:shortlag], axis=1)
    xcs_n = xcf_short[:, shortlag:]
    print('xcl shapes: ', xcl_p.shape, xcl_n.shape)

    # xcs_p      = xcl_p[shorts, TODO: need short to long idcs!!]
    # xcs_n      = xcf_short[:,:,shortlag:-1];        #negative-lag xcorrs

    # -------------------------------------------------------------------------
    # Start of Regularisation--------------------------------------------------
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(T)
        else:
            M = methodparam
        if verbose:
            print('xDF_Calc::: AC Regularisation: Tukey tapering of M = ' +
                  str(int(np.round(M))))
        ac = tukeytaperme(ac, nLg, M)
        xc_p = tukeytaperme(xc_p, nLg, M)
        xc_n = tukeytaperme(xc_n, nLg, M)

        # print(np.round(ac[0,0:50],4))

    elif method.lower() == 'truncate':
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError(
                    'What?! Choose adaptive as the option, or pass an integer for truncation')
        #     if verbose: print('xDF_Calc::: AC Regularisation: Adaptive Truncation')
        #     [ac,bp] = shrinkme(ac,nLg)
        # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)

            XX = np.triu_indices(N, 1)[0]
            YY = np.triu_indices(N, 1)[1]
            short_idx, long_idx = 0, 0
            for idx in np.arange(np.size(XX)):  # idx = coords_to_upperidx(i,j,N)
                i, j = XX[idx], YY[idx]
                maxBP = np.max([bp[i], bp[j]])
                if is_short[idx]:
                    xcs_p[short_idx, :] = curbtaperme(
                        xcs_p[short_idx, :], nLg, maxBP, verbose=False)
                    xcs_n[short_idx, :] = curbtaperme(
                        xcs_n[short_idx, :], nLg, maxBP, verbose=False)
                    short_idx += 1
                else:
                    xcl_p[long_idx, :] = curbtaperme(
                        xcl_p[long_idx, :], nLg, maxBP, verbose=False)
                    xcl_n[long_idx, :] = curbtaperme(
                        xcl_n[long_idx, :], nLg, maxBP, verbose=False)
                    long_idx += 1
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    'xDF_Calc::: AC Regularisation: Non-adaptive Truncation on M = ' + str(methodparam))
            ac = curbtaperme(ac, nLg, methodparam)
            xc_p = curbtaperme(xc_p, nLg, methodparam)
            xc_n = curbtaperme(xc_n, nLg, methodparam)
            bp = methodparam*np.ones_like(ac[:, 0])

        else:
            raise ValueError(
                'xDF_Calc::: methodparam for truncation method should be either str or int.')

    # -------------------------------------------------------------------------
    # Start of the Monster Equation--------------------------------------------
    # -------------------------------------------------------------------------

    # MORITZ: To set b_n properly, instead of using weights w_k^xDF = N-2-k we want to use w_k^alpha = max(0,N (1-k/(N**alpha)))= max(0, N - k * N**(1-alpha))
    # TODO: find the best alpha?
    # let's compute the monster equation more memory-efficiently, but less time efficiently?!:
    VarHatRho = np.zeros((N, N))
    print(ac.shape)
    print(f'Mean truncation lag: {bp.mean()}')
    if alpha < 1:
        # faster decay replaces truncation
        weights = np.maximum(0, T - np.arange(0, nLg) * T**(1-alpha))
    else:
        weights = np.arange(nLg, 0, -1)/T  # original weights
    # Moritz: using my weights and remove 1-1/T
    # print(f'weights {weights.shape}, ac {ac.shape}, xc_p {xc_p.shape}, xc_n {xc_n.shape}')
    # print(ac)
    num_wrongrho = np.sum(np.abs(rho) > 1)
    if num_wrongrho > 0:
        print(f'{num_wrongrho} rho entries greater than 1')
    VarHatRho = (1-rho**2)**2 / T  # * (1-1/T) / T # lag-0 term
    # maxBP=int(np.max(bp))
    len_long = xcl_p.shape[1]
    weights_long, ac_long = weights[:len_long], ac[:, :len_long]
    len_short = xcs_p.shape[1]
    weights_short, ac_short = weights[:len_short], ac[:, :len_short]

    XX = np.triu_indices(N, 1)[0]
    YY = np.triu_indices(N, 1)[1]
    short_idx, long_idx = 0, 0
    for idx in np.arange(np.size(XX)):  # idx = coords_to_upperidx(i,j,N)
        i, j = XX[idx], YY[idx]
        # for i in range(N):
        #     for j in range(i+1,N):

        # if method.lower()=='truncate':
        #     maxBP        = int(np.max([bp[i],bp[j]]))
        # else:
        #     maxBP=nLg
        # VarHatRho[i,j] += (rho[i,j]**2 * np.sum(weights[:maxBP] * (ac[i,:maxBP]**2+ac[j,:maxBP]**2+xc_p[i,j,:maxBP]**2+xc_n[i,j,:maxBP]**2))\
        #     - 2 * rho[i,j] * np.sum(weights[:maxBP] * ((ac[i,:maxBP]+ac[j,:maxBP])*(xc_p[i,j,:maxBP]+xc_n[i,j,:maxBP])))\
        #     + 2* np.sum(weights[:maxBP] * (ac[i,:maxBP]*ac[j,:maxBP] + xc_p[i,j,:maxBP]*xc_n[i,j,:maxBP])))/T
        if is_short[idx]:
            VarHatRho[i, j] += (rho[i, j]**2 * np.sum(weights_short * (ac_short[i, :]**2+ac_short[j, :]**2+xcs_p[short_idx, :]**2+xcs_n[short_idx, :]**2))
                                - 2 * rho[i, j] * np.sum(weights_short * (
                                    (ac_short[i, :]+ac_short[j, :])*(xcs_p[short_idx, :]+xcs_n[short_idx, :])))
                                + 2 * np.sum(weights_short * (ac_short[i, :]*ac_short[j, :] + xcs_p[short_idx, :]*xcs_n[short_idx, :])))/T
            short_idx += 1
        else:
            VarHatRho[i, j] += (rho[i, j]**2 * np.sum(weights_long * (ac_long[i, :]**2+ac_long[j, :]**2+xcl_p[long_idx, :]**2+xcl_n[long_idx, :]**2))
                                - 2 * rho[i, j] * np.sum(weights_long * (
                                    (ac_long[i, :]+ac_long[j, :])*(xcl_p[long_idx, :]+xcl_n[long_idx, :])))
                                + 2 * np.sum(weights_long * (ac_long[i, :]*ac_long[j, :] + xcl_p[long_idx, :]*xcl_n[long_idx, :])))/T
            long_idx += 1
        VarHatRho[j, i] = VarHatRho[i, j]
    # print(f"VarHatRho:\n {VarHatRho}")

    # Truncate to Theoritical Variance --------------------------------------
    TV_val = (1-rho**2)**2/T
    TV_val[range(N), range(N)] = 0

    idx_ex = np.where(VarHatRho < TV_val)
    NumTVEx = (np.shape(idx_ex)[1])/2
    # print(NumTVEx)

    if NumTVEx > 0 and TV:
        if verbose:
            print('Variance truncation is ON.')
        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        VarHatRho[idx_ex] = TV_val[idx_ex]
        # print(N)
        # print(np.shape(idx_ex)[1])
        FGE = N*(N-1)/2
        if verbose:
            print('xDF_Calc::: ' + str(NumTVEx) + ' (' + str(round((NumTVEx/FGE)*100, 3)
                                                             ) + '%) edges had variance smaller than the textbook variance!')
    else:
        if verbose:
            print('xDF_Calc::: NO truncation to the theoritical variance.')
    # Sanity Check:
    #        for ii in np.arange(NumTVEx):
    #            print( str( idx_ex[0][ii]+1 ) + '  ' + str( idx_ex[1][ii]+1 ) )

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    # delta method; make sure the N is correct! So they cancel out.
    sf = VarHatRho/((1-rho**2)**2)
    sf[range(N), range(N)] = 1  # diagonal is zeroed out anyways
    num_neg = np.sum(sf < 0)
    if num_neg > 0:
        # why???
        print(
            f'xDF_Calc::: Bad sign: Encountered {num_neg} negative VarHatRho entries ({np.round(100*num_neg/(sf.shape[0]*(sf.shape[0]-1)),2)} % of edges)')
    # sometimes VarHatRho can have negative entries??
    sf = np.maximum(1e-8, sf)

    try:
        rzf = rf/np.sqrt(sf)  # TODO: Moritz added 1e-8
    except Warning:
        print(rf, sf, sf[sf < 1e-8])
        rzf = rf/np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    VarHatRho[range(N), range(N)] = 0
    # NaN screws up everything, so get rid of the diag, but becareful here.
    f_pval[range(N), range(N)] = 0
    rzf[range(N), range(N)] = 0
    #result = np.tanh(rzf)

    # print(f"Zcc:\n{rzf}")
    # print(f"TanhZcc:\n{result}") # try to get everything in the interval [-1,1]
    xDFOut = {'p': f_pval,
              'z': rzf,
              'znaive': znaive,
              'v': VarHatRho,
              'TV': TV_val,
              'TVExIdx': idx_ex,
              'bp': bp}

    return xDFOut


# return VarHatRho, rzf, f_pval, znaive
