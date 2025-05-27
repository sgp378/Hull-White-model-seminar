import numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import minimize, Bounds
from scipy.special import ndtr, gammainc
from scipy.linalg import sqrtm
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

# Conversions between ZCB prices, spot rates forward rates and libor rates
def zcb_prices_from_spot_rates(T,R):
    M = len(T)
    p = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-8:
            p[i] = 1
        else:
            p[i] = np.exp(-R[i]*T[i])
    return p

def spot_rates_from_zcb_prices(T,p):
    M = len(T)
    r = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-12:
            r[i] = np.nan
        else:
            r[i] = -np.log(p[i])/T[i]
    return r

def forward_rates_from_zcb_prices(T,p,horizon = 1):
    # horizon = 0 corresponds to approximated instantaneous forward rates. Note that the first entry of T is assumed to be T[0] = 0
    M = len(T)
    f = np.zeros([M])
    if horizon == 0:
        f[0] = (np.log(p[0])-np.log(p[1]))/(T[1]-T[0])
        f[-1] = (np.log(p[-2])-np.log(p[-1]))/(T[-1]-T[-2])
        m = 1
        while m < M - 1.5:
            f[m] = (np.log(p[m-1])-np.log(p[m+1]))/(T[m+1]-T[m-1])
            m += 1
    elif 0 < horizon:
        m = horizon
        while m < M - 0.5:
            f[m] = (np.log(p[m-horizon])-np.log(p[m]))/(T[m]-T[m-horizon])
            m += 1
    return f

def forward_libor_rates_from_zcb_prices(T,p,horizon = 1):
    M = len(T)
    f = np.zeros([M])
    i = horizon
    while i < M - 0.5:
        f[i] = (p[i-horizon]-p[i])/(p[i]*(T[i]-T[i-horizon]))
        i += 1
    return f

def accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p):
    T_fix = []
    if type(fixed_freq) == str:
        if fixed_freq == "quarterly":
            for i in range(1,int((T_N-T_n)*4) + 1):
                if T_n + i*0.25 > t:
                    T_fix.append(T_n + i*0.25)
        elif fixed_freq == "semiannual":
            for i in range(1,int((T_N-T_n)*2) + 1):
                if T_n + i*0.5 > t:
                    T_fix.append(T_n + i*0.5)
        elif fixed_freq == "annual":
            for i in range(1,int(T_N-T_n) + 1):
                if T_n + i > t:
                    T_fix.append(T_n + i)
    elif type(fixed_freq) == int or type(fixed_freq) == float or type(fixed_freq) == np.int32 or type(fixed_freq) == np.int64 or type(fixed_freq) == np.float64:
        for i in range(1,int((T_N-T_n)/fixed_freq) + 1):
            if T_n + i*fixed_freq > t:
                T_fix.append(T_n + i*fixed_freq)
    p_fix = np.array(for_values_in_list_find_value_return_value(T_fix,T,p))
    T_fix = np.array(T_fix)
    S = (T_fix[0] - T_n)*p_fix[0]
    for i in range(1,len(T_fix)):
        S += (T_fix[i] - T_fix[i-1])*p_fix[i]
    return S

def swap_rate_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p,float_freq = 0,L = 0):
    S = accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p)
    if t <= T_n:
        if T_n < 1e-6:
            p_n = 1
        else:
            Ind_n, output_n = find_value_return_value(T_n,T,p)
            if Ind_n == True:
                p_n = output_n[0][1]
        Ind_N, output_N = find_value_return_value(T_N,T,p)
        if Ind_N == True:
            p_N = output_N[0][1]
        R = (p_n-p_N)/S
    elif t > T_n:
        if float_freq == 0:
            print(f"WARNING! Since t is after inception, 'float_freq' must be given as an argument")
            R = np.nan
        else:
            if type(float_freq) == str:
                if float_freq == "quarterly":
                    float_freq = 0.25
                elif float_freq == "semiannual":
                    float_freq = 0.5
                elif fixed_freq == "annual":
                    float_freq = 1
            i, I_done = 0, False
            while I_done == False and i*float_freq < T_N:
                if i*float_freq >= t:
                    T_n = i*float_freq
                    I_done = True
                i += 1
            if I_done == True:
                [p_n,p_N] = for_values_in_list_find_value_return_value([T_n,T_N],T,p)
                R = (((T_n-t)*L+1)*p_n-p_N)/S
            else:
                print(f"WARNING! Not able to compute the par swap rate")
                R = np.nan
    return R, S

def spot_rate_bump(T_bump,size_bump,T,R_input,p_input):
    R, p = R_input.copy(), p_input.copy()
    if type(T_bump) == int or type(T_bump) == float or type(T_bump) == np.float64 or type(T_bump) == np.int32 or type(T_bump) == np.int64:
        I_bump, idx_bump = value_in_list_returns_I_idx(T_bump,T)
        R[idx_bump] = R[idx_bump] + size_bump
        p[idx_bump] = np.exp(-R[idx_bump]*T_bump)
    elif type(T_bump) == tuple or type(T_bump) == list or type(T_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump[i]
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
    return R, p

def market_rate_bump(idx_bump,size_bump,T_inter,data,interpolation_options = {"method": "linear"}):
    data_bump = copy.deepcopy(data)
    if type(idx_bump) == int or type(idx_bump) == float or type(idx_bump) == np.float64 or type(idx_bump) == np.int32 or type(idx_bump) == np.int64:
        data_bump[idx_bump]["rate"] += size_bump
        T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
        p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
    elif type(idx_bump) == tuple or type(idx_bump) == list or type(idx_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64 or type(size_bump) == np.int32 or type(size_bump) == np.int64:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump[i]
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
    return p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump, data_bump

#  Fixed rate bond
def macauley_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/pv
    return D

def modified_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/(pv*(1+ytm))
    return D

def convexity(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]**2/(1+ytm)**T[i]
    D = D/pv
    return D

def price_fixed_rate_bond_from_ytm(ytm,T,C):
    price = 0
    N = len(T)
    for i in range(0,N):
        price += C[i]/(1+ytm)**T[i]
    return price

def ytm(pv,T,C,ytm_init = 0.05):
    args = (pv, T, C, 1)
    result = minimize(ytm_obj,ytm_init,args = args, options={'disp': False})
    ytm = result.x[0]
    return ytm

def ytm_obj(ytm,pv,T,C,scaling = 1):
    N = len(T)
    pv_new = 0
    for i in range(0,N):
        pv_new += C[i]/(1+ytm[0])**T[i]
    sse = scaling*(pv-pv_new)**2
    return sse

# Cox-Ingersoll-Ross short rate sigma_model
def zcb_price_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float32 or type(T) == np.float64:
        gamma = np.sqrt(a**2+2*sigma**2)
        D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
        A = ((2*gamma*np.exp(0.5*T*(a+gamma)))/D)**((2*a*b)/(sigma**2))
        B = 2*(np.exp(gamma*T)-1)/D
        p = A*np.exp(-r0*B)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        p = np.zeros([M])
        for i in range(0,M):
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*T[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*T[i])-1)/D
            p[i] = A*np.exp(-r0*B)
    else:
        print(f"T not a recognized type")
        p = False
    return p

def spot_rate_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if T < 1e-6:
            r = r0
        else:
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*T*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*T)-1)/D
            r = (-np.log(A)+r0*B)/(T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        r = np.zeros([M])
        for i in range(0,M):
            if T[i] < 1e-6:
                r[i] = r0
            else:
                gamma = np.sqrt(a**2+2*sigma**2)
                D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
                A = ((2*gamma*np.exp(0.5*T[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
                B = 2*(np.exp(gamma*T[i])-1)/D
                r[i] = (-np.log(A)+r0*B)/(T[i])
    else:
        print(f"T not a recognized type")
        r = False
    return r

def forward_rate_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if T < 1e-6:
            f = r0
        else:
            c = (2*a*b)/(sigma**2)
            gamma = np.sqrt(a**2+2*sigma**2)
            N = 2*gamma*np.exp(0.5*T*(a+gamma))
            N_T = gamma*(gamma+a)*np.exp(0.5*T*(a+gamma))
            D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
            D_T = gamma*(a+gamma)*np.exp(gamma*T)
            M = 2*(np.exp(gamma*T)-1)
            M_T = 2*gamma*np.exp(gamma*T)
            f = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        f = np.zeros([N])
        for i in range(0,N):
            if T[i] < 1e-6:
                f[i] = r0
            else:
                c = (2*a*b)/(sigma**2)
                gamma = np.sqrt(a**2+2*sigma**2)
                N = 2*gamma*np.exp(0.5*T[i]*(a+gamma))
                N_T = gamma*(gamma+a)*np.exp(0.5*T[i]*(a+gamma))
                D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
                D_T = gamma*(a+gamma)*np.exp(gamma*T[i])
                M = 2*(np.exp(gamma*T[i])-1)
                M_T = 2*gamma*np.exp(gamma*T[i])
                f[i] = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    else:
        print(f"T not a recognized type")
        f = False
    return f

def mean_cir(r0,a,b,sigma,T):
    if T == np.inf:
        mean = b
    else:
        df = (4*a*b)/sigma**2
        nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
        factor = (sigma**2*(1-np.exp(-a*T)))/(4*a)
        mean = factor*ncx2.stats(df,nc,moments='m')
    return mean

def stdev_cir(r0,a,b,sigma,T):
    if T == np.inf:
        stdev = sigma*np.sqrt(b/(2*a))
    else:
        df = (4*a*b)/sigma**2
        nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
        factor = (sigma**2*(1-np.exp(-a*T)))/(4*a)
        stdev = factor*np.sqrt(ncx2.stats(df,nc,moments='v'))
    return stdev

def ci_cir(r0,a,b,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if type_ci == "lower":
            if T < 1e-6:
                lb, ub = r0, r0
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif type_ci == "upper":
            if T < 1e-6:
                lb, ub = r0, r0
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif type_ci == "two_sided":
            if T < 1e-6:
                lb, ub = r0, r0
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif type_ci == "upper":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif type_ci == "two_sided":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
                    # (4*b)/(sigma**2*(1-np.exp(-b*tau[i])))*
    else:
        print(f"tau not a recognized type")
        lb,ub = False, False
    return lb, ub

def simul_cir(r0,a,b,sigma,M,T,method = "exact"):
    delta = T/M
    r = np.zeros([M+1])
    r[0] = r0
    if method == "exact":
        delta_sqrt = np.sqrt(delta)
        df = (4*a*b)/sigma**2
        factor = (sigma**2*(1-np.exp(-a*delta)))/(4*a)
        for m in range(1,M+1):
            nc = (4*a*np.exp(-a*delta)*r[m-1])/(sigma**2*(1-np.exp(-a*delta)))
            # r_hat = factor*np.random.noncentral_chisquare(df,nc)
            r_hat = factor*ncx2.rvs(df,nc)
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    elif method == "euler":
        delta_sqrt = np.sqrt(delta)
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r_hat = r[m-1] + a*(b-r[m-1])*delta + sigma*np.sqrt(r[m-1])*delta_sqrt*Z[m-1]
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    elif method == "milstein":
        delta_sqrt = np.sqrt(delta)
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r_hat = r[m-1] + a*(b-r[m-1])*delta + sigma*np.sqrt(r[m-1])*delta_sqrt*Z[m-1] + 0.25*sigma**2*delta*(Z[m-1]**2-1)
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    return r

def fit_cir_obj(param,R_star,T,scaling = 1):
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_cir_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

# Vasicek short rate model
def zcb_price_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
        p = np.exp(A-r0*B)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        p = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
            p[i] = np.exp(A-r0*B)
    else:
        print(f"T not of a recognized type")
        p = False
    return p

def spot_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
        if T < 1e-6:
            r = r0
        elif T >= 1e-6:
            r = (-A+r0*B)/T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        r = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
            if T[i] < 1e-6:
                r[i] = r0
            else:
                r[i] = (-A+r0*B)/T[i]
    else:
        print(f"T not of a recognized type")
        r = False
    return r

def forward_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        B_T = np.exp(-a*T)
        if T < 1e-6:
            f = r0
        elif T >= 1e-6:
            f = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        f = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            B_T = np.exp(-a*T[i])
            if T[i] < 1e-6:
                f[i] = r0
            else:
                f[i] = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    else:
        print(f"T not of a recognized type")
        f = False
    return f

def mean_vasicek(r0,a,b,sigma,T):
    if T == np.inf:
        mean = b/a
    else:
        mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
    return mean

def stdev_vasicek(r0,a,b,sigma,T):
    if T == np.inf:
        std = np.sqrt(sigma**2/(2*a))
    else:
        std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
    return std

def ci_vasicek(r0,a,b,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, np.inf
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = -np.inf, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, mean + z*std
        # print(f"type_ci: {type_ci}, z: {z}")
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, mean + z*std
    else:
        print(f"T is not of recognized type")
        lb,ub = False, False
    return lb, ub

def simul_vasicek(r0,a,b,sigma,M,T,method = "exact",seed = None):
    if seed is not None:
        np.random.seed(seed)
    delta = T/M
    r = np.zeros([M+1])
    r[0] = r0
    Z = np.random.standard_normal(M)
    if method == "exact":
        for m in range(1,M+1):
            r[m] = r[m-1]*np.exp(-a*delta) + (b/a)*(1-np.exp(-a*delta)) + sigma*np.sqrt((1-np.exp(-2*a*delta))/(2*a))*Z[m-1]
    elif method == "euler" or method == "milstein":
        delta_sqrt = np.sqrt(delta)
        for m in range(1,M+1):
            r[m] = r[m-1] + (b-a*r[m-1])*delta + sigma*delta_sqrt*Z[m-1]
    return r

def euro_option_price_vasicek(K,T1,T2,p_T1,p_T2,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(T2-T1)))*np.sqrt((1-np.exp(-2*a*T1))/(2*a))
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def caplet_prices_vasicek(sigma,strike,a,T,p):
    price_caplet = np.zeros([len(T)])
    for i in range(2,len(T)):
        price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*euro_option_price_vasicek(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
    return price_caplet

def fit_vasicek_obj(param,R_star,T,scaling = 1):
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_vasicek_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

# Ho-Lee model
def theta_ho_lee(t,param,method = "default",f_T = None):
    if method == "default":
        sigma = param
        if type(t) == int or type(t) == float or type(t) == np.int32 or type(t) == np.int64 or type(t) == np.float64:
            theta = f_T + sigma**2*t
        elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
            N = len(t)
            theta = np.zeros(N)
            for n in range(0,N):
                theta[n] = f_T[n] + sigma**2*t[n]
    elif method == "nelson_siegel":
        f_inf, a, b, sigma = param
        if type(t) == int or type(t) == float or type(t) == np.int32 or type(t) == np.int64 or type(t) == np.float64:
            K = len(a)
            theta = -a[0]*b[0]*np.exp(-b[0]*t) + sigma**2*t
            for k in range(1,K):
                theta += a[k]*k*t**(k-1)*np.exp(-b[k]*t) - a[k]*b[k]*t**k*np.exp(-b[k]*t)
        elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
            K = len(a)
            M = len(t)
            theta = np.zeros([M])
            for m in range(0,M):
                theta[m] = -a[0]*b[0]*np.exp(-b[0]*t[m]) + sigma**2*t[m]
                for k in range(1,K):
                    theta[m] += a[k]*k*t[m]**(k-1)*np.exp(-b[k]*t[m]) - a[k]*b[k]*t[m]**k*np.exp(-b[k]*t[m])

    return theta

def zcb_price_ho_lee(t,T,r,sigma,T_star,p_star,f_star):
    if type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        p = N*[None]
        for i in range(0,N):
            p_t = for_values_in_list_find_value_return_value(t,T_star,p_star)
            p_T = for_values_in_list_find_value_return_value(T[i],T_star,p_star)
            p[i] = (p_T/p_t)*np.exp((T[i]-t)*(f_star-r) - (sigma**2/2)*t*(T[i]-t)**2)
    elif type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        p_t = for_values_in_list_find_value_return_value(t,T_star,p_star)
        p_T = for_values_in_list_find_value_return_value(T,T_star,p_star)
        p = (p_T/p_t)*np.exp((T-t)*(f_star-r) - (sigma**2/2)*t*(T-t)**2)
    return np.array(p)

def mean_var_ho_lee(f,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        mean, var = f + 0.5*sigma**2*T, sigma**2*T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        mean, var = np.zeros(N), np.zeros(N)
        for i in range(0,N):
            mean[i], var[i] = f[i] + 0.5*sigma**2*T[i], sigma**2*T[i]
    return mean, var

def ci_ho_lee(f,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        if type_ci == "lower":
            mean = f + 0.5*sigma**2*T
            std = sigma*np.sqrt(T)
            z = norm.ppf(size_ci,0,1)
            lb, ub = mean - z*std, np.inf
        elif type_ci == "upper":
            mean = f + 0.5*sigma**2*T
            std = sigma*np.sqrt(T)
            z = norm.ppf(size_ci,0,1)
            lb, ub = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            mean = f + 0.5*sigma**2*T
            std = sigma*np.sqrt(T)
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            lb, ub = mean - z*std, mean + z*std
        print(f"type_ci: {type_ci}, z: {z}")
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                mean = f[i] + 0.5*sigma**2*T[i]
                std = sigma*np.sqrt(T[i])
                lb[i], ub[i] = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                mean = f[i] + 0.5*sigma**2*T[i]
                std = sigma*np.sqrt(T[i])
                lb[i], ub[i] = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            for i in range(0,N):
                mean = f[i] + 0.5*sigma**2*T[i]
                std = sigma*np.sqrt(T[i])
                lb[i], ub[i] = mean - z*std, mean + z*std
    else:
        print(f"T is not of recognized type")
        lb,ub = False, False
    return lb, ub

def simul_ho_lee(r0,f_T,sigma,T,method = "euler",f = None,seed = None):
    if seed is not None:
        np.random.seed(seed)
    M = len(f_T)
    delta = T/M
    delta_sqrt = np.sqrt(delta)
    Z = np.random.standard_normal(M)
    if method == "exact":
        r, W = np.zeros(M), np.zeros(M)
        r[0] = r0
        for m in range(1,M):
            W[m] = W[m-1] + delta_sqrt*Z[m-1]
            r[m] = f[m] + 0.5*sigma**2*(m*delta)**2 + sigma*W[m]
    elif method == "euler":
        r = np.zeros(M)
        r[0] = r0
        for m in range(1,M):
            r[m] = r[m-1] + (f_T[m-1] + sigma**2*(m-1)*delta)*delta + sigma*delta_sqrt*Z[m-1]
    return r

def euro_option_price_ho_lee(K,T1,T2,p_T1,p_T2,sigma,type = "call"):
    sigma_p = sigma*(T2-T1)*np.sqrt(T1)
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def caplet_prices_ho_lee(strike,sigma,T,p):
    price_caplet = np.zeros([len(T)])
    for i in range(2,len(T)):
        price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*euro_option_price_ho_lee(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],sigma,type = "put")
    return price_caplet

# Hull-White Extended Vasicek
def theta_hwev(t,f,f_T,a,sigma):
    if type(t) == int or type(t) == float or type(t) == np.int32 or type(t) == np.int64 or type(t) == np.float64:
        theta = f_T + (sigma**2/a)*(np.exp(-a*t)-np.exp(-2*a*t)) + a*(f + 0.5*(sigma/a)**2*(1-np.exp(-a*t))**2)
    elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
        N = len(t)
        theta = np.zeros(N)
        for n in range(0,N):
            theta[n] = f_T[n] + (sigma**2/a)*(np.exp(-a*t[n])-np.exp(-2*a*t[n])) + a*(f[n] + 0.5*(sigma/a)**2*(1-np.exp(-a*t[n]))**2)
    return theta

def zcb_price_hwev(t,T,r,a,sigma,T_star,p_star,f_star):
    if type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        p = N*[None]
        for i in range(0,N):
            p_t = for_values_in_list_find_value_return_value(t,T_star,p_star)
            p_T = for_values_in_list_find_value_return_value(T[i],T_star,p_star)
            B = (1-np.exp(-a*(T[i]-t)))/a
            p[i] = (p_T/p_t)*np.exp(B*(f_star-r) - (sigma**2/(4*a))*B**2*(1-np.exp(-2*a*t)))
    elif type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        p_t = for_values_in_list_find_value_return_value(t,T_star,p_star)
        p_T = for_values_in_list_find_value_return_value(T,T_star,p_star)
        B = (1-np.exp(-a*(T-t)))/a

        p = (p_T/p_t)*np.exp(B*(f_star-r) - (sigma**2/(4*a))*B**2*(1-np.exp(-2*a*t)))
    return np.array(p)

def mean_var_hwev(a,sigma,T,f,f_T):
    N = len(T)
    mean, var, integral = np.zeros(N), np.zeros(N), np.zeros(N)
    mean[0] = f[0]
    for n in range(1,N):
        var[n] = sigma**2*(1-np.exp(-2*a*(T[n]-T[0])))/(2*a)
        integral[n] = integral[n-1] + 0.5*np.exp(a*T[n-1])*(f_T[n-1] + (sigma**2/a)*(np.exp(-a*T[n-1])-np.exp(-2*a*T[n-1])) + a*(f[n-1] + 0.5*(sigma/a)**2*(1-np.exp(-a*T[n-1]))**2))*(T[n]-T[n-1]) + 0.5*np.exp(a*T[n])*(f_T[n] + (sigma**2/a)*(np.exp(-a*T[n])+np.exp(-2*a*T[n])) + a*(f[n] + 0.5*(sigma/a)**2*(1-np.exp(-a*T[n]))**2))*(T[n]-T[n-1])
        # integral[n] = integral[n-1] + 0.5*np.exp(a*T[n-1])*(f_T[n-1] + (sigma**2/a)*(np.exp(-a*T[n-1])+np.exp(-2*a*T[n-1])) + a*(f[n-1] + 0.5*(sigma/a)**2*(1-np.exp(-a*T[n-1]))**2))*(T[n]-T[n-1]) + 0.5*np.exp(a*T[n])*(f_T[n] + (sigma**2/a)*(np.exp(-a*T[n])+np.exp(-2*a*T[n])) + a*(f[n] + 0.5*(sigma/a)**2*(1-np.exp(-a*T[n]))**2))*(T[n]-T[n-1])
        mean[n] = np.exp(-a*(T[n]-T[0]))*f[0] + np.exp(-a*T[n])*integral[n]
    return mean, var

def ci_hwev(a,sigma,T,f,f_T,size_ci,type_ci = "two_sided"):
    mean, var = mean_var_hwev(a,sigma,T,f,f_T)
    N = len(T)
    lb, ub = np.zeros([N]), np.zeros([N])
    if type_ci == "lower":
        z = norm.ppf(size_ci,0,1)
        for n in range(0,N):
            lb[n], ub[n] = mean[n] - z*np.sqrt(var[n]), np.inf
    elif type_ci == "upper":
        z = norm.ppf(size_ci,0,1)
        for n in range(0,N):
            lb[n], ub[n] = -np.inf, mean[n] + z*np.sqrt(var[n])
    elif type_ci == "two_sided":
        z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
        for n in range(0,N):
            lb[n], ub[n] = mean[n] - z*np.sqrt(var[n]), mean[n] + z*np.sqrt(var[n])
    return lb, ub

def simul_hwev(r0,t,theta,a,sigma,method = "euler",seed = None):
    if seed is not None:
        np.random.seed(seed)
    M = len(t)
    delta = t[-1]/M
    delta_sqrt = np.sqrt(delta)
    Z = np.random.standard_normal(M)
    if method == "euler":
        r = np.zeros(M)
        r[0] = r0
        for m in range(1,M):
            r[m] = r[m-1] + (theta[m-1] - a*r[m-1])*delta + sigma*delta_sqrt*Z[m-1]
    return r

def euro_option_price_hwev(K,T1,T2,p_T1,p_T2,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(T2-T1)))*np.sqrt((1-np.exp(-2*a*T1))/(2*a))
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def caplet_prices_hwev(strike,a,sigma,T,p):
    price_caplet = np.zeros([len(T)])
    for i in range(2,len(T)):
        price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*euro_option_price_hwev(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
    return price_caplet

# Libor market model
def drift_lmm(L,alpha,sigma,rho):
    N = len(L)
    drift = np.zeros(N)
    for i in range(0,N-1):
        for k in range(i+1,N):
            drift[i] += alpha[k]*L[k]/(1+alpha[k]*L[k])*sigma[i]*sigma[k]*rho[i,k]
    return drift

def simul_lmm(L0,T,sigma,rho,M):
    N = len(L0)
    alpha = np.zeros(N)
    for n in range(1,N):
        alpha[n-1] = T[n] - T[n-1]
    delta = T[1]*N/M
    delta_sqrt = np.sqrt(delta)
    log_L_simul = np.nan*np.ones([N,M+1])
    log_L_simul[:,0] = np.log(L0)
    stage = 0
    Mps = int(M/N)
    while stage < N:
        Z = np.random.standard_normal([N-stage,Mps])
        rho_sqrt = np.real(sqrtm(rho[stage:N,stage:N]))
        for m in range(0,Mps):
            drift = drift_lmm(log_L_simul[stage:N,stage*Mps+m],alpha[stage:N],sigma[stage:N],rho[stage:N,stage:N])
            log_L_simul[stage:N,stage*Mps+m+1] = log_L_simul[stage:N,stage*Mps+m] - (0.5*sigma[stage:N]**2 + drift)*delta + delta_sqrt*sigma[stage:N]*np.matmul(rho_sqrt,Z[:,m])
        stage += 1
    return np.exp(log_L_simul)

# Swap Market Model
def simul_smm(R0,T,sigma,rho,M,type = "regular"):
    N = len(R0) - 1
    delta = T[1]*(N+1)/M
    delta_sqrt = np.sqrt(delta)
    alpha = np.zeros(N+1)
    for n in range(1,N+2):
        alpha[n-1] = T[n] - T[n-1]
    rho_sqrt = np.real(sqrtm(rho))
    sigma = np.matmul(np.diag(sigma),rho_sqrt)
    R_simul = np.nan*np.ones([N+1,M+1])
    R_simul[:,0] = R0
    stage = 0
    Mps = int(M/(N+1))
    while stage < N+1:
        Z = np.random.standard_normal([N+1-stage,Mps])
        rho_sqrt = np.real(sqrtm(rho[stage:,stage:]))
        for m in range(1,Mps+1):
            t = delta*(stage*Mps + m - 1)
            alpha[stage] = T[stage+1] - t
            drift = drift_smm(R_simul[stage:,stage*Mps],sigma[stage:,stage:],alpha[stage:])
            R_simul[stage:,stage*Mps + m] = R_simul[stage:,stage*Mps + m-1] + drift*R_simul[stage:,stage*Mps + m-1]*delta + delta_sqrt*R_simul[stage:,stage*Mps + m-1]*np.matmul(sigma[stage:,stage:],Z[:,m-1])
        stage += 1
    return R_simul

def matrix_swap_to_p(alpha,R):
    N = len(R)
    M = np.zeros([N,N])
    for c in range(0,N):
        M[0,c] = R[0]*alpha[c]
    M[0,-1] += 1
    for r in range(1,N):
        M[r,r-1] = -1
        for c in range(r,N):
            M[r,c] = R[r]*alpha[c]
        M[r,-1] += 1
    return M

def zcb_prices_from_swap_rates_normal_smm(T,R_swap):
    N = len(R_swap)
    alpha = np.zeros(N)
    p = np.ones(N+1)
    for n in range(1,N+1):
        alpha[n-1] = T[n] - T[n-1]
    X = matrix_swap_to_p(alpha,R_swap)
    y = np.zeros([N])
    y[0] = 1
    p = np.linalg.solve(X,y)
    return p

def drift_smm(R,sigma,alpha):
    N = len(R) - 1
    drift = np.zeros([N+1])
    X = matrix_swap_to_p(alpha,R)
    y = np.zeros([N+1])
    y[0] = 1
    p = np.linalg.solve(X,y)
    S = np.zeros(N+1)
    for n in range(0,N+1):
        for j in range(0,N+1-n):
            S[n] += alpha[j]*p[j]
    phi = np.zeros([N,N+1])
    for n in range(0,N):
        for j in range(0,N-1):
            phi[n,:] += S[j+1]/S[n]*alpha[j+1]*R[j+1]*sigma[j+1,:]
            for k in range(n+1,j+1):
                phi[n] *= (1 + alpha[k]*R[k])
        drift[n] = - np.dot(sigma[n,:],phi[n,:])
    return drift

# Nelson-Siegel function
def F_ns(param,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        f_inf, a, b = param
        K = len(a)
        F = f_inf*T + a[0]*np.exp(-b[0]*T)
        for k in range(0,K):
            F += a[k]*b[k]**(-k-1)*gammainc(k+1,b[k]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        f_inf, a, b = param
        K = len(a)
        M = len(T)
        F = np.zeros([M])
        for m in range(0,M):
            F[m] = f_inf*T[m]
            for k in range(0,K):
                F[m] += a[k]*b[k]**(-k-1)*gammainc(k+1,b[k]*T[m])
    return F

def f_ns(param,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        f_inf, a, b = param
        K = len(a)
        f = f_inf
        for k in range(0,K):
            f += a[k]*T**k*np.exp(-b[k]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        f_inf, a, b = param
        K = len(a)
        M = len(T)
        f = np.zeros([M])
        for m in range(0,M):
            f[m] = f_inf
            for k in range(0,K):
                f[m] += a[k]*T[m]**k*np.exp(-b[k]*T[m])
    return f

def f_ns_jac(param,T):
    f_inf, a, b = param
    N = len(a) - 1
    jac = np.zeros([2*(N+1)+1])
    jac[0] = 1
    for n in range(0,N+1):
        jac[1+n] = T**n*np.exp(-b[n]*T)
        jac[1+N+1+n] = -a[n]*T**(n+1)*np.exp(-b[n]*T)
    return jac

def f_ns_hess(param,T):
    f_inf, a, b = param
    N = len(a) - 1
    hess = np.zeros([2*(N+1)+1,2*(N+1)+1])
    for n in range(0,N+1):
        hess[1+n,1+N+1+n] = - T**(n+1)*np.exp(-b[n]*T)
        hess[1+N+1+n,1+n] = - T**(n+1)*np.exp(-b[n]*T)
        hess[1+N+1+n,1+N+1+n] = a[n]*T**(n+2)*np.exp(-b[n]*T)
    return hess

def f_ns_T(param,T):
    if type(T) == int or type(T) == float or type(T) == np.float32 or type(T) == np.float64:
        a, b = param
        N = len(a)
        f_T = -a[0]*b[0]*np.exp(-b[0]*T)
        for n in range(1,N):
            f_T += a[n]*n*T**(n-1)*np.exp(-b[n]*T) - a[n]*b[n]*T**n*np.exp(-b[n]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        a, b = param
        N = len(a)
        M = len(T)
        f_T = np.zeros([M])
        for m in range(0,M):
            f_T[m] = -a[0]*b[0]*np.exp(-b[0]*T[m])
            for n in range(1,N):
                f_T[m] += a[n]*n*T[m]**(n-1)*np.exp(-b[n]*T[m]) - a[n]*b[n]*T[m]**n*np.exp(-b[n]*T[m])
    return f_T

def theta_ns(param,t):
    if type(t) == int or type(t) == float or type(t) == np.float32 or type(t) == np.float64:
        f_inf, a, b, sigma = param
        print(f_inf, a, b, sigma)
        K = len(a)
        theta = -a[0]*b[0]*np.exp(-b[0]*t) + sigma**2*t
        for k in range(1,K):
            theta += a[k]*k*t**(k-1)*np.exp(-b[k]*t) - a[k]*b[k]*t**k*np.exp(-b[k]*t)
    elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
        f_inf, a, b, sigma = param
        K = len(a)
        M = len(t)
        theta = np.zeros([M])
        for m in range(0,M):
            theta[m] = -a[0]*b[0]*np.exp(-b[0]*t[m]) + sigma**2*t[m]
            for k in range(1,K):
                theta[m] += a[k]*k*t[m]**(k-1)*np.exp(-b[k]*t[m]) - a[k]*b[k]*t[m]**k*np.exp(-b[k]*t[m])
    return theta

# Caplets
def black_caplet_price(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'put':
        price = alpha*p*(R*ndtr(-d2) - L*ndtr(-d1))
    else:
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    return price

def black_caplet_delta(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        delta = alpha*p*ndtr(d1) - alpha/(1+alpha*L)*price
    return delta

def black_caplet_gamma(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == "call":
        gamma = alpha*p*(norm.pdf(d1)/(L*sigma*np.sqrt(T))-2*alpha/((1+alpha*L)**2)*(alpha*R*ndtr(d2) + ndtr(d1)))
    return gamma

def black_caplet_vega(sigma,T,R,alpha,p,L,type = "call"):
    if type == "call":
        d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
        vega = alpha*p*L*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_caplet_theta(sigma,T,r,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        theta = r*price - alpha*p*(sigma*L*norm.pdf(d1))/(2*np.sqrt(T))
    return theta

def black_caplet_iv(C,T,R,alpha,p,L, iv0 = 0.2, max_iter = 200, prec = 1.0e-5):
    iv = iv0
    for i in range(0,max_iter):
        price = black_caplet_price(iv,T,R,alpha,p,L,type = "call")
        vega = black_caplet_vega(iv,T,R,alpha,p,L,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

# Swatiopns
def black_swaption_price(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'put':
        price = S*(K*ndtr(-d2) - R*ndtr(-d1))
    else:
        price = S*(R*ndtr(d1) - K*ndtr(d2))
    return price

def black_swaption_delta(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    # d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        # price = S*(R*ndtr(d1) - K*ndtr(d2))
        delta = S*ndtr(d1)  # - price/R + S*(R*norm.pdf(d1) - K*norm.pdf(d2))/(R*sigma*np.sqrt(T))
    return delta

def black_swaption_gamma(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        gamma = (2/R**2)*price + (S/R)*(norm.pdf(d1)/(sigma*np.sqrt(T)) - 2*ndtr(d1))
    return gamma

def black_swaption_vega(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        vega = S*R*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_swaption_theta(sigma,T,K,S,r,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        theta = r*price - S*R*sigma*norm.pdf(d1)/(2*np.sqrt(T))
    return theta

def black_swaption_iv(C,T,K,S,R,type = "call", iv0 = 0.2, max_iter = 1000, prec = 1.0e-10):
    iv = iv0
    for i in range(0,max_iter):
        price = black_swaption_price(iv,T,K,S,R,type = "call")
        vega = black_swaption_vega(iv,T,K,S,R,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

# SABR model
def sigma_sabr(K,T,F_0,sigma_0,beta,upsilon,rho,type = "call"):
    if abs(F_0-K) < 1e-8:    # SABR ATM formula
        sigma = sigma_0*F_0**(beta-1)*(1+(((1-beta)**2/24)*(sigma_0**2*(F_0)**(2*beta-2)) + (rho*beta*upsilon*sigma_0/4)*(F_0)**(beta-1) + (2-3*rho**2)/24*upsilon**2)*T)
    else:
        z = (upsilon/sigma_0)*(F_0*K)**((1-beta)/2)*np.log(F_0/K)
        x = np.log((np.sqrt(1-2*rho*z+z**2) + z - rho)/(1-rho))
        D = (F_0*K)**((1-beta)/(2))*(1 + ((1-beta)**2/24)*np.log2(F_0/K) + ((1-beta)**4/1920)*np.emath.logn(4,F_0/K))
        A = 1 + (((1-beta)**2/24)*sigma_0**2*(F_0*K)**(beta-1) + (rho*beta*upsilon*sigma_0/4)*(F_0*K)**((beta-1)/2) + ((2-3*rho**2)/24)*upsilon**2)*T
        sigma = (sigma_0/D)*(z/x)*A
    return sigma

def sabr_simul(F_0,sigma_0,beta,upsilon,rho,M,T):
    sigma, F = np.zeros([M+1]), np.zeros([M+1])
    sigma[0], F[0] = sigma_0, F_0
    delta = T/M
    Z = np.random.standard_normal([2,M])
    delta_sqrt = np.sqrt(delta)
    rho_sqrt = np.sqrt(1-rho**2)
    for m in range(1,M+1):
        F[m] = F[m-1] + sigma[m-1]*F[m-1]**beta*delta_sqrt*Z[0,m-1]
        sigma[m] = sigma[m-1] + upsilon*sigma[m-1]*delta_sqrt*(rho*Z[0,m-1] + rho_sqrt*Z[1,m-1])
    return F, sigma

def fit_sabr_obj(param,sigma_market,K,T,R):
    sigma_0, beta, upsilon, rho = param
    N = len(sigma_market)
    sse = 0
    for n in range(0,N):
        sigma_model = sigma_sabr(K[n],T,R,sigma_0,beta,upsilon,rho,type = "call")
        sse += (sigma_market[n]-sigma_model)**2
    return sse

# Statistics
def data_into_bins(data,N_bins,bin_min = "default",bin_max = "default"):
    # Divide the data in a one-simensional np.array into N_bins bins of equal size.
    N = len(data)
    data = np.sort(data,kind = "stable")
    if bin_min == "default":
        bin_min = data[0]
    if bin_max == "default":
        bin_max = data[-1]
    limits_bins = np.array([bin_min + i*(bin_max-bin_min)/N_bins for i in range(0,N_bins+1)])
    freq = np.zeros([N_bins])
    data_bins = []
    i, j = 0, 0
    while i < N_bins:
        freq_temp = 0
        data_temp = []
        while data[j] <= limits_bins[i+1]:
            data_temp.append(data[j])
            freq_temp += 1
            if j < N - 1:
                j += 1
            elif j == N - 1:
                break
        data_bins.append(np.array(data_temp))
        freq[i] = freq_temp
        i += 1
    return np.array(data_bins), limits_bins, freq

# List operations
def find_value_return_value(val,L1,L2,precision = 10e-8):
    # This function searches for 'val' in 'L1' and returns index 'idx' of 'val' in 'L1' and 'L2[idx]'.
    Ind, output = False, []
    for idx, item in enumerate(L1):
        if abs(val-item) < precision:
            Ind = True
            output.append((idx,L2[idx]))
    return Ind, output

def for_values_in_list_find_value_return_value(L1,L2,L3,precision = 10e-8):
    # For all 'item' in L1, this function searches for 'item' in L2 and returns the value corresponding to same index from 'L3'.
    if type(L1) == int or type(L1) == float or type(L1) == np.float64 or type(L1) == np.int32 or type(L1) == np.int64:
        output = None
        Ind, output_temp = find_value_return_value(L1,L2,L3,precision)
        if Ind == True:
            output = output_temp[0][1]
    elif type(L1) == tuple or type(L1) == list or type(L1) == np.ndarray:
        output = len(L1)*[None]
        for i, item in enumerate(L1):
            Ind, output_temp = find_value_return_value(item,L2,L3,precision)
            if Ind == True:
                output[i] = output_temp[0][1]
    return output

# ZCB curvefitting
def zcb_curve_fit(data_input,interpolation_options = {"method": "linear"},scaling = 1):
    data = copy.deepcopy(data_input)
    data_known = []
    libor_data, fra_data, swap_data = [], [], []
    # Separateing the data and constructing data_known from fixings
    for item in data:
        if item["instrument"] == "libor":
            libor_data.append(item)
            data_known.append({"maturity":item["maturity"],"rate":np.log(1+item["rate"]*item["maturity"])/item["maturity"]})
        elif item["instrument"] == "fra":
            fra_data.append(item)
        elif item["instrument"] == "swap":
            swap_data.append(item)
    # Adding elements to data_knwon based on FRAs
    I_done = False
    while I_done == False:
        for fra in fra_data:
            I_exer, known_exer = value_in_list_of_dict_returns_I_idx(fra["exercise"],data_known,"maturity")
            I_mat, known_mat = value_in_list_of_dict_returns_I_idx(fra["maturity"],data_known,"maturity")
            if I_exer == True and I_mat == False:
                data_known.append({"maturity":fra["maturity"],"rate":(known_exer["rate"]*known_exer["maturity"]+np.log(1+(fra["maturity"]-fra["exercise"])*fra["rate"]))/fra["maturity"]})
                I_done = False
                break
            if I_exer == False and I_mat == True:
                pass
            if I_exer == True and I_mat == True:
                pass
            else:
                I_done = True
    T_known, T_swap, T_knot = [], [], []
    R_known = []
    # Finding T's and corresponding R's where there is some known data
    for known in data_known:
        T_known.append(known["maturity"])
        R_known.append(known["rate"])
    # Finding T_swap - The times where there is a cashflow to at least one of the swaps.
    for swap in swap_data:
        T_knot.append(swap["maturity"])
        if swap["float_freq"] == "quarterly":
            if value_in_list_returns_I_idx(0.25,T_known)[0] == False and value_in_list_returns_I_idx(0.25,T_swap)[0] == False:
                T_swap.append(0.25)
        elif swap["float_freq"] == "semiannual":
            if value_in_list_returns_I_idx(0.5,T_known)[0] == False and value_in_list_returns_I_idx(0.5,T_swap)[0] == False:
                T_swap.append(0.5)
        elif swap["float_freq"] == "annual":
            if value_in_list_returns_I_idx(1,T_known)[0] == False and value_in_list_returns_I_idx(1,T_swap)[0] == False:
                T_swap.append(1)
        if swap["fixed_freq"] == "quarterly":
            for i in range(1,4*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.25,T_known)[0] == False and value_in_list_returns_I_idx(i*0.25,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.25,T_swap)[0] == False:
                    T_swap.append(i*0.25)
        elif swap["fixed_freq"] == "semiannual":
            for i in range(1,2*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.5,T_known)[0] == False and value_in_list_returns_I_idx(i*0.5,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.5,T_swap)[0] == False:
                    T_swap.append(i*0.5)
        elif swap["fixed_freq"] == "annual":
            for i in range(1,swap["maturity"]):
                if value_in_list_returns_I_idx(i,T_known)[0] == False and value_in_list_returns_I_idx(i*1,T_knot)[0] == False and value_in_list_returns_I_idx(i,T_swap)[0] == False:
                    T_swap.append(i)
    # Finding T_fra and T_endo
    T_endo, T_fra = [], []
    fra_data.reverse()
    for fra in fra_data:
        if value_in_list_returns_I_idx(fra["maturity"],T_known)[0] == False:
            I_fra_mat, idx_fra_mat = value_in_list_returns_I_idx(fra["maturity"],T_fra)
            I_endo_mat, idx_endo_mat = value_in_list_returns_I_idx(fra["maturity"],T_endo)
            if I_fra_mat is False and I_endo_mat is False:
                T_fra.append(fra["maturity"])
            elif I_fra_mat is True and I_endo_mat is False:
                pass
            elif I_fra_mat is False and I_endo_mat is True:
                pass
            elif I_fra_mat is True and I_endo_mat is True:
                T_fra.pop(idx_fra_mat)
        if value_in_list_returns_I_idx(fra["exercise"],T_known)[0] == False:
            I_fra_exer, idx_fra_exer = value_in_list_returns_I_idx(fra["exercise"],T_fra)
            I_endo_exer, idx_endo_exer = value_in_list_returns_I_idx(fra["exercise"],T_endo)
            if I_fra_exer is False and I_endo_exer is False:
                T_endo.append(fra["exercise"])
            elif I_fra_exer is True and I_endo_exer is False:
                T_fra.pop(idx_fra_exer)
                T_endo.append(fra["exercise"])
            elif I_fra_exer is False and I_endo_exer is True:
                pass
            elif I_fra_exer is True and I_endo_exer is True:
                T_fra.pop(idx_fra_exer)
    fra_data.reverse()
    # Fitting the swap portion of the curve
    T_swap_fit = T_known + T_swap + T_knot
    T_swap_fit.sort(), T_fra.sort(), T_endo.sort()
    R_knot_init = [None]*len(swap_data)
    for i, swap in enumerate(swap_data):
        indices = []
        if swap["fixed_freq"] == "quarterly":
            for j in range(1,4*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.25,T_swap_fit)[1])
        elif swap["fixed_freq"] == "semiannual":
            for j in range(1,2*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.5,T_swap_fit)[1])
        elif swap["fixed_freq"] == "annual":
            for j in range(1,swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j,T_swap_fit)[1])
        swap["indices"] = indices
        R_knot_init[i] = swap["rate"]
        i += 1
    args = (T_known,T_knot,T_swap_fit,R_known,swap_data,interpolation_options,1)
    result = minimize(zcb_curve_swap_fit_obj,R_knot_init,method = 'nelder-mead',args = args,options={'xatol': 1e-6,'disp': False})
    T_swap_curve, R_swap_curve = T_known + T_knot, R_known + list(result.x)
    T_fra_fit = T_swap_curve + T_fra + T_endo
    T_fra_fit.sort()
    R_fra_fit, R_fra_fit_deriv = interpolate(T_fra_fit,T_swap_curve,R_swap_curve,interpolation_options)
    R_fra_init = [None]*len(T_fra)
    for i in range(0,len(T_fra)):
        R_fra_init[i] = R_fra_fit[value_in_list_returns_I_idx(T_fra[i],T_fra_fit)[1]]
    args = (T_fra,T_known,T_endo,T_fra_fit,R_fra_fit,fra_data,interpolation_options,scaling)
    result = minimize(zcb_curve_fra_fit_obj,R_fra_init,method = 'nelder-mead',args = args,options={'xatol': 1e-6,'disp': False})
    R_fra = list(result.x)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra_fit)):
        I_fra, idx_fra = value_in_list_returns_I_idx(T_fra_fit[i],T_fra)
        if I_fra is True:
            R_fra_fit[i] = R_fra[idx_fra]
        elif I_fra is False:
            I_endo, idx_endo = value_in_list_returns_I_idx(T_fra_fit[i],T_endo)
            if I_endo is True:
                R_fra_fit[i] = R_endo[idx_endo]
    return np.array(T_fra_fit), np.array(R_fra_fit)

def zcb_curve_interpolate(T_inter,T,R,interpolation_options = {"method":"linear"}):
    N = len(T_inter)
    p_inter = np.ones([N])
    R_inter = np.zeros([N])
    f_inter = np.zeros([N])
    R_inter, R_inter_deriv = interpolate(T_inter,T,R,interpolation_options = interpolation_options)
    for i in range(0,N):
        f_inter[i] = R_inter[i] + R_inter_deriv[i]*T_inter[i]
        p_inter[i] = np.exp(-R_inter[i]*T_inter[i])
    return p_inter, R_inter, f_inter, T_inter

def extrapolate(x_extra,x,y,extrapolation_options = {"method":"linear"}):
    # Extrapoltion of value corresponding to a choice of x_extra
    if extrapolation_options["method"] == "linear":
        if x_extra < x[0]:
            a = (y[1]-y[0])/(x[1]-x[0])
            b = y[0]-a*x[0]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        elif x[-1] < x_extra:
            a = (y[-1]-y[-2])/(x[-1]-x[-2])
            b = y[-1]-a*x[-1]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "hermite":
        if x_extra < x[0]:
            coefs = hermfit(x[0:extrapolation_options["degree"]+1],y[0:extrapolation_options["degree"]+1],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        elif x[-1] < x_extra:
            coefs = hermfit(x[-extrapolation_options["degree"]-1:],y[-extrapolation_options["degree"]-1:],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "nelson_siegel":
        if x_extra < x[0]:
            x1, x2 = x[1]-x[0], x[2]-x[0]
            coefs = nelson_siegel_coef(x1,x2,y[0],y[1],y[2])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[0])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[0]))
        elif x[-1] < x_extra:
            x1, x2 = x[-2]-x[-3], x[-1]-x[-3]
            coefs = nelson_siegel_coef(x1,x2,y[-3],y[-2],y[-1])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[-3])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[-3]))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    return y_extra, y_extra_deriv

def interpolate(x_inter,x,y,interpolation_options = {"method":"linear", "transition": None}):
    N, M = len(x_inter), len(x)
    y_inter, y_inter_deriv = np.nan*np.ones([N]), np.nan*np.ones([N])
    if interpolation_options["method"] == "linear":
        coefs = np.nan*np.ones([M,2])
        for m in range(0,M-1):
            coefs[m,1] = (y[m+1]-y[m])/(x[m+1]-x[m])
            coefs[m,0] = y[m]-coefs[m,1]*x[m]
        coefs[M-1,1] = (y[M-1] - y[M-2])/(x[M-1]-x[M-2])
        coefs[M-1,0] = y[M-1] - coefs[M-1,1]*x[M-1]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        y_inter_deriv[n] = coefs[0,1]
                    elif idx == M-1:
                        y_inter_deriv[n] = coefs[M-1,1]
                    else:
                        y_inter_deriv[n] = 0.5*coefs[idx-1,1] + 0.5*coefs[idx,1]
                        y_inter_deriv[n] = coefs[idx,1]
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before, y_after = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n], coefs[idx_after,0] + coefs[idx_after,1]*x_inter[n]

                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*coefs[idx_before,1] + (1-w_before)*coefs[idx_after,1]
                    else:
                        y_inter[n] = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n]
                        y_inter_deriv[n] = coefs[idx_before,1]
    elif interpolation_options["method"] == "hermite":
        coefs = np.nan*np.ones([M,interpolation_options["degree"]+1])
        degrees = np.ones(M, dtype = "int")
        for m in range(0, M-1):
            left = min(int(interpolation_options["degree"]/2),m)
            right = min(M-1-m,int((interpolation_options["degree"]+1)/2))
            degrees[m] = left + right
            coefs[m,0:left+right+1] = hermfit(x[m-left:m+right+1],y[m-left:m+right+1],degrees[m])
        coefs[M-1], degrees[M-1] = coefs[M-2], degrees[M-2]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:degrees[idx]+1]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_after = hermval(x_inter[n],coefs[idx_after,0:degrees[idx_after]+1])
                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1])) + (1-w_before)*hermval(x_inter[n],hermder(coefs[idx_after,0:degrees[idx_after]+1]))
                    else:
                        y_inter[n] = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1]))
    elif interpolation_options["method"] == "nelson_siegel":
        coefs, type_fit = np.nan*np.ones([M,3]), M*[None]
        for m in range(1,M-1):
            if (y[m] > y[m-1] and y[m+1] > y[m]) or (y[m] < y[m-1] and y[m+1] < y[m]):
                coefs[m,0:3] = nelson_siegel_coef(x[m]-x[m-1],x[m+1]-x[m-1],y[m-1],y[m],y[m+1])
                type_fit[m] = "nelson_siegel"
            else:
                coefs[m,0:3] = hermfit(x[m-1:m+2],y[m-1:m+2],2)
                type_fit[m] = "hermite"
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        if type_fit[idx+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx]
                            y_inter_deriv[n] = -coefs[idx+1,1]*coefs[idx+1,2]*np.exp(-coefs[idx+1,2]*x_ns)
                        elif type_fit[idx+1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx+1,0:3]))
                    elif idx == M-1:
                        if type_fit[idx-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-2]
                            y_inter_deriv[n] = -coefs[idx-1,1]*coefs[idx-1,2]*np.exp(-coefs[idx-1,2]*x_ns)
                        elif type_fit[idx-1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx-1,0:3]))
                    else:
                        if type_fit[idx] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-1]
                            y_inter_deriv[n] = -coefs[idx,1]*coefs[idx,2]*np.exp(-coefs[idx,2]*x_ns)
                        elif type_fit[idx] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:3]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if idx_before == 0:
                        if type_fit[idx_before+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_before]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_ns), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                            y_inter_deriv[n] = -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                        elif type_fit[idx_before+1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                    elif idx_after == M-1:
                        if type_fit[idx_after-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_after-2]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after-1,0] + coefs[idx_after-1,1]*np.exp(-coefs[idx_after-1,2]*x_ns), -coefs[idx_after-1,1]*coefs[idx_after-1,2]*np.exp(-coefs[idx_after-1,2]*x_ns)
                        elif type_fit[idx_after-1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after-1,0:3]), hermval(x_inter[n],hermder(coefs[idx_after-1,0:3]))
                    else:
                        if interpolation_options["transition"] == "smooth":
                            w_left = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                            if type_fit[idx_before] == "nelson_siegel":
                                x_left = x_inter[n] - x[idx_before-1]
                                y_left, y_left_deriv = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_left), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_left)
                            elif type_fit[idx_before] == "hermite":
                                y_left, y_left_deriv = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
                            if type_fit[idx_after] == "nelson_siegel":
                                x_right = x_inter[n] - x[idx_after-1]
                                y_right, y_right_deriv = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_right), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_right)
                            elif type_fit[idx_after] == "hermite":
                                y_right, y_right_deriv = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                            y_inter[n], y_inter_deriv[n] = w_left*y_left + (1-w_left)*y_right, w_left*y_left_deriv + (1-w_left)*y_right_deriv
                        else:
                            if type_fit[idx_before] == "nelson_siegel":
                                x_ns = x_inter[n] - x[idx_before-1]
                                y_inter[n], y_inter_deriv[n] = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_ns), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_ns)
                            elif type_fit[idx_before] == "hermite":
                                y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
    return y_inter, y_inter_deriv

def nelson_siegel_coef(x1,x2,y0,y1,y2):
    alpha = (y0-y2)/(y0-y1)
    b_hat = 2*(alpha*x1-x2)/(alpha*x1**2-x2**2)
    result = minimize(nelson_siegel_coef_obj,b_hat,method = "nelder-mead",args = (alpha,x1,x2),options={'xatol': 1e-12,"disp": False})
    if type(result.x) == np.ndarray:
        b = result.x[0]
    elif type(result.x) == int or type(result.x) == int or type(result.x) == np.int32 or type(result.x) == np.int64 or type(result.x) == np.float64:
        b = result.x
    a = (y0-y1)/(1-np.exp(-b*x1))
    f_inf = y0 - a
    return f_inf, a, b

def nelson_siegel_coef_obj(b,alpha,x1,x2):
    se = (alpha-(1-np.exp(-b*x2))/(1-np.exp(-b*x1)))**2
    return se

def swap_indices(data,T):
    for item in data:
        if item["instrument"] == "swap":
            indices = []
            if item["fixed_freq"] == "quarterly":
                for i in range(1,4*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.25,T)[1])
            elif item["fixed_freq"] == "semiannual":
                for i in range(1,2*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.5,T)[1])
            elif item["fixed_freq"] == "annual":
                for i in range(1,item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i,T)[1])
            item["indices"] = indices
    return data

def R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data):
    R_fra.reverse(), T_fra.reverse()
    R_endo = [None]*len(T_endo)
    for i in range(0,len(T_fra)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_fra[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_fra[i]*T_fra[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_fra.reverse(), T_fra.reverse()
    R_endo.reverse(), T_endo.reverse()
    for i in range(0,len(T_endo)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_endo[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_endo[i]*T_endo[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_endo.reverse(), T_endo.reverse()
    return R_endo

def zcb_curve_fra_fit_obj(R_fra,T_fra,T_known,T_endo,T_all,R_all,fra_data,interpolation_options,scaling = 1):
    sse = 0
    R_fra = list(R_fra)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra)):
        if T_fra[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_fra[i],T_all)[1]] - R_fra[i])**2
    for i in range(0,len(T_endo)):
        if T_endo[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_endo[i],T_all)[1]] - R_endo[i])**2
    sse *= scaling
    return sse

def zcb_curve_swap_fit_obj(R_knot,T_known,T_knot,T_all,R_known,swap_data,interpolation_options,scaling = 1):
    sse = 0
    R_knot = list(R_knot)
    R_all, R_deriv = interpolate(T_all,T_known + T_knot,R_known + R_knot,interpolation_options)
    p = zcb_prices_from_spot_rates(T_all,R_all)
    for n, swap in enumerate(swap_data):
        if swap["fixed_freq"] == "quarterly":
            alpha = 0.25
        if swap["fixed_freq"] == "semiannual":
            alpha = 0.5
        if swap["fixed_freq"] == "annual":
            alpha = 1
        S_swap = 0
        for idx in swap["indices"]:
            S_swap += alpha*p[idx]
        R_swap = (1 - p[swap["indices"][-1]])/S_swap
        sse += (R_swap - swap["rate"])**2
    sse *= scaling
    return sse

def value_in_list_returns_I_idx(value,list,precision = 1e-12):
    output = False, None
    for i, item in enumerate(list):
        if abs(value-item) < precision:
            output = True, i
            break
    return output

def idx_before_after_in_iterable(value,list):
    idx_before, idx_after = None, None
    if value < list[0]:
        idx_before, idx_after = None, 0
    elif list[-1] < value:
        idx_before, idx_after = len(list) - 1, None
    else:
        for i in range(0,len(list)):
            if list[i] < value:
                idx_before = i
            elif list[i] > value:
                idx_after = i
                break
    return idx_before, idx_after

def value_in_list_of_dict_returns_I_idx(value,L,name,precision = 1e-12):
    output = False, None
    for item in L:
        if abs(value-item[name]) < precision:
            output = True, item
            break
    return output

# Fitting the initial term structure of forward rates (For use in the Ho-Lee and Hull-White extended Vasicek models)
def theta(t,sigma,args):
    if args["model"] == "nelson-siegel":
        a = args["a"]
        b = args["b"]
        if type(t) == int or type(t) == float or type(t) == np.int32 or type(t) == np.int64 or type(t) == np.float64:
            K = len(a)
            theta = -a[0]*b[0]*np.exp(-b[0]*t) + sigma**2*t
            for k in range(1,K):
                theta += a[k]*k*t**(k-1)*np.exp(-b[k]*t) - a[k]*b[k]*t**k*np.exp(-b[k]*t)
        elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
            K = len(a)
            M = len(t)
            theta = np.zeros([M])
            for m in range(0,M):
                theta[m] = -a[0]*b[0]*np.exp(-b[0]*t[m]) + sigma**2*t[m]
                for k in range(1,K):
                    theta[m] += a[k]*k*t[m]**(k-1)*np.exp(-b[k]*t[m]) - a[k]*b[k]*t[m]**k*np.exp(-b[k]*t[m])
    if args["model"] == "interpolation":
        # Takes a vector theta on some grid T an expands that vector to the grid in t by linear interpolation
        T = args["T"]
        theta_star = args["theta_star"]
        M, N = len(t), len(T)
        theta = np.zeros([M])
        i, j = 0, 0
        while i < M:
            while j < N:
                if t[i] < T[j]:
                    print(f"WARNING! Not able to compute theta for t: {t[i]}. t less than T, t: {t[i]}, T: {T[j]}")
                    i += 1
                elif T[j] <= t[i] <= T[j+1]:
                    w_right = (t[i] - T[j])/(T[j+1]-T[j])
                    theta[i] = w_right*theta_star[j+1] + (1-w_right)*theta_star[j]
                    if i + 1 > M - 1:
                        j = N
                    i += 1
                elif t[i] > T[j+1]:
                    if j + 1 > N - 1:
                        print(f"WARNING! Not able to compute theta for t: {t[i]}. t greater than T, t: {t[i]}, T: {T[j]}")
                    else:
                        j += 1
    return theta