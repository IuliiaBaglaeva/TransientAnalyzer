# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import windows, find_peaks
from scipy.ndimage import convolve1d
from scipy.optimize import minimize, bisect
from copy import deepcopy
from scipy.interpolate import CubicSpline, interp1d
import gpflow
import re
from pybaselines import Baseline
from .LinearGibbs import LinearGibbs
from .EMOneDimGaussian import EMOneDimGaussian


class TransientAnalyzer:
    """The Class of **TransientAnalyzer** is used to analyze transient signals (calcium transients, spikes, sparks, contractions, etc.).
   This class can robustly analyze individual noisy signals. TransientAnalyzer is based on the Gaussian process 
   with the **Linear Gibbs kernel** of the covariance matrix. The class takes Numpy Arrays of time and signals   as input. 
   TransientAnalyser includes automatic detection of the starting points of the transients, approximation of transient using Gaussian process regression (GPR) and
   determination of signal parameters describing amplitude and kinetics of individual transients. Automatic detection can be superseded by the input of stimulus time points as Numpy array.
   
    There are two parameters which regulate detection of onset times: window_size (default to 20) and prominence(defaults to 1) that can improve the detection of starting point in case of problems.
    In case of wrong detection of starting point (some kind of displacement) change the window_size. If the number of detection transients is higher (lower) than it actually is (it can be in case of the high noise),
    decrease (or increase) the prominence. There is also possibility to provide stimuli times (t_stim parameter) instead of automatic detection.

GPR is a type of Bayesian Regression with the prior distribution being a multivariate Gaussian distribution with an arbitrary mean (usually set to 0) 
and a total N-dimensional covariance matrix K with the elements:
   .. math:: K(x, y) = k(x, y)+ \sigma^2 \delta(x, y),

   where ``k(x, y)`` is the covariance matrix, ``σ`` is the noise amplitude and ``δ(x,y)`` is the Kronecker delta.

   Considering the Bayesian framework the aim is to find the posterior distribution, i.e.the prior multivariate Gaussian distribution conditioned on the observed data.
   The covariance matrix ``k(x, y)`` defines the mutual relationship between all points of the true signal.

    Since transients signals are non-stationary processes that most often have a fast rise and a slower decay, the kernel of GP being used is :class:`LinearGibbs`
    
    The class provides the following parameters:
     * Baseline;
     * Amplitude;
     * Rise time at percentiles (*x-(100-x)%*) defined by user (20-80%, 10-90%, etc.);
     * Time-to-peak (*TTP*);
     * FDHM - Full Duration at Half Maximum
     * Decay time at percentiles (*(100-x)-x%*) defined by user (80-20%, 90-10%, etc.);
     * Transients durations at percentiles (*100-x%*) defined by user (80%, 90%, etc.);    
     """

    def __init__(self, time, Sig, start_gradient = 0, kernel="Gibbs",
                 window_size=20, window_size2 = 0, prominence=1, t_stim=None,
                 detrend = False, alpha_mult = 1, beta = 0.25, shift = 0, 
                 quantile1=0.1, quantile2=0.2, is_fall = None):
        """
        
        :param time: array of the values of the time during contraction
        :type time: array_like
        :param Sig: array of the values of the transients
        :type Sig: array_like
        :param start_gradient: starting point for the heavy ball algorithm used for the transient's start detection
        :type start_gradient: float, optional
        :param kernel: kernel used for analyze, can be chosen between RBF and Gibbs, defaults to "Gibbs"
        :type kernel: str, optional
        :param window_size: size of box filter window for transients start detection, can be modified in case of the bad detection of transients start, defaults to 20
        :type window_size: int, optional
        :param window_size2: size of box filter window for smoothing transient in order to detect the transients start. The values less or equal to zero mean no usage of the filter. defaults to 0
        :type window_size2: float, optional
        :param prominence: measures how much a peak stands out from the surrounding baseline, can be used when no or extra transients are detected, defaults to 1
        :type prominence: float, optional
        :param t_stim: stimulation times, defaults to None
        :type t_stim: array_like, optional
        :param detrend: defines whether to detrend the data or not, defaults to False
        :type detrend: bool
        :param alpha_mult: defines the multiplier of learning rate, important to change if the estimated t0 goes to infinitively large values, defaults to 1
        :type alpha_mult: float
        :param beta: "inertion" parameter used in transients start detection, defaults to 0.25
        :type beta: float
        :param shift: shift(in data points) to the left so that the estimated starting time is before the actual start of transient
        :type shift: int
        :param quantile1: the first quantile for parameters of transients, will be used for the detection of rise,decay times and durations as the ones between quantile1 and 1 - quantile1 percents of the corresponding transient phase,  defaults to 0.1
        :type quantile1: float
        :param quantile2: the second quantile for parameters of transients, will be used for the detection of rise,decay times and durations as the ones between quantile1 and 1 - quantile1 percents of the corresponding transient phase,  defaults to 0.2
        :type quantile2: float
        :param is_fall: defines whether the transients are falling or not. If the parameter is None, it will be defined automatically by the gaussian mixture model. Defaults to None
        :type is_fall: bool
        """
        self.transients = []
        self.parameters = []
        self.dt = 0
        self.t_stim = t_stim
        self.baselines = []
        self.t0s = []
        self.t0s_est = []
        self._window_size = window_size
        self._prominence = prominence
        self._n_baseline = 80
        self.frac_spline = 0.04
        self._kernel = None
        self._window_size2 = window_size2
        self._beta = beta
        self._shift = shift
        self._alpha_mult = alpha_mult
        if is_fall is None:
            model = EMOneDimGaussian()
            model.fit(Sig)
            idx_max = np.argmax(model._w[0])
            if model._mu[0][idx_max] > model._mu[0][(idx_max + 1) % 2]:
                self.is_fall = True
            else:
                self.is_fall = False
        else:
            self.is_fall = is_fall
        self._SetData(time, Sig, t_stim,detrend)
        if start_gradient < 0:
            self._start_gradient = 2 * np.min(np.diff(self.Time))
        else:
            self._start_gradient = start_gradient
        if kernel == "Gibbs":
            self._kernel = LinearGibbs(np.min(np.diff(self.Time)) * 0.5)
        else:
            self._kernel = gpflow.kernels.RBF()
        self.quantile1 = quantile1
        self.quantile2 = quantile2

    def _SetData(self, time, Sig, t_stim, detrend):
        """Sets data provided by the user (private method)

        :param time: array of the values of the time during contraction
        :type time: array_like
        :param Sig: array of the values of the transients
        :type Sig: array_like
        :param t_stim: stimulation times, defaults to None
        :type t_stim: array_like, optional

        """
        same_length = (len(time) == len(Sig))
        if time is None or Sig is None or not same_length:
            raise Exception("One of the arrays is not defined or they do not have the same length")
        if detrend:
            baseline_fitter = Baseline(x_data=time)
            imodpoly = baseline_fitter.imodpoly(Sig, poly_order=4)[0]
            Sig = Sig - imodpoly + imodpoly[0]
        self.Time = np.copy(time)
        self.Sig = np.copy(Sig)
        self._FindEstAllT0(t_stim)
        return

    def SetWindowSize(self, value):
        """SetWindowSize changes window size of box filter window

        :param value: window size of the box filter
        :type value: int
        """
        self._window_size = int(value)
        self._FindEstAllT0(self.t_stim)

    def SetProminence(self, value):
        """SetProminence allows to change the parameter prominence that measures how much a peak stands out from the surrounding baseline

        :param value: prominence
        :type value: int
        """
        self._prominence = value
        self._FindEstAllT0(self.t_stim)

    def GetExpTransient(self, idx, output=True):
        """ GetExpTransient allows to see one of experimental transients

        :param idx: index of the signal
        :type value: int
        :param output: defines whether the baseline of the transient is also provided in the signal array, applies only to the first transient, defaults to True
        :type value: bool, optional
        :return: two 1-d arrays, which contain time and signal
        :rtype: array-like, array-like and array-like or None
        """
        if idx == 0 and output:
            num1 = 0
        else:
            num1 = self.borders[idx]
        num2 = self.borders[idx + 1]
        return np.copy(self.Time[num1:num2]), np.copy(self.Sig[num1:num2])

    def GetAllExpTransients(self):
        """ GetAllExpTransients allows to see all transients which should be analysed

        :return: array of experimental transients and time
        :rtype: array
        """
        time_out = []
        sig_out = []
        for i in range(len(self.borders) - 1):
            t, c = self.GetExpTransient(i)
            time_out.append(t)
            sig_out.append(c)
        return np.array(time_out, dtype=object), np.array(sig_out, dtype=object)

    def _MinFunc(self, x, f, is_rise):
        """_MinFunc allows to find maximum of the transient

        :param x: array of time
        :type x: array_like
        :param f: signal to be maximized or minimized
        :type f: UnivariateSpline
        :param is_rise: binary value equals to 1 if it is rising transient signal (so the algorithm tries to find maximum) and 0 if not (to find minimum)
        :type is_rise: bool
        :return: f(x)
        :rtype: array_like
        """
        return (1 - 2 * is_rise) * f(np.array(x))[0]

    def _FindTimeFraction(self, x, alpha, ymax, ymin, f):
        """_FindTimeFraction aims to find time which corresponds to the quantile of the signal

        :param x: time
        :type x: array_like
        :param alpha: quantile
        :type alpha: float
        :param ymax: peak of the signal
        :type ymax: float
        :param ymin: baseline of the signal
        :type ymin: float
        :param f: signal's spline
        :type f: UnivariateSpline
        :return: value to be minimized
        :rtype: array_like
        """
        return f(x) - alpha * (ymax - ymin) - ymin

    def _FindEstAllT0(self, t_stim):
        """_FindEstAllT0 finds all estimated t0
        Onset time (t0) is determined as a peak (using scipy.signal.find_peaks function) of the difference between the boxfiltered (using scipy.signal.windows.boxcar function)
        transient trace and the real signal.

        :param t_stim: possibility to use stimulation time
        :type t_stim: array_like or None

        """
        self.dt = (self.Time[1] - self.Time[0])
        if t_stim is None:
            if self._window_size2 <= 0:
                win2 = np.ones(1)
                init_shift = int(0)
            else:
                win2 = windows.boxcar(self._window_size2)
                init_shift = int((self._window_size2 - 1)//2)
            sig = convolve1d(self.Sig, win2, mode='mirror') / np.sum(win2)
            if self.is_fall:
                sig *= -1
            win = windows.boxcar(self._window_size)
            filtered = convolve1d(sig, win, mode='mirror') / np.sum(win)
            res = filtered - sig
            self.borders, _ = find_peaks(res, prominence=self._prominence * np.max(res), height=0)
            self.borders = self.borders.astype("int64") + init_shift - int(self._shift)
            self.t0s_est = self.Time[self.borders]
        else:
            self.t0s_est = deepcopy(t_stim)
            self.borders = []
            for t0 in self.t0s_est:
                self.borders.append(np.argmax(t0 <= self.Time))
            self.borders = np.array(self.borders, dtype = 'int64')
        self.t0s = [-1] * len(self.borders)
        self.baselines = [-1] * len(self.borders)
        self.transients = [-1] * len(self.borders)
        self.parameters = [-1] * len(self.borders)
        self.borders = np.append(self.borders, self.Time.shape[0] - 1).astype("uint64")
        n = len(self.borders) - 1
        for i in range(n):
            if i == 0:
                dn = self.borders[0]
                n0 = int(max(dn - self._n_baseline, 0))
                self.baselines[i] = np.mean(self.Sig[n0:dn])

    def _FindT0(self, t0, spline, mean):
        """_FindT0 aims to improve the onset time by finding an intersection between the GP and the baseline

        :param t0: time
        :type t0: float
        :param spline: GP spline
        :type spline: UnivariateSpline
        :param mean: baseline value
        :type mean: float
        :return: squared difference between value of GP and baseline at t0
        :rtype: array_like

        """
        res = spline(t0)[0] / mean(t0) - 1
        return res * res

    def _Gradient(self, func,x_start, args,alpha = 0.002, beta = 0.25):
        """Heavy ball method (Polyak, 1964)

        :param func: the function to be optimized
        :type func: callable
        :param x_start: starting point
        :type x_start: float
        :param alpha: parameter of gradient
        :type alpha: float
        :param beta: inertion parameter of the method
        :type beta: float
        :param args: list of arguments for the func
        :type args: tuple, optional
        :return: minimum of func and corresponding x (in inverse order)
        :rtype: float, float

        """
        dx = 1e-3 * self.dt
        x = x_start
        xprev = x
        xprevm1 = x
        f = func(x, *args)
        dydx = 0
        eps = 1e-6
        n_iter = 0
        fl = func(x - dx, *args)
        fr = func(x + dx, *args)
        dydx = (fr - fl) / (2 * dx)
        alpha *= 20/np.abs(dydx)
        while n_iter <= 10:
            fl = func(x - dx, *args)
            fr = func(x + dx, *args)
            dydx = (fr - fl) / (2 * dx)
            x = xprev - alpha * dydx + beta * (xprev - xprevm1)
            f = func(x, *args)
            if np.abs(x - xprev) < eps:
                n_iter += 1
            else:
                n_iter = 0
            xprevm1 = xprev
            xprev = x
        return x, f

    def _FitSingleTransient(self, idx):
        """_FitSingleTransient approximates one of chosen transient by Gaussian process (GP) and obtains the signal parameters which describe amplitude and kinetics
        of individual signal.

        :param idx: number of transient
        :type idx: int
        """
        self.parameters[idx] = []
        t, sig = self.GetExpTransient(idx, False)
        ti = t[0]
        t -= t[0]
        t_start = - self._start_gradient
        t_end = self._start_gradient
        Yfit = sig[::]
        if idx > 0:
            Yfit[0] = self.transients[idx-1](ti  - self.t0s_est[idx-1])
        else:
            Yfit[0] = self.baselines[idx]
        Xfit = t
        gpr = gpflow.models.GPR((Xfit.reshape(-1, 1), Yfit.reshape(-1, 1)), kernel=self._kernel)
        opt = gpflow.optimizers.Scipy()
        try:
            opt.minimize(gpr.training_loss, variables=gpr.trainable_variables)
        except:
            gpr.kernel.A = gpflow.Parameter(1.0, transform = gpflow.utilities.positive())
            opt.minimize(gpr.training_loss, variables=gpr.trainable_variables)
        if self.is_fall:
            maxy_estidx = np.argmin(Yfit)
        else:
            maxy_estidx = np.argmax(Yfit)
        t_max_est = Xfit[maxy_estidx]
        xp = np.arange(t_start, Xfit[-1], self.frac_spline * self.dt)
        y = np.zeros_like(xp)
        mean, _ = gpr.predict_y(xp.reshape(-1, 1))
        y = mean[:, 0]
        cs = CubicSpline(xp, y)
        self.transients[idx] = cs
        Peak_sig = minimize(self._MinFunc, x0=[t_max_est], bounds=((xp[0],xp[-1]),), args=(self.transients[idx], bool(np.abs(True - self.is_fall))))
        if self.is_fall:
            Peak_sig.fun *= - 1
        # find t0
        x_findt0 = np.linspace(t_start, min(Peak_sig.x[0],t_end), 100)
        if idx > 0:
            baseline_spline = CubicSpline(x_findt0,self.transients[idx-1](x_findt0 + ti - self.t0s_est[idx-1]))
        else:
            baseline_spline = CubicSpline(x_findt0,np.array([self.baselines[idx]] * len(x_findt0)))
        mean, _ = gpr.predict_y(x_findt0.reshape(-1, 1))
        t0_values = mean
        cs_t0 = CubicSpline(x_findt0, t0_values)
        t0_sig,_ = self._Gradient(self._FindT0, x_findt0[-1], args=(cs_t0,baseline_spline),beta = self._beta, alpha = 0.002 * self._alpha_mult)
        baseline = baseline_spline(t0_sig)
        self.t0s[idx] = t0_sig + ti
        Amp_sig = -Peak_sig.fun - baseline
        TTP_sig = Peak_sig.x[0] - t0_sig
        base_sig = baseline
        self.parameters[idx].append(baseline)
        self.parameters[idx].append(Amp_sig)
        self.parameters[idx].append(self.t0s[idx])
        if idx == 0:
            self.parameters[idx].append(np.nan)
        else:
            self.parameters[idx].append(self.t0s[idx] - self.t0s[idx - 1])
        if self.t_stim is None:
            self.parameters[idx].append(np.nan)
        else:
            self.parameters[idx].append(self.t0s[idx]-self.t0s_est[idx])
        # get rise time, FDHM, decay time
        rise_times = np.zeros(5)
        decay_times = np.zeros_like(rise_times)
        quantiles = np.array([self.quantile1,self.quantile2,0.5,1 - self.quantile2, 1 - self.quantile1])
        for i in range(5): #durations of rise phase
            try:
                rise_times[i] = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                                args=(quantiles[i], -Peak_sig.fun, base_sig, self.transients[idx]))
                self.parameters[idx].append(rise_times[i] - t0_sig)        
            except:
                rise_times[i] = None
                self.parameters[idx].append(np.nan)
        self.parameters[idx].append(TTP_sig)
        if self.is_fall:
            maxy_estidx2 = np.argmax(Yfit[maxy_estidx:])
        else:
            maxy_estidx2 = np.argmin(Yfit[maxy_estidx:])
        min_f_after_ttp = minimize(self._MinFunc, x0=[Xfit[maxy_estidx + maxy_estidx2]], bounds=((TTP_sig,t[-1]),), args=(self.transients[idx], bool(np.abs(False - self.is_fall))))
        for i in range(5):  #durations of decay phase
            try:
                decay_times[i] = bisect(self._FindTimeFraction, TTP_sig, min_f_after_ttp.x[0], 
                args=(1 - quantiles[i], -Peak_sig.fun, base_sig, self.transients[idx]))
                self.parameters[idx].append(decay_times[i] - t0_sig)
            except:
                decay_times[i] = None
                self.parameters[idx].append(np.nan)
        for i in reversed(range(2)):
            if rise_times[i] is not None and rise_times[4-i] is not None:
                self.parameters[idx].append(rise_times[4-i] - rise_times[i])  # rise times
            else:
                self.parameters[idx].append(np.nan)
        if rise_times[2] is not None and decay_times[2] is not None:
            self.parameters[idx].append(decay_times[2] - rise_times[2])
        else:
            self.parameters[idx].append(np.nan)
        for i in reversed(range(2)):
            if decay_times[i] is not None and decay_times[4-i] is not None:
                self.parameters[idx].append(decay_times[4-i] - decay_times[i])  # decay times
            else:
                self.parameters[idx].append(np.nan)
        if idx is not len(self.borders) - 2:
            self.baselines[idx + 1] = float(self.transients[idx](self.t0s_est[idx + 1] - self.t0s[idx]))

    def FitAllTransients(self, disp=False):
        """FitAllTransients fits transients

        :param disp: shows the progress of transients, defaults to False
        :type disp: bool, optional


        """
        for i in range(len(self.borders) - 1):
            self._FitSingleTransient(i)
            if disp:
                print(f"Transient #{i + 1} is completed.")

    def GetApproxTransient(self, idx, dt):
        """GetApproxTransient provides the approximated transient with the sampling rate provided by the user. Raises error when no approximation was done

        :param idx: index of transient
        :type idx: int
        :param dt: time step of the approximation transient
        :type dt: float
        :return: time and transients arrays
        :rtype: array-like, array-like and array-like or None

        """
        if self.transients[idx] == -1:
            raise Exception("No approximation was done. Run FitAllTransients() first.")
        t, _ = self.GetExpTransient(idx, False)
        if idx != len(self.t0s) - 1:
            T = np.arange(self.t0s[idx], self.t0s[idx + 1], dt)
        else:
            T = np.arange(self.t0s[idx], t[-1], dt)
        return T, self.transients[idx](T - t[0])

    def GetTransientsTable(self, x_label="x", y_label="y"):
        """GetTransientsTable provides the table with the input and approximated traces

        :param x_label: name of x axis
        :type x_label: str
        :param y_label: name of y axis
        :type y_label: str
        :return: Table with the transients
        :rtype: DataFrame
        """
        t_exp_out = np.array([])
        t_model_out = np.array([])
        sig_exp_out = np.array([])
        sig_model_out = np.array([])
        for i in range(len(self.t0s)):
            t, sig = self.GetExpTransient(i)
            if i == 0: #draw baseline
                t_bl, sig_bl = self.GetExpTransient(i, False)
                t_start = np.array([t[0],t_bl[0] - self.dt])
                y_start = np.array([self.baselines[i],self.baselines[i]])
                t_model_out = np.hstack((t_model_out, t_start))
                sig_model_out = np.hstack((sig_model_out, y_start))
            t_model, sig_model = self.GetApproxTransient(i, 0.2 * self.dt)
            t_exp_out = np.hstack((t_exp_out, t))
            t_model_out = np.hstack((t_model_out, t_model))
            sig_exp_out = np.hstack((sig_exp_out, sig))
            sig_model_out = np.hstack((sig_model_out, sig_model))
        sig_spline = interp1d(t_model_out, sig_model_out,fill_value = "extrapolate")
        sig_model_out = sig_spline(t_exp_out[t_exp_out >= t_model_out[0]])
        columns = [x_label, y_label, f"Approximated {y_label}"]
        output = np.zeros((t_exp_out.shape[0], 3))
        output.fill(np.nan)
        output[:, 0] = t_exp_out
        output[:, 1] = sig_exp_out
        output[t_exp_out >= t_model_out[0], 2] = sig_model_out
        df_out = pd.DataFrame(output, columns=columns[:3])
        return df_out

    def GetParametersTable(self, x_label="x", y_label="y"):
        """GetParametersTable allows to get the table with parameters.
       Table of parameters contains values of *t0* - the start time of the transient or of the stimulus, *delta t0* - the difference in t0 between two
       neighboring transients and Delay - the difference between stimulus time and the start of the transient.

        :param x_label: name of x axis
        :type x_label: str
        :param y_label: name of y axis
        :type y_label: str
        :return: the table with parameters
        :rtype: pandas DataFrame


        """
        q1 = int(self.quantile1 * 100)
        q2 = int(self.quantile2 * 100)
        # parse labels so that if they have specific form then use units in columns
        x_out = re.search(r",[\s\S]*$", x_label)
        if x_out is not None:
            x_out = x_out[0]
        else:
            x_out = ""
        y_out = re.search(r",[\s\S]*$", y_label)
        if y_out is not None:
            y_out = y_out[0]
        else:
            y_out = ""
        quantiles = np.array([q1,q2,50,100 - q2, 100 - q1])
        list_r = [f"r_{q}%{x_out}" for q in quantiles]
        list_d = [f"d_{q}%{x_out}" for q in quantiles]
        list_tr = [f"t_{100-q}%-{q}%{x_out}" for q in quantiles[3:]]
        list_td = [f"t_{q}%-{100-q}%{x_out}" for q in quantiles[3:]]
        columns = [f"Baseline{y_out}",  f"Amplitude{y_out}", f"t0{x_out}", f"delta t0{x_out}", f"Delay{x_out}"]
        columns.extend(list_r)
        columns.append(f"TTP{x_out}") 
        columns.extend(list_d)
        columns.extend(list_tr)
        columns.append(f"FDHM{x_out}")
        columns.extend(list_td)
        pars = np.array(self.parameters)
        df = pd.DataFrame(pars, columns=columns)
        return df

    def ParametersToExcel(self, filename, x_label="x", y_label="y"):
        """ParametersToExcel saves parameters to an excel file

        :param x_label: name of x axis
        :type x_label: str
        :param y_label: name of y axis
        :type y_label: str
        :param filename: name of the file which you want to save
        :type filename: string
        """
        df = self.GetParametersTable(x_label, y_label)
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        sheetname = "Parameters"
        df.to_excel(writer, sheet_name=sheetname)
        worksheet = writer.sheets[sheetname]
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max((series.astype(str).map(len).max(),  # len of largest item
                           len(str(series.name))  # len of column name/header
                           )) + 5  # adding a little extra space
            worksheet.set_column(idx + 1, idx + 1, max_len)  # set column width
        writer.save()
