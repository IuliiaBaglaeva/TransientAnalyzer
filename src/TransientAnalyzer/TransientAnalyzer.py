# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import windows, find_peaks
from scipy.optimize import minimize, bisect
from copy import deepcopy
from scipy.interpolate import CubicSpline, interp1d
import gpflow
from LinearGibbs import LinearGibbs
import re


class TransientAnalyzer:
    """The Class of **TransientAnalyzer** is used to analyze transient signals (calcium transients, spikes, sparks, contractions, etc.).
   This class is able to analyse robustly individual noisy signals. TransientAnalyzer is based on Gaussian process with the
   **Linear Gibbs kernel** of covariance matrix. Numpy Arrays of signals and time are an input to the class.
   TransientAnalyser includes automatic detection of starts of transient *t0*, approximation of transient using Gaussian process regression (GPR) and
   the obtainment of signal parameters, describing amplitude and kinetics of individual signal. There is also possibility to provide stimuli time points instead of automatic detection.
   GPR is a type of Bayesian Regression with the prior distribution being a multivariate Gaussian distribution with an arbitrary mean (usually set to 0) and a total N-dimensional covariance matrix K with the elements:

   .. math:: K(x, y) = k(x, y)+ \sigma^2 \delta(x, y),

   where ``k(x, y)`` is the covariance matrix, ``σ`` is the noise amplitude and ``δ(x,y)`` is the Kronecker delta.

   Considering the Bayesian framework the aim is to find the posterior distribution - the prior multivariate Gaussian distribution
   conditioned on the observed data.
   The covariance matrix ``k(x, y)`` defines the mutual relationship between all points of the true signal.

    Since transients signals are non-stationary processes with fast rise and slow decay the kernel of GP being used is :class:`LinearGibbs`
    """

    def __init__(self, time, Sig, kernel="Gibbs",
                 window_size=20, prominence=1, t_stim=None,
                 quantile1=0.1, quantile2=0.2):
        """
        There are two parameters which regulate detection of onset times: window_size (default to 20) and prominence(defaults to 1) that can improve the detection of t0 in case of problems.
        In case of wrong detection of t0 (some kind of displacement) change the window_size. If the number of detection transients is higher (lower) than it actually is (it can be in case of the high noise),
        decrease (or increase) the prominence. There is also possibility to provide stimuli times (t_stim parameter) instead of automatic detection.

        :param time: array of the values of the time during contraction
        :type time: array_like
        :param Sig: array of the values of the transients
        :type Sig: array_like
        :param kernel: kernel used for analyze, can be chosen between RBF and Gibbs, defaults to "Gibbs"
        :type kernel: str, optional
        :param window_size: size of box filter window, can be modified in case of the bad detection of transients start, defaults to 20
        :type window_size: int, optional
        :param prominence: measures how much a peak stands out from the surrounding baseline, can be used when no or extra transients are detected, defaults to 1
        :type prominence: float, optional
        :param t_stim: stimulation times, defaults to None
        :type t_stim: array_like, optional
        :param quantile1: the first quantile for parameters of transients, will be used for the detection of rise,decay times and durations as the ones between quantile1 and 1 - quantile1 percents of the corresponding transient phase,  defaults to 0.1
        :type quantile1: float
        :param quantile2: the second quantile for parameters of transients, will be used for the detection of rise,decay times and durations as the ones between quantile1 and 1 - quantile1 percents of the corresponding transient phase,  defaults to 0.2
        :type quantile2: float
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
        self._n_baseline = 100
        self.frac_spline = 0.04
        self._kernel = None
        if kernel == "Gibbs":
            self._kernel = LinearGibbs()
        else:
            self._kernel = gpflow.kernels.RBF()
        self.quantile1 = quantile1
        self.quantile2 = quantile2
        self._SetData(time, Sig, t_stim)
        self._n_samples_offset = 8

    def _SetData(self, time, Sig, t_stim):
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
        :param output: defines whether it is used for the output of the data or for approximation, defaults to True
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
            win = windows.boxcar(self._window_size)
            filtered = np.convolve(self.Sig, win, mode='same') / np.sum(win)
            res = filtered - self.Sig
            self.borders, _ = find_peaks(res, prominence=self._prominence * np.max(res), height=0)
            self.borders = self.borders.astype("int64")
            self.t0s_est = self.borders * self.dt + self.Time[0]
        else:
            self.t0s_est = deepcopy(t_stim)
            self.borders = self.t0s_est / self.dt
            self.borders = self.borders.astype('int64')
        self.t0s = [-1] * len(self.borders)
        self.baselines = [-1] * len(self.borders)
        self.transients = [-1] * len(self.borders)
        self.parameters = [-1] * len(self.borders)
        self.borders = np.append(self.borders, (self.Time[-1] - self.Time[0]) / self.dt).astype("uint64")
        n = len(self.borders) - 1
        for i in range(n):
            if i == 0:
                dn = int((self.t0s_est[0] - self.Time[0]) / self.dt)
                n0 = max(dn - self._n_baseline, 0)
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
        res = spline(t0)[0] / mean - 1
        return res * res

    def _Gradient(self, x_start, cs, baseline):
        """Gradient descent

        :param x_start: starting point
        :type t0: float
        :param cs: GP spline
        :type spline: UnivariateSpline
        :param baseline: baseline value
        :type mean: float
        :return: time, at which GPs intersect
        :rtype: float

        """
        dx = 1e-5
        t = x_start - 0.5 * self.dt
        tprev = t
        f = self._FindT0(t, cs, baseline)
        dydx = 0
        fprev = f
        alpha = 0.002
        eps = 1e-5
        n_iter = 0
        while f > 0:
            fl = self._FindT0(t - dx, cs, baseline)
            fr = self._FindT0(t + dx, cs, baseline)
            dydx = (fr - fl) / (2 * dx)
            t = tprev - alpha * dydx
            f = self._FindT0(t, cs, baseline)
            if np.abs(f - fprev) < eps:
                n_iter += 1
                if n_iter > 10:
                    break
            else:
                n_iter = 0
            tprev = t
            fprev = f
        return t

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
        # Calcium signal
        baseline = self.baselines[idx]
        Yfit = sig[::]
        Yfit[0] = baseline
        Xfit = t
        # remove the points after peak
        Yfit2 = Yfit[::]
        Yfit = Yfit2[(Xfit > self.dt * self._n_samples_offset) | (
                    Yfit2 >= baseline)]  # too low values (which appear at high noise) at the start can break the GP approximation
        Xfit = Xfit[(Xfit > self.dt * self._n_samples_offset) | (Yfit2 >= baseline)]
        gpr = gpflow.models.GPR((Xfit.reshape(-1, 1), Yfit.reshape(-1, 1)), kernel=self._kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(gpr.training_loss, variables=gpr.trainable_variables)
        t_start = -2 * self.dt
        xp = np.arange(t_start, Xfit[-1], self.frac_spline * self.dt)
        # find t0
        x_findt0 = np.linspace(t_start, 2 * self.dt, 30)
        mean, _ = gpr.predict_y(x_findt0.reshape(-1, 1))
        t0_values = mean
        cs_t0 = CubicSpline(x_findt0, t0_values)
        t0_sig = self._Gradient(x_findt0[-1], cs_t0, baseline)
        self.t0s[idx] = t0_sig + ti
        self.parameters[idx].append(self.t0s[idx])
        if idx == 0:
            self.parameters[idx].append(np.nan)
        else:
            self.parameters[idx].append(self.t0s[idx] - self.t0s[idx - 1])
        self.parameters[idx].append(baseline)
        y = np.zeros_like(xp)
        mean, _ = gpr.predict_y(xp.reshape(-1, 1))
        y = mean[:, 0]
        cs = CubicSpline(xp, y)
        self.transients[idx] = cs
        Peak_sig = minimize(self._MinFunc, x0=[t0_sig + 1.55 * self._n_samples_offset * self.dt],
                            args=(self.transients[idx], True))
        Amp_sig = -Peak_sig.fun
        TTP_sig = Peak_sig.x[0]
        base_sig = baseline
        self.parameters[idx].append(Amp_sig)
        self.parameters[idx].append(TTP_sig - t0_sig)
        # get rise time, FDHM, decay time
        try:
            q1_rise = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                             args=(self.quantile1, -Peak_sig.fun, base_sig, self.transients[idx]))
        except:
            q1_rise = None
        try:
            q2_rise = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                             args=(self.quantile2, -Peak_sig.fun, base_sig, self.transients[idx]))
        except:
            q2_rise = None
        try:
            half_rise = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                               args=(0.5, -Peak_sig.fun, base_sig, self.transients[idx]))
        except:
            half_rise = None
        try:
            q1m_rise = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                              args=(1 - self.quantile1, -Peak_sig.fun, base_sig, self.transients[idx]))
        except:
            q1m_rise = None
        try:
            q2m_rise = bisect(self._FindTimeFraction, t0_sig, TTP_sig,
                              args=(1 - self.quantile2, -Peak_sig.fun, base_sig, self.transients[idx]))
        except:
            q2m_rise = None
        q1_decay = bisect(self._FindTimeFraction, TTP_sig, t[-1],
                          args=(self.quantile1, -Peak_sig.fun, base_sig, self.transients[idx]))
        q2_decay = bisect(self._FindTimeFraction, TTP_sig, t[-1],
                          args=(self.quantile2, -Peak_sig.fun, base_sig, self.transients[idx]))
        half_decay = bisect(self._FindTimeFraction, TTP_sig, t[-1],
                            args=(0.5, -Peak_sig.fun, base_sig, self.transients[idx]))
        q1m_decay = bisect(self._FindTimeFraction, TTP_sig, t[-1],
                           args=(1 - self.quantile1, -Peak_sig.fun, base_sig, self.transients[idx]))
        q2m_decay = bisect(self._FindTimeFraction, TTP_sig, t[-1],
                           args=(1 - self.quantile2, -Peak_sig.fun, base_sig, self.transients[idx]))
        if q1m_rise is not None and q1_rise is not None:
            self.parameters[idx].append(q1m_rise - q1_rise)  # rise times
        else:
            self.parameters[idx].append(np.nan)
        if q1m_rise is not None and q1_rise is not None:
            self.parameters[idx].append(q2m_rise - q2_rise)
        else:
            self.parameters[idx].append(np.nan)
        self.parameters[idx].append(q1_decay - t0_sig)  # durations
        self.parameters[idx].append(q2_decay - t0_sig)
        if half_rise is not None:
            self.parameters[idx].append(half_decay - half_rise)
        else:
            self.parameters[idx].append(np.nan)
        self.parameters[idx].append(q2m_decay - t0_sig)
        self.parameters[idx].append(q1m_decay - t0_sig)
        self.parameters[idx].append(q1_decay - q1m_decay)  # decay times
        self.parameters[idx].append(q2_decay - q2m_decay)
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
        sig_model_out = sig_spline(t_exp_out[t_exp_out > t_model_out[0]])
        columns = [x_label, y_label, f"Approximated {y_label}"]
        output = np.zeros((t_exp_out.shape[0], 3))
        output.fill(np.nan)
        output[:, 0] = t_exp_out
        output[:, 1] = sig_exp_out
        output[t_exp_out > t_model_out[0], 2] = sig_model_out
        df_out = pd.DataFrame(output, columns=columns[:3])
        return df_out

    def GetParametersTable(self, x_label="x", y_label="y"):
        """GetParametersTable allows to get the table with parameters such as:

     * Amplitude;
     * Baseline;
     * Time-to-peak (*TTP*);
     * Rise time at quantiles (*x-(1-x)%*) defined by user (20-80%, 10-90%, etc.);
     * Transients durations at quantiles (*x%*) defined by user (10%, 20%, etc.) and at 50% (Full Duration at Half Maximum (FDHM));
     * Decay time at quantiles (*(1-x)-x%*) defined by user (80-20%, 90-10%, etc.);

       Table of parameters contains values of *t0* - the start time of the transient or of the stimulus, *delta t0* - the difference in t0 between two
       neighboring transients.

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
        columns = [f"t0{x_out}", f"delta t0{x_out}", f"Baseline{y_out}", f"Amplitude{y_out}", f"TTP{x_out}",
                   f"t_{q1}%-{100 - q1}%{x_out}", f"t_{q2}%-{100 - q2}%{x_out}", f"d_{q1}%{x_out}",
                   f"d_{q2}%{x_out}", f"d_50%{x_out}", f"d_{100 - q2}%{x_out}",
                   f"d_{100 - q1}%{x_out}", f"t_{100-q1}%-{q1}%{x_out}", f"t_{100-q2}%-{q2}%{x_out}"]
        pars = np.array(self.parameters)
        pars[1:, 1] = pars[1:, 0] - pars[:-1, 0]
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