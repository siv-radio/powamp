# -----------------------------------------------------------------------------
#
# This file is part of the PowAmp package.
# Copyright (C) 2022-2024 Igor Sivchek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------

"""
The base of a power amplifier model with one active device.
"""

from collections import namedtuple

import numpy as np

from . import hbmath as hbm
from .actdev import ADManager
from .loadnet import LoadManager
from .simdata import SimDBManager


class OneAD:
    """
    The base class for all models with one active device (linear or nonlinear).

    The mathematical model that is used here uses the harmonic balance method
    for calculation of voltages and currents.
    The program is based on the PhD dissertation written and secured by Igor
    Sivchek in 2017.
    About the harmonic balance, see ["Nonlinear Microwave and RF Circuits",
    Stephen A. Maas, 2nd ed., 2003].

    The general representation of an equivalent circuit of a power amplifier
    with one active device.

     Independent |          Passive          | Active
     voltage     |          linear           | device
     source      |          subcircuit       |

     (2)  i_2       -----------  i_1 / i_lin    i_ad (1)
      ----->--------|         |------<----------->-----
      |             |         |                       |
      |             |         |                       |
      |             |         |                       |
      v i_pwr       |         |                       |
      |             |         |                       |
    v_pwr           |   PLC   |                       ad
      |             |         |                       |
      |             |         |                       |
      |             |         |                       |
      |             |         |                       |
      |             |         |                       |
      --------------|         |------------------------
    (gnd)           -----------

    Representation with an equvalent current source "i_es" instead of
    the voltage source "v_pwr".

           -----------       i_1 / i_lin  i_ad  (1)
      -----|         |----o------<--------->------
      |    |         |    |                      |
      |    |         |    |                      |
      |    |         |    |                      |
      |    |         |    |                      |
      |    |         |   i_es                    |
      |    |   PLC   |    |                      ad
      |    |         |    v                      |
      |    |         |    |                      |
      |    |         |    |                      |
      |    |         |    |                      |
      |    |         |    |                      |
      -----|         |----o-----------------------
    (gnd)  -----------

    Some abbreviations:
    AD / ad - active device.
    PLC - two-port passive linear subcircuit.
    v_pwr - power supply voltage.
    i_pwr - power supply current.
    i_lin - linear subcircuit current.
    i_ad - active device's direct current.

    The AD has only quasistatic losses (i. e. in conductive state).
    No sources of dynamic losses (i. e. at switching moments) has been
    implemented.
    Note: "bidirect:switch" type of the AD is linear, while "forward:switch"
    and "freewheel:switch" are not. Hence "bidirect:switch" type does not
    require an iterative algorithm to find the required AD voltage that brings
    convergence.

    Fields (attributes) that must be in the heirs of this class:
    _name : str
        'name' parameter. The name is common for all objects of a PA model
        class. It is set externally without a setter.
    """

    def __init__(self):
        """
        The constructor of an "OneAD" object.
        It does not receive any arguments. Any parameters have to be set via
        related setters or configurators.
        """
        self.__t_wrk = 2*np.pi  # Time period of a working / tuning frequency.
        self.__ad = ADManager()  # Active device.
        self.__load = LoadManager()  # Load network.
        self.__dft = hbm.DFT()  # Discrete Fourier transform.
        self.__hb_opts = {'maxiter': 64, 'reltol': 1e-9}  # Options of a harmonic balance simulation.
        self.__hb_cnt_mgr = hbm.HBCntMgr()  # Harmonic balance counter manager.

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def get_name(self):
        """
        Get the name a power amplifier (PA).

        Returns
        -------
        name : str
            The name of a PA.
        """
        return self._name

    def set_h_len(self, val):
        """
        Set a number of harmonics "h_len".
        h_len >= 2.

        Parameters
        ----------
        val : int
            A number of harmonics to calculate.

        Returns
        -------
        "self" reference to the caller object.

        Notes
        -----
        This parameter sets the precision of all further calculations.
        A higher "h_len" value gives higher precision, but it requires more
        computational resources.
        In practice, h_len = 50 is usually enough. To get more accurate
        results, use h_len = 100 or even 200, but beware, in some cases it
        can take too much time to produce computations.
        """
        # Set a number of harmonics in the "DFT" object.
        self.__dft.set_dft_length(val)
        return self

    def get_h_len(self):
        """
        Get a number of harmonics "h_len".

        Returns
        -------
        h_len : int
            A number of harmonics.
        """
        return self.__dft.get_dft_length()

    def get_s_len(self):
        """
        Get a number of time samples "s_len" in a time period "t_wrk".
        s_len = 2*h_len - 1, s_len >= 3, where h_len - a number of harmonics.

        Returns
        -------
        s_len : int
            A number of time samples.
        """
        return hbm.h2s(self.get_h_len())

    def set_f_wrk(self, val):
        """
        Set a working frequency.

        Parameters
        ----------
        val : float
            A working frequency.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__t_wrk = 1.0/float(val)
        return self

    def get_f_wrk(self):
        """
        Get a working frequency.

        Returns
        -------
        result : float
            A working frequency.
        """
        return 1.0/self.__t_wrk

    def get_w_wrk(self):
        """
        Get an angular frequency.
        w = 2*pi*f, where f - working frequency.

        Returns
        -------
        result : float
            The value of a working angular frequency.
        """
        return hbm.f2w(1.0/self.__t_wrk)

    def set_t_wrk(self, val):
        """
        Set the time period of a working frequency.

        Parameters
        ----------
        val : float
            A time period.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__t_wrk = float(val)
        return self

    def get_t_wrk(self):
        """
        Get the time period of a working frequency.

        Returns
        -------
        result : float
            A time period.
        """
        return self.__t_wrk

    def get_sample_duration(self):
        """
        It returns a time sample duration value "dt".
        dt = t_wrk/s_len, where
        t_wrk - the time period of a working frequency,
        s_len - a number of time samples in a time period.

        Returns
        -------
        dt : float
            A time sample duration.
        """
        return hbm.sample_duration(s_len=self.get_s_len(), t_wrk=self.__t_wrk)

    def set_hb_maxiter(self, val):
        """
        Set the maximum number of iterations of the harmonic balance (HB)
        algorithm.

        Parameters
        ----------
        val : int
            The maximum number of HB iterations.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__hb_opts['maxiter'] = int(val)
        return self

    def get_hb_maxiter(self):
        """
        Get the maximum number of iterations of the harmonic balance (HB)
        algorithm.

        Returns
        -------
        maxiter : int
            The maximum number of HB iterations.
        """
        return self.__hb_opts['maxiter']

    def set_hb_reltol(self, val):
        """
        Set a relative tolerance of the maximum error value among the elements
        of the current-error vector "cev".

        Parameters
        ----------
        val : float
            A relative tolerance value.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__hb_opts['reltol'] = float(val)
        return self

    def get_hb_reltol(self):
        """
        Get a relative tolerance of the maximum error value among the elements
        of the current-error vector "cev".

        Returns
        -------
        reltol : float
            A relative tolerance value.
        """
        return self.__hb_opts['reltol']

    HBOptions = namedtuple(
        'HBOptions', 'h_len s_len f_wrk t_wrk dt maxiter reltol')
    # Note: the working frequency also relates to a power amplifier model,
    # since it affects on the active device. However, it is placed here for
    # convenience.

    def set_hb_options(self, *,
                       h_len=None,
                       t_wrk=None, f_wrk=None,
                       maxiter=None, reltol=None):
        """
        Set options of a harmonic balance simulation.

        Parameters
        ----------
        h_len : int, optional
            A number of harmonics.
        t_wrk : float, optional
            The time period of a working frequency.
        f_wrk : float, optional
            A working frequency.
        maxiter : int, optional
            The maximum number of iterations in a loop of searching a
            solution.
        reltol : float, optional
            Relative tolerance of the maximum error value among the elements
            of the current-error vector "cev".

        Notes
        -----
        If a model spent more energy than received (perpetuum mobile), or you
        see some other suspicious artefacts, you can try to increase the
        "reltol" option.
        """
        if (t_wrk is not None) and (f_wrk is not None):
            raise RuntimeError("Ambiguous period / frequency definition.")
        if h_len is not None:           self.set_h_len(h_len)
        if t_wrk is not None:           self.set_t_wrk(t_wrk)
        if f_wrk is not None:           self.set_f_wrk(f_wrk)
        if maxiter is not None:         self.set_hb_maxiter(maxiter)
        if reltol is not None:          self.set_hb_reltol(reltol)
        return self

    def get_hb_options(self):
        """
        Get options of a harmonic balance (HB) simulation.

        Parameters
        ----------
        hb_opts : HBOptions
            A "namedtuple" that contains options of an HB simulation. See
            related separate getters and setters to get information about the
            fields.
        """
        return self.HBOptions(
            h_len=self.get_h_len(), s_len=self.get_s_len(),
            f_wrk=self.get_f_wrk(),
            t_wrk=self.get_t_wrk(), dt=self.get_sample_duration(),
            maxiter=self.get_hb_maxiter(), reltol=self.get_hb_reltol())

    # Note: this name started with "config" to highlight that it is not a
    # regualar setter with a strict set of input parameters.
    def config_ad(self, **kwargs):
        """
        Configure the active device (AD).

        Parameters
        ----------
        kwargs : dict
            A dictionary of AD parameters. For example:
            name : str, optional
                A string that contains the name of an active device.
                For example, "bidirect:switch", "forward:switch", or
                "freewheel:switch".
            r_ad : float, optional
                A number that is greater than zero.
                Forward AD resistance in conductive state. In case of
                "freewheel:switch" AD type it is forward conduction resistance
                in conductive state.
            r_fwd : float, optional
                A number that is greater than zero.
                Backward conduction (i. e. freewheeling / anti-parallel diode
                conduction) of "freewheel:switch" AD type only.

        Returns
        -------
        "self" reference to the caller object.

        Notes
        -----
        Examples of sets of parameters:
        {name='bidirect:switch', r_ad=0.1}
        {name='forward:switch', r_ad=0.1}
        {name='freewheel:switch', r_ad=0.1, r_fwd=0.1}
        {r_ad=0.1}  # Can be used if the name has already been set.

        Setting an AD name will reset the existing AD parameters.
        """
        self.__ad.config_ad(**kwargs)
        return self

    def get_ad_config(self):
        """
        Get the active device (AD) configuration.

        Returns
        -------
        result : Params
            A "namedtuple" that contains a set of AD parameters.
        """
        return self.__ad.get_ad_config()

    def get_ad_name(self):
        """
        It returns the active device (AD) name (i. e. its type in
        non-programming meaning).

        Returns
        -------
        name : str
            A string that contains the AD name.
        """
        return self.__ad.get_name()

    def is_ad_linear(self):
        """
        Check if an active device (AD) model is linear. If it is, then its
        parameters do not depend on any electrical characteristics (voltages
        and currents) in a circuit.

        Returns
        -------
        result : bool
            "True" if an AD model is linear, "False" otherwise.
        """
        return self.__ad.is_linear()

    def has_ad_linear_repr(self):
        """
        Check if an active device (AD) model has a linear representation.
        If the linear representation exists, it is imprecise in general case,
        but can be applicable in some operational area of electrical
        characteristics. For example, 'bidirect:switch' is a valid linear
        representation of 'forward:switch' and 'freewheel:switch' if
        (v_ad_t > 0).

        Returns
        -------
        result : bool
            "True" if an AD model has a linear representation,
            "False" otherwise.
        """
        return self.__ad.has_linear_repr()

    def config_load(self, **kwargs):
        """
        Configure the load network.

        Parameters
        ----------
        kwargs : dict
            A dictionary of load network parameters. For example,
            name : str, optional
                A string that contains the name of a required load network.
                For example, 'zbar'.
            f_cent : float, optional
                A central (working) frequency.
            z_cent : complex, optional
                An input impedance value of the load network at a central
                frequency.

        Returns
        -------
        "self" reference to the caller object.

        Notes
        -----
        Examples of sets of parameters:
        {name='zbar', f_cent=460e3, z_cent=(1.1 + 1j*2.2)}
        {'name': 'parcir:le', 'f_tun': 1.4e6, 'z_in_req': 10.0, 'q_eqv': 2.0,
         'g_cp': 1e-6, 'r_lp': 1e-6}
        {'name': 'pinet:le', 'f_tun': 3.1e6, 'z_in_req': 8.0, 'midcoef': 0.75,
         'g_ci': 1e-6, 'g_co': 1e-6, 'r_lm': 1e-6, 'r_out': 4.0}
        {name='custom', obj=loadnet}
        {z_cent=(3.3 - 1j*4.4)}  # Can be used if the name has already been
                                 # set.

        Setting a load network name will reset the existing load parameters.
        """
        # Pass parameters into a "LoadManager".
        self.__load.config_load(**kwargs)
        # Return the reference to the object for method chaining.
        return self

    def get_load_config(self):
        """
        Get the load network configuration.

        Returns
        -------
        result : Params
            A "namedtuple" that contains a set of load network parameters.
        """
        return self.__load.get_load_config()

    def get_load_name(self):
        """
        Get the name of a load network.

        Returns
        -------
        name : str
            The name of a load network.
        """
        return self.__load.get_name()

    LoadCheck = namedtuple('LoadCheck', 'success status message')

    def check_load(self):
        """
        Check that if a load network is tunable, it has been tuned properly.

        Returns
        -------
        result : LoadCheck
            A "namedtuple" that contains following fields:
            success : bool
                If "True", the load network is:
                1) non-tunable or
                2) tunable and has been successfuly tuned.
                If "False", the load network is tunable, but a tuning procedure
                has failed.
            status : int
                An error code.
            message : str
                A message that describes the result of a check.

        Notes
        -----
        It can be useful to call this method before getting a "z_out" value to
        guarantee that the value is correct.
        """
        if self.__load.is_tunable():
            tunres = self.get_load_config().tunres
            return self.LoadCheck(
                success=tunres.success,
                status=tunres.status, message=tunres.message)
        return self.LoadCheck(
            success=True,
            status=0, message="The load network has a non-tunable type.")

    def get_z_out(self, f=None):
        """
        Get an input impedance value of a load network.

        Parameters
        ----------
        f : float, optional
            A frequency value. If it is "None", then it uses an internal
            working frequency (i. e. the 1st harmonic frequency).

        Returns
        -------
        z_out : complex
            An input impedance value of a load network at a certain frequency.

        Notes
        -----
        It does not check if a load network has been successfully tuned in the
        case of a tunable load. So that, it can return inapplicable data. It is
        recommended to check if a tuning was successful before calling this
        method.
        """
        if f is None:   return self.__load.get_z_in(w=self.get_w_wrk())
        else:           return self.__load.get_z_in(w=hbm.f2w(f))

    # Protected methods.

    def _f2t_array(self, fa):
        """
        Convert a given "numpy.array" of harmonics "fa" into an array of time
        samples.
        ta = idftm*fa, where
        idftm - an inverse discrete Fourier transform matrix.

        Parameters
        ----------
        fa : ndarray
            An array of harmonics. Its size is "s_len", where
            s_len - a number of time samples.
            Each complex number is represented by a couple of real numbers.
            The 0th harmonic is a real number and has no imaginary part.

        Returns
        -------
        ta : ndarray
            An array of time samples. Its size is "s_len", where
            s_len - a number of time samples.
        """
        return self.__dft.f2t(fa)

    def _t2f_array(self, ta):
        """
        Convert a given "numpy.array" of time samples "ta" into an array of
        harmonics.
        fa = dftm*ta, where dftm - a discrete Fourier transform matrix.

        Parameters
        ----------
        ta : ndarray
            An array of time samples. Its size is "s_len", where
            s_len - a number of time samples.

        Returns
        -------
        fa : ndarray
            An array of harmonics. Its size is "s_len", where
            s_len - a number of time samples.
            Each complex number is represented by a couple of real numbers.
            The 0th harmonic is a real number and has no imaginary part.
        """
        return self.__dft.t2f(ta)

    def _select_harmonic(self, *, ta, h):
        """
        Select a required h-th harmonic from a time domain array "ta".

        Parameters
        ----------
        ta : ndarray
            An array in time domain. Its size is "s_len", where
            s_len - a number of time samples.
        h : int
            A required harmonic number.

        Returns
        -------
        harm : complex
            A required harmonic value.
        """
        if h == 0:
            return np.complex(self.__dft.select_harmonic(ta=ta, h=0), 0)
        return np.complex(*self.__dft.select_harmonic(ta=ta, h=h))

    def _make_admittance_matrix(self, adm_func):
        """
        Get the admittance matrix for a given admittance function "adm_func".

        Parameters
        ----------
        adm_func : callable f(w)
            An admittance function of an angular frequency "w" that produces
            complex numbers.

        Returns
        -------
        y_f : ndarray
            An admittance matrix in frequency domain. Its size is
            (s_len, s_len), where s_len - a number of time samples.
            Each complex number is represented by a couple of real numbers.
            The 0th harmonic is a real number and has no imaginary part.
        """
        w_wrk = self.get_w_wrk()  # Working angular frequency.
        return hbm.make_admittance_matrix(
            adm_func=adm_func, h_len=self.get_h_len(), w_wrk=w_wrk)

    def _get_curr_in_fd(self, adm_func, v_f):
        """
        Get a current vector as the product of an admittance matrix and
        a voltage vector in frequency domain.
        i = y*v, where
        y - an admittance matrix,
        v - a voltage vector.
        The admittance matrix is defined implicitly by an admittance function
        "adm_func".

        Parameters
        ----------
        adm_func : callable f(w)
            An admittance function of an angular frequency "w" that produces
            complex numbers.
        v_f : ndarray
            A voltage vector of size "s_len" in frequency domain, where
            s_len = (2*h_len - 1) - a number of time samples,
            h_len - a number of harmonics including 0th.

        Returns
        -------
        i_f : ndarray
            A current vector of size "s_len" in frequency domain.

        Notes
        -----
        This function is used to avoid the creation of the admittance matrix
        which mostly consists of zeros.
        """
        w = self.get_w_wrk()  # Working angular frequency.
        i_f = np.zeros((self.get_s_len(),))
        i_f[0] = adm_func(0.0).real*v_f[0]  # The 0th harmonic.
        for h in range(1, self.get_h_len()):
            y = adm_func(h*w)  # Admittance.
            hr = 2*h - 1  # Index of a real part.
            hi = 2*h  # Index of an imaginary part.
            i_f[hr] = y.real*v_f[hr] - y.imag*v_f[hi]  # Real part.
            i_f[hi] = y.real*v_f[hi] + y.imag*v_f[hr]  # Imaginary part.
        return i_f

    def _get_volt_in_fd(self, adm_func, i_f):
        """
        Get a voltage vector as the result of left division of an admittance
        matrix and a current vector in frequency domain.
        v = y\i = (y^-1)*i = z*i, z = y^-1, where
        i - a current vector,
        y - an admittance matrix,
        z - an impedance matrix.
        The admittance matrix is defined implicitly by an admittance function
        "adm_func".

        Parameters
        ----------
        adm_func : callable f(w)
            An admittance function of an angular frequency "w" that produces
            complex numbers.
        i_f - current vector of size "s_len" in frequency domain, where
            s_len = (2*h_len - 1) - a number of time samples,
            h_len - a number of harmonics including 0th.

        Returns
        -------
        v_f : ndarray
            A voltage vector of size "s_len" in frequency domain.

        Notes
        -----
        This function is used to avoid the creation of the admittance matrix
        which mostly consists of zeros and also the left division of the
        admittance matrix and the "i_f" vector.
        """
        w = self.get_w_wrk()  # Working angular frequency.
        v_f = np.zeros((self.get_s_len(),))
        v_f[0] = i_f[0]/adm_func(0.0).real  # The 0th harmonic.
        for h in range(1, self.get_h_len()):
            z = 1.0/adm_func(h*w)  # Impedance.
            hr = 2*h - 1  # Index of a real part.
            hi = 2*h  # Index of an imaginary part.
            v_f[hr] = z.real*i_f[hr] - z.imag*i_f[hi]  # Real part.
            v_f[hi] = z.real*i_f[hi] + z.imag*i_f[hr]  # Imaginary part.
        return v_f

    def _get_dftm(self):
        """
        It returns a referense to a discrete Fourier transform matrix (DFTM).

        Returns
        -------
        dftm : ndarray
            A DFTM of size (s_len, s_len), where
            s_len - a number of time samples.
        """
        return self.__dft.forward()

    def _get_idftm(self):
        """
        It returns a referense to an inverse discrete Fourier transform matrix
        (IDFTM).

        Returns
        -------
        idftm : ndarray
            An IDFTM of size (s_len, s_len), where
            s_len - a number of time samples.
        """
        return self.__dft.inverse()

    def _get_y_out(self, w):
        """
        Get an output admittance value (i. e. input admittance of a load
        network) at a given angular frequency "w".

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y_out : complex
            An output admittance value.
        """
        return self.__load.get_y_in(w=w)

    def _get_z_out(self, w):
        """
        Get an output impedance value (i. e. input impedance of a load
        network) at a given angular frequency "w".

        Parameters
        ----------
        w : float
            An angular frequency value.

        Returns
        -------
        z_out : complex
            An output impedance value.

        Notes
        -----
        Do not mix-up with "get_z_out(self, f)".
        """
        return self.__load.get_z_in(w=w)

    def _has_hb_counter(self):
        """
        Check if at least one harmonic balance (HB) counter exists.

        Returns
        -------
        result : bool
            A flag that is "True" if an HB counter exists, "False" otherwise.
        """
        return self.__hb_cnt_mgr.has_counter()

    def _make_hb_counter(self):
        """
        Create a harmonic balance (HB) counter.

        Returns
        -------
        hb_cnt : HBCounter
            An HB counter object.
        """
        return self.__hb_cnt_mgr.make_counter()

    def _incr_ss_evals(self, ss_evals=1):
        """
        Increase the total number of steady-state (SS) response evaluations by
        a given value.
        It affects on the data that will be provided by all existing counters.

        Paramters
        ---------
        ss_evals : int
            A number of SS evaluations to add.
        """
        self.__hb_cnt_mgr.incr_ss_evals(ss_evals=ss_evals)

    def _incr_hb_iters(self, hb_iters=1):
        """
        Increase the total number of harmonic balance (HB) iterations by a
        given value.
        It affects on the data that will be provided by all existing counters.

        Paramters
        ---------
        hb_iters : int
            A number of HB iterations to add.
        """
        self.__hb_cnt_mgr.incr_hb_iters(hb_iters=hb_iters)

    def _get_vi_ad(self, *, y_11_f, i_es_f, try_linear=False, v_init=None):
        """
        Get the voltage and current time samples of the nonlinear active
        device (AD).
        It uses Newton's algorithm to solve a system of nonlinear equations.
        The function is used for characteristics calculation of a power
        amplifier (PA) model.
        If an AD is linear, then it provides a solution within 1.5 iterations.

        Parameters
        ----------
        y_11_f : ndarray
            An admittance matrix "y_11_f" of a two-port network in frequency
            domain. Its size is (s_len, s_len), where s_len - a number of time
            samples.
        i_es_f : ndarray
            A harmonics vector of the respective equivalent current source.
            Its size is "s_len".
        try_linear : bool, optional
            Try to use a linear counterpart of an AD. If an AD is nonlinear and
            a linear representation exists, it will calculate the result
            faster. If there is only a nonlinear representation of an AD, it
            will use it anyway.
            The option is useful in a PA tuning process when (v_ad_t >= 0) is
            expected in the final result. It does not matter that (v_ad_t < 0)
            at some intermediate steps.
        v_init : ndarray, optional
            An array of initial "v_ad_t" time samples.

        Returns
        -------
        v_ad_t : ndarray
            A vector of AD voltage time samples. Its size is "s_len".
        i_ad_t : ndarray
            A vector of AD current time samples. Its size is "s_len".

        Notes
        -----
        This is the most important part of each model that is based on this
        class. If you want to deeply understand this library, then get the
        comprehension of this method. The Maas' book (see below) is highly
        recommended.

        The model is simplified and does not provide the ability to take into
        account a nonlinear capacitance.

        If an AD is linear, it does not have any dependencies on voltages and
        currents in a circuit. It can have only time dependent variables.
        For example, a diode is a passive nonlinear device because its current
        depends on voltage.

        About a direct solution in the case of a linear AD.
        If an AD is linear, it is possible to find a solution (voltage on the
        AD) directly without using the Newton's algorithm.
        Although this method always runs the Newton's algorithm, it is handy to
        understand when it can give the fastest solution. It provides the
        ability to significantly speed-up a tuning process of a PA model.

        v_ad_t = -idftm*(y_11_f + dftm*g_ad_t*idftm)^-1 *
                 (i_es_f + dftm*(i_on_t - g_ad_t*v_on_t)),
        i_es_f = y_12_f*v_pwr_f, where
        (i)dftm - (inverse) discrete Fourier transform matrix of size
            (s_len, s_len);
        y_11_f, y_12_f - matrices of sizes (s_len, s_len) of Y-parameters of
            a two-port passive linear network in frequency domain;
        g_ad_t - a vector of size "s_len" of AD conductance time samples;
        i_on_t - a vector of size "s_len" of AD independent current time
            samples related to the AD in conductive ("on") state (i. e. it
            describes an independent AD current source);
        v_on_t - a vector of size "s_len" of AD independent voltage time
            samples related to the AD in conductive ("on") state (i. e. it
            describes an independent AD voltage source);
        v_pwr_f - a vector of size "s_len" of power supply voltage harmonics;
        i_es_f - a current harmonics vector of size "s_len";
        s_len - a number of time samples.

        References
        ----------
        1. "Nonlinear Microwave and RF Circuits", Stephen A. Maas, 2nd ed.,
           2003.
        2. "Povyshenie KPD i vykhodnoy moshchnosti okonechnykh kaskadov
           svyaznykh radioperedayushchikh ustroystv diapazonov ONCh - NCh na
           generatornykh lampakh" / "Increasing the efficiency and output power
           of the output stages of communication transmitters in VLF - LF
           ranges with power tubes", I. V. Sivchek, PhD dissertation, SPbPU,
           2017. In Russian.
        3. "Mathematical Model of Class E Amplifier Based on the Harmonic
           Balance Method", I. V. Sivchek, 2016. In Russian.
           https://ieeexplore.ieee.org/document/7878867
        """
        if try_linear:
            # If an AD has a linear representation, try to use it. Otherwise it
            # call usual nonlinear representation.
            get_ij = self.__ad.try_get_linear_ij
        else:
            # Use a common nonlinear AD represenatation. Of course if an AD is
            # linear, it will be linear.
            get_ij = self.__ad.get_ij

        # The maximum number of HB iterations.
        maxiter = self.get_hb_maxiter()

        # "abstol" is the maximum error (epsilon) of an element in a
        # current-error vector "cev".
        # Use the biggest magnitude from the 0th (DC) and 1st harmonics.
        reltol = self.get_hb_reltol()
        abstol = reltol*max(abs(i_es_f[0]), abs(i_es_f[1] + 1j*i_es_f[2]))

        # Create a discrete Fourier transform matrix (DFTM) and
        # an inverse discrete Fourier transform matrix (IDFTM).
        dftm = self._get_dftm()  # Get a reference to the DFTM.
        idftm = self._get_idftm()  # Get a reference to the IDFTM.

        # Error function to minimize.
        # cev = y_12_f*v_20_f + y_11_f*v_10_f + i_ad_f.
        # i_ad_f - current through the AD in frequency domain.
        # i_es_f = y_12_f*v_20_f.
        # i_es_f - the imaginary current source between nodes 1 and 0.
        # Its current directs from node 1 to 0.
        # Only its 0th element is not zero.

        s_len = self.get_s_len()  # Number of using samples in time domain.
        i_ad_t = np.empty((s_len,))  # The initial condition for finding "i_ad_t".
        if v_init is None:
            v_ad_t = np.zeros((s_len,))  # The initial condition for finding "v_ad_t".
        else:
            v_ad_t = v_init  # Passing by reference.
        v_ad_f = self._t2f_array(v_ad_t)
        # print("OneAD._get_vi_ad. Initial v_ad_f=", v_ad_f)  # Test.

        # The Newton's method of solving a nonlinear equation.
        # After finding "cev" value on a current iteration, it checks if
        # (max(abs(cev)) < abstol).
        # If it is "True", then a soultion has been found.
        # The maximum number of iterations is limited and equals "maxiter".
        for iteration in range(maxiter):
            # Warning: it will work only if the AD has a proper "get_ij"
            # method.
            i_ad_t, j_ad_t = get_ij(v_ad_t=v_ad_t)
            i_ad_f = self._t2f_array(i_ad_t)
            # See ["Nonlinear Microwave and RF Circuits", Stephen A. Maas,
            # 2nd ed., 2003, p. 132, eq. 3.25].
            cev = i_es_f + np.dot(y_11_f, v_ad_f) + i_ad_f
            # print("OneAD._get_vi_ad. cev=", cev)  # Test.

            # The cycle interuption if a target tolerance has been achieved.
            cev_max = np.max(np.abs(cev))
            if cev_max < abstol:
                break

            # Jacobian.
            # j = y_11_f + j_ad_f.
            # A Jacobian has a simple form as below only when "g_ad_t" has no
            # smooth dependency on "v_ad_t". Like in switch mode.
            j_ad_f = y_11_f + np.dot(dftm, np.dot(np.diagflat(j_ad_t), idftm))

            # Calculation of a new vector of arguments.
            v_ad_f = v_ad_f - hbm.left_division(j_ad_f, cev)

            # print("OneAD._get_vi_ad. v_ad_f=", v_ad_f)  # Test.
            v_ad_t = self._f2t_array(v_ad_f)  # Time samples of AD voltage.

        # HB counter: increment the numbers of steady-state response
        # evaluations and HB iterations.
        if self._has_hb_counter():
            self._incr_ss_evals()
            self._incr_hb_iters(hb_iters=iteration)

        # This is a pretty convenient output for debugging.
        print("_get_vi_ad. try_linear={0}, cev_max={1}, iter={2}".format(try_linear, cev_max, iteration))
        return (v_ad_t, i_ad_t)

        # The end of "OneAD._get_vi_ad" method.

    def _make_sim_db_manager(self):
        """
        Make a simulation database manager object.
        Use it to create a convenient set of electrical characteristics from
        raw simulation data.

        Returns
        -------
        sim_db_mgr : SimDBManager
            A simulation database manager object.
        """
        return SimDBManager(freq2time=self._f2t_array)

    # The end of "OneAD" class.
