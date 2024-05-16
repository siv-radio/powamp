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
Harmonic balance mathematics.
A basic library for building harmonic balance models of power amplifiers.
"""

import numpy as np

__all__ = [
    'CX_NAN', 's2h', 'h2s', 'realpair2complex', 'make_admittance_matrix',
    'f2w', 'w2f', 'pulse_duration', 'sample_duration', 'samples_per_pulse',
    'make_time_samples', 'make_freq_bins', 'left_division', 'DFT', 'HBCntMgr']

# Complex not a number constant.
CX_NAN = complex(np.nan, np.nan)


def s2h(s):
    """
    It returns a number of using harmonics including the 0th and 1st ones.
    The main equation: 2*h = s + 1.

    Parameters
    ----------
    s : int
        A number of time samples.
        "s" is an odd number, s >= 3.

    Returns
    -------
    h : int
        A number of harmonics.
    """
    # // - floor division operator. Its result is always a whole number.
    return (s + 1)//2


def h2s(h):
    """
    It returns a number of using time samples.
    The main equation: 2*h = s + 1.

    Parameters
    ----------
    h : int
        A number of using harmonics including the 0th and 1st ones.
        h >= 2.

    Returns
    -------
    s : int
        A number of time samples.
    """
    # h = (s + 1)//2, 2*h = s + 1, 2*h - 1 = s.
    return 2*h - 1


def realpair2complex(arr):
    """
    Convert an array of pairs of real numbers into an array of complex numbers.

    Parameters
    ----------
    arr : ndarray or list
        An array of real and imaginary parts of harmonics. Its size is "s_len",
        where s_len - number of time samples, it is an odd number,
        s_len >= 3.
        [re0, re1, im1, re2, im2, ..., re<h_len-1>, im<h_len-1>], where
        h_len - number of harmonics.

    Returns
    -------
    result : ndarray
        An array of complex numbers of harmonics. Its size is "h_len", where
        h_len - number of harmonics.
        [cx0, cx1, cx2, ..., cx<h_len-1>], where
        "cx0" always has only a real part,
        cx = re0, cx1 = re1 + 1j*im1, cx2 = re2 + 1j*im2, ...,
        cx<h_len-1> = re<h_len-1> + 1j*im<h_len-1>.
    """
    h_len = s2h(len(arr))
    result = np.zeros((h_len,), dtype=complex)
    result[0] = arr[0]  # The 0th harmonic.
    for h in range(1, h_len):
        result[h] = arr[2*h - 1] + 1j*arr[2*h]
    return result


def make_admittance_matrix(*, adm_func, h_len, w_wrk):
    """
    The function calculates an admittance matrix (2-D array).

    Parameters
    ----------
    adm_func : callable
        An admittance function of angular frequency "y(w)".
    h_len : int
        A number of harmonics.
    w : float
        A working angular frequency. It relates to the 1st harmonic.

    Returns
    -------
    y : ndarray
        An admittance matrix. Its size is (s_len, s_len), where
        s_len = 2*h_len - 1.
        Each cell contains a real floating point number.
    """
    s_len = h2s(h_len)  # Number of using samples in time domain.
    y = np.zeros((s_len, s_len))  # Creation of a zero Y matrix.
    y[0, 0] = adm_func(0.0).real
    for h in range(1, h_len):
        y_cx = adm_func(h*w_wrk)  # "cx" means "complex".
        y[2*h - 1, 2*h - 1] = y_cx.real
        y[2*h, 2*h] = y_cx.real
        y[2*h - 1, 2*h] = -y_cx.imag
        y[2*h, 2*h - 1] = y_cx.imag
    return y


def f2w(f):
    """
    It returns the angular frequency of a given frequency.
    w = 2*pi*f, where
    w - angular frequency,
    f - frequency.

    Parameters
    ----------
    f : float
        A frequency.

    Returns
    -------
    w : float
        An angular frequency.

    Notes
    -----
    "w" is just because it looks similar to Greek "omega" small letter.
    """
    return 2*np.pi*f


def w2f(w):
    """
    It returns the frequency of a given angular frequency.
    f = w/(2*pi), where
    f - frequency,
    w - angular frequency.

    Parameters
    ----------
    w : float
        An angular frequency.

    Returns
    -------
    f : float
        A frequency.

    Notes
    -----
    "w" is just because it looks similar to Greek "omega" small letter.
    """
    return w/(2*np.pi)


def pulse_duration(*, dc, t_wrk):
    """
    It returns a pulse duration.
    tp = dc*t_wrk, where
    dc - duty cycle,
    t_wrk - time period.

    Parameters
    ----------
    dc : float
        A duty cycle (i. e. share of time "t_wrk" when an active device is in
        conductive state).
        "dc" is in (0, 1) range.
    t_wrk : float
        A time period.

    Returns
    -------
    tp : float
        A pulse duration.
    """
    return dc*t_wrk


def sample_duration(*, s_len, t_wrk):
    """
    It returns a time sample duration.
    dt = t_wrk/s_len, where
    t_wrk - time period,
    s_len - number of time samples.

    Parameters
    ----------
    s_len : int
        A number of time samples.
    t_wrk : float
        A time period.

    Returns
    -------
    dt : float
        A time sample duration.
    """
    return t_wrk/s_len


def samples_per_pulse(*, s_len, dc):
    """
    It returns a number of time samples during which an active device is in
    conductive state.
    sp = ceil(dc*s_len), where
    dc - duty cycle,
    s_len - number of time samples,
    ceil - ceiling function.

    Parameters
    ----------
    s_len : int
        A number of time samples.
    dc : float
        A duty cycle (i. e. the share of a time period when an active device is
        in conductive state).

    Returns
    -------
    sp : int
        A number of time samples during which an active device is in conductive
        state.
    """
    return int(np.ceil(dc*s_len))


def make_time_samples(*, s_len, t_wrk):
    """
    Make an array of time samples.
    tsarr = [0*dt, 1*dt, 2*dt, ..., (s_len - 1)*dt],
    dt = t_wrk/s_len, where
    s_len - number of time samples,
    t_wrk - time period.

    Parameters
    ----------
    t_wrk : float
        A time period.
    s_len : int
        A number of time samples.

    Returns
    -------
    tsarr : ndarray
        An array of time samples. Its size is "s_len".
    """
    dt = t_wrk/s_len  # Timestep.
    return dt*np.arange(s_len)


def make_freq_bins(*, h_len, t_wrk):
    """
    Get an array of frequency bins.
    fbarr = [0*f, 1*f, 2*f, ..., (h_len - 1)*f],
    f = 1/t_wrk, where
    h_len - number of harmonics,
    t_wrk - time period.

    Parameters
    ----------
    h_len : int
        A number of harmonics (and the bins).
    t_wrk : float
        A time period.

    Returns
    -------
    fbarr : ndarray
        An array of frequency bins. Its size is "h_len".
    """
    f_wrk = 1.0/t_wrk  # Frequency.
    return f_wrk*np.arange(h_len)


def left_division(lhs, rhs):
    """
    Produce the left divison operation on "numpy.ndarray" objects.
    result = lhs\rhs.
    It is an equivalent of
    result = inv(lhs)*rhs, where
    inv - matrix inversion function.
    However the left division operation is more computationally efficient than
    a matrix inversion operation with a consequent multiplication.

    Parameters
    ----------
    lhs : ndarray
        A matrix of size (m_len, m_len).
    rhs : ndarray
        A matrix of size (m_len, p_len).

    Returns
    -------
    result : ndarray
        A matrix of size (m_len, p_len).

    Notes
    -----
    This function is a wrapper of "numpy.linalg.lstsq".
    It throws an exception if a computation does not converge.
    """
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


# Usage of a separate DFTM and an IDFTM provides better convergence. For
# example, it throws an exception when only an IDFTM is used and h_len = 1024.
# There is no this problem when both of the matrices are used.
class DFT:
    """
    DFT - discrete Fourier transform.
    It provides a forward DFT matrix (DFTM) and an inverse DFT matrix (IDFTM).
    h_len - (integer not less than 2) number of harmonics.
    The sizes of the output matrices is (s_len, s_len), where
    s_len = 2*h_len - 1.
    It uses caching concept. A DFTM and an IDFTM are built only once until
    "h_len" changes.
    """

    def __init__(self, h_len=None):
        """
        A DFT constructor.

        Parameters
        ----------
        h_len : int
            A number of harmonics. h_len >= 2.
            If "h_len" is "None" than it constructs an empty object. Otherwise,
            it builds a DFTM and an IDFTM for this number of harmonics.
        """
        self.__h_len = 0  # (Integer not less than 2) number of harmonics.
        self.__valid = False  # "True" if the DFTM and IDFTM are valid and "False" otherwise.
        self.__fwd_mtx = None  # The forward DFT matrix, "numpy.ndarray".
        self.__inv_mtx = None  # The inverse DFT matrix, "numpy.ndarray".
        # print("DFT.__init__(), __fwd_mtx:", self.__fwd_mtx)  # Test.
        # print("DFT.__init__(), __inv_mtx:", self.__inv_mtx)  # Test.
        if h_len is not None:
            self.__h_len = h_len
            # Lazy evaluations. The matrices will be built only on demand.

    def __copy__(self):
        """
        Disable a "copy" operation.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def set_dft_length(self, h_len):
        """
        Set a (integer not less than 2) number of harmonics.

        Parameters
        ----------
        h_len : int
            A number of using harmonics including the 0th and 1st ones.
            h_len >= 2.

        Returns
        -------
        "self" reference to the caller object.
        """
        if h_len < 2:
            raise ValueError(
                "A number of harmonics must be not less than 2, "
                "but given: {0}.".format(h_len))
        if h_len != self.__h_len:
            self.__h_len = h_len
            self.__valid = False
        return self

    def get_dft_length(self):
        """
        Get a number of harmonics.

        Returns
        -------
        h_len : int
            A number of using harmonics including the 0th and 1st ones.
        """
        return self.__h_len

    def forward(self, h_len=None):
        """
        It returns a DFTM.

        Parameters
        ----------
        h_len : int
            A number of harmonics. h_len >= 2.
            If "h_len" is "None" then it returns a reference to cached data.

        Returns
        -------
        dftm : ndarray
            A DFTM has size (s_len, s_len), where s_len - number of
            time samples.
        """
        self.__process(h_len)
        return self.__fwd_mtx

    def inverse(self, h_len=None):
        """
        It returns an IDFTM.

        Parameters
        ----------
        h_len : int
            A number of harmonics. h_len >= 2.
            If "h_len" is "None" then it returns a reference to cached data.

        Returns
        -------
        dftm : ndarray
            An IDFTM has size (s_len, s_len), where s_len - number of
            time samples.
        """
        self.__process(h_len)
        return self.__inv_mtx

    def t2f(self, ta):
        """
        Convert a time domain array "ta" into a frequency domain array "fa"
        using a discrete Fourier transform matrix "dftm".
        fa = dftm*ta, where
        dftm - a discrete Fourier transform matrix of size (s_len, s_len),
        ta - a time domain array of size (s_len, cols),
        fa - a frequency domain array of size (s_len, cols),
        s_len - number of time samples,
        cols - number of columns in "ta" array.

        Parameters
        ----------
        ta : ndarray
            An array in time domain. Its size is (s_len, cols).

        Returns
        -------
        fa : ndarray
            An array in frequency domain. Its size is (s_len, cols).

        Notes
        -----
        Probably, the most often usage of this method is to convert a time
        series "ta" into an array of harmonics "fa". In this case
        fa = dftm*ta, where
        dftm - a discrete Fourier transform matrix of size (s_len, s_len),
        ta - a time series array of size "s_len",
        fa - a harmonics array of size "s_len".
        """
        return np.dot(self.forward(), ta)

    def f2t(self, fa):
        """
        Convert a frequency domain array "fa" into a time domain array "ta"
        using an inverse discrete Fourier transform matrix "idftm".
        ta = idftm*fa, where
        idftm - an inverse discrete Fourier transform matrix of size
        (s_len, s_len),
        fa - a frequency domain array of size (s_len, cols),
        ta - a time domain array of size (s_len, cols),
        s_len - number of time samples,
        cols - number of columns in "fa" array.

        Parameters
        ----------
        fa : ndarray
            An array in frequency domain. Its size is (s_len, cols).

        Returns
        -------
        ta : ndarray
            An array in time domain. Its size is (s_len, cols).

        Notes
        -----
        Probably, the most often usage of this method is to convert an array of
        harmonics "fa" into a time series "ta". In this case
        ta = idftm*fa, where
        idftm - an inverse discrete Fourier transform matrix of size
            (s_len, s_len),
        fa - a harmonics array of size "s_len",
        ta - a time series array of size "s_len".
        """
        return np.dot(self.inverse(), fa)

    def select_harmonic(self, *, ta, h):
        """
        Select a required h-th harmonic from the time domain array "ta".

        The full array conversion:
        fa = dftm*ta, where
        dftm - a discrete Fourier transform matrix (DFTM) of size
            (s_len, s_len),
        ta - a time domain array of size (s_len, cols),
        fa - a frequency domain array of size (s_len, cols),
        s_len - number of time samples,
        cols - number of columns in "ta" array.
        In case of the operation on selecting a harmonic, it multiplies only
        one (h == 0) or two (h > 0) rows from a DFTM by "ta" array.

        Parameters
        ----------
        ta : ndarray
            An array in time domain. Its size is (s_len, cols).
        h : int
            A required harmonic number.

        Returns
        -------
        harm : float or ndarray of float
            There are several cases of output format:
            1. "ta" is an 1-D array of (s_len,) size and (h == 0). The result
               is a floating point number that represents the real part of the
               0th harmonic.
            2. "ta" is an 1-D array of (s_len,) size and (h > 0). The result is
               an array of (2,) size that contains the real and imaginary parts
               of the h-th harmonic.
            3. "ta" is a 2-D array of (s_len, cols) size and (h == 0). The
               result is an array of (cols,) size that contains the real parts
               of the 0th harmonics.
            4. "ta" is a 2-D array of (s_len, cols) size and (h > 0). The
               result is an array of (2, cols) size that contains the real (the
               0th row) and imaginary (the 1st row) parts of the h-th
               harmonics.

        Notes
        -----
        Sometimes there is a necessity to find only a certain harmonic value.
        For example, to find the 1st harmonic of a voltage or current that
        relates to the working frequency. This method is useful in that case,
        because it avoids the full calculation of the array in frequency
        domain.
        """
        dftm = self.forward()  # A reference to the DFTM.
        if h == 0:
            # Calculate the 0th harmonic (without the imaginary part).
            return np.dot(dftm[0, :], ta)
        # Calculate the h-th harmonic, where (h > 0).
        s = h2s(h)
        return np.dot(dftm[s : s + 2, :], ta)

    def __process(self, h_len):
        """
        Check if the matrices are built, and build them if they are not or
        "h_len" value is different.
        """
        if h_len is None:
            if not self.__valid:
                # Build a DFTM and an IDFTM with the current number of
                # harmonics.
                self.__build()
        else:
            if h_len != self.__h_len:
                self.__h_len = h_len
                self.__build()

    def __build(self):
        """
        Build a DFTM and an IDFTM.
        """
        s_len = h2s(self.__h_len)  # Number of using samples in time domain.
        # Creation of a DFTM and an IDFTM.
        # The matrices creation was checked on a matrices of size (5, 5).
        self.__fwd_mtx = np.zeros((s_len, s_len))  # DFTM creation.
        self.__fwd_mtx[0, :] = 1/s_len  # Fill the 0th row of the matrix with ones.
        self.__inv_mtx = np.zeros((s_len, s_len))  # IDFTM creation.
        self.__inv_mtx[:, 0] = 1.0  # Fill the 0th column of the matrix with ones.
        s = np.arange(s_len)  # Numbers of time samples.
        for h in range(1, self.__h_len):  # Numbers of harmonics.
            # h*w*s*dt = 2*pi*h*s/s_len.
            odd = np.cos(2*np.pi*h*s/s_len)
            even = -np.sin(2*np.pi*h*s/s_len)
            self.__fwd_mtx[2*h - 1, :] = odd*2/s_len
            self.__fwd_mtx[2*h, :] = even*2/s_len
            self.__inv_mtx[:, 2*h - 1] = odd
            self.__inv_mtx[:, 2*h] = even
        self.__valid = True

# The end of "DFT" class.


class HBCntMgr():
    """
    Harmonic balance (HB) counter manager.
    A related object is used to count the numbers of steady-state (SS) reponse
    evaluations and HB iterations. It requires to have at least one counter to
    start common counting of SS evaluations and HB iterations. Each counter has
    its own starting position and can be used to count the numbers throughout
    the related procedures. Several counters created in different procedures
    provide the ability to count the particular and total numbers. When no
    counters left, the manager resets the numbers that it stores.

    An object is non-copyable.
    It can work properly only in a single-threading program.
    There are no cyclical references.
    """

    class __HBCounter():
        """
        Harmonic balance (HB) counter.
        It counts the numbers of steady-state (SS) reponse evaluations and HB
        iterations from the moment of an instance creation.

        An object is non-copyable.
        """

        def __init__(self, *, mgr, ssev_offset, hbit_offset):
            """
            Constructor.

            Parameters
            ----------
            mgr : HBCntMgr
                Reference to the realated manager that creates the counter.
            ssev_offset : int
                Current number of SS evaluations. It is used as zero level for
                the counter.
            hbit_offset : int
                Current number of HB iterations. It is used as zero level for
                the counter.
            """
            self.__mgr = mgr  # Reference to a "HBCntMgr" instance.
            self.__ssev_offset = ssev_offset
            self.__hbit_offset = hbit_offset

        def __del__(self):
            """
            Destructor.
            It is used to tell the related "HBCntMgr" object that a counter
            is deleting.
            """
            self.__mgr._decr_count()

        def __copy__(self):
            """
            Disable a "copy" operation.
            """
            raise TypeError("A 'copy' operation is not supported.")

        def __deepcopy__(self):
            """
            Disable a "deepcopy" operation.
            """
            raise TypeError("A 'deepcopy' operation is not supported.")

        def get_ss_evals(self):
            """
            Get a number of steady-state (SS) reponse evaluations.

            Returns
            -------
            ss_evals : int
                A number of SS evaluations.
            """
            return self.__mgr._get_ss_evals() - self.__ssev_offset

        def get_hb_iters(self):
            """
            Get a number of harmonic balance (HB) iterations.

            Returns
            -------
            hb_iters : int
                A number of HB iterations.
            """
            return self.__mgr._get_hb_iters() - self.__hbit_offset

        # The end of "HBCntMgr.__HBCounter" class.

    def __init__(self):
        """
        Constructor.
        Creates a harmonic balance (HB) counter manager.
        The numbers of steady-state (SS) response evaluations and HB iterations
        are set to zeros.
        No related counters exist after a manager creation.
        """
        self.__ss_evals = 0
        self.__hb_iters = 0
        self.__cnt = 0  # A counter of instances.

    def __copy__(self):
        """
        Disable a "copy" operation.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def __deepcopy__(self, memo={}):
        """
        "deepcopy" operation creates a new instance of this class with default
        settings.

        memo : dict
            To keep track on the object that are already copied during a
            "deepcopy" procedure and prevent an infinite loop.
        """
        # Get the type of this class.
        cls = self.__class__
        # Create and return a new instance of this class with default settings.
        return cls.__new__(cls)

    def has_counter(self):
        """
        Tells whether a counter exists or not.

        Returns
        -------
        flag : bool
            "True" if at least one related counter exists, "False" otherwise.
        """
        return self.__cnt > 0

    def make_counter(self):
        """
        Create a counter.
        A counter will start counting with the current numbers of steady-state
        (SS) response evaluations and harmonic balance (HB) iterations.

        Returns
        -------
        hb_cnt : HBCounter
            An instance of an HB counter.
        """
        self.__cnt += 1
        return self.__HBCounter(
            mgr=self,
            ssev_offset=self.__ss_evals, hbit_offset=self.__hb_iters)

    def incr_ss_evals(self, ss_evals):
        """
        Increase the total number of steady-state (SS) response evaluations by
        a given value.
        It affects on the data that will be provided by all existing counters.

        Paramters
        ---------
        ss_evals : int
            A number of SS evaluations to add.
        """
        self.__ss_evals += ss_evals

    def incr_hb_iters(self, hb_iters):
        """
        Increase the total number of harmonic balance (HB) iterations by a
        given value.
        It affects on the data that will be provided by all existing counters.

        Paramters
        ---------
        hb_iters : int
            A number of HB iterations to add.
        """
        self.__hb_iters += hb_iters

    def _get_count(self):
        """
        Get a number of existing counters.

        Returns
        -------
        cnt : int
            A number of existing counters.
        """
        return self.__cnt

    def _decr_count(self):
        """
        Decrease a number of existing counters by 1.
        It is used when a counter is deleting.
        If no counters left, it will reset the common counters inside the
        manager.
        """
        self.__cnt -= 1
        # "<=" instead of "==" is a precaution.
        if self.__cnt <= 0:     self.__reset()

    def _get_ss_evals(self):
        """
        Get the current total number of steady-state (SS) harmonic evaluations.

        Returns
        -------
        ss_evals : int
            A number of SS evaluations.
        """
        return self.__ss_evals

    def _get_hb_iters(self):
        """
        Get the current total number of harmonic balance (HB) iterations.

        Returns
        -------
        hb_iters : int
            A number of HB iterations.
        """
        return self.__hb_iters

    def __reset(self):
        """
        Reset the common numbers of steady-state (SS) response evaluations and
        harmonic balance (HB) iterations.
        It is used when no counters left.
        """
        self.__ss_evals = 0
        self.__hb_iters = 0

    # The end of "HBCntMgr" class.
