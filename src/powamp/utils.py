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
Power amplifier (PA) utilities.
A library of tools for calculation of some electrical parameters and
characteristics.
"""

import numpy as np

__all__ = ['b_n_from', 'c_from', 'x_n_from', 'l_from', 'p_avg', 'rms', 'mpoc']


def b_n_from(*, c, r, w):
    """
    Calculate a normalized capacitive susceptance value from capacitance,
    resistance, and angular frequency.
    b_n = w*c*r, where
    w - angular frequency, c - capacitance, r - normalizing resistance.

    Parameters
    ----------
    c : float
        A capacitance value.
    r : float
        A normalizing resistance value.
    w : float
        An angular frequency value.

    Returns
    -------
    b_n : float
        A normalized susceptance value.
    """
    return w*c*r


def c_from(*, b_n, r, w):
    """
    Calculate a capacitance value from normalized capacitive susceptance,
    resistance, and angular frequency.
    c = b_n/(w*r), where
    b_n - normalized susceptance, w - angular frequency,
    r - normalizing resistance.

    Parameters
    ----------
    b_n : float
        A normalized susceptance value.
    r : float
        A normalizing resistance value.
    w : float
        An angular frequency value.

    Returns
    -------
    c : float
        A capacitance value.
    """
    return b_n/(r*w)


def x_n_from(*, l, r, w):
    """
    Calculate a normalized inductive reactance value from inductance,
    resistance, and angular frequency.
    x_n = w*l/r, where
    w - angular frequency, l - inductance, r - normalizing resistance.

    Parameters
    ----------
    l : float
        An inductance value.
    r : float
        A normalizing resistance value.
    w : float
        An angular frequency value.

    Returns
    -------
    x_n : float
        A normalized reactance value.
    """
    return w*l/r


def l_from(*, x_n, r, w):
    """
    Calculate an inductance value from normalized inductive reactance,
    resistance, and angular frequency.
    l = x_n*r/w, where
    x_n - normalized reactance, w - angular frequency,
    r - normalizing resistance.

    Parameters
    ----------
    x_n : float
        A normalized reactance value.
    r : float
        A normalizing resistance value.
    w : float
        An angular frequency value.

    Returns
    -------
    l : float
        An inductance value.
    """
    return x_n*r/w


def p_avg(*, v_t, i_t):
    """
    Get the average active (real) power in an electrical component from given
    vectors of voltage and current.
    p_avg = sum(v_t*i_t)/size, where
    v_t - array of voltage samples in time domain,
    i_t - array of current samples in time domain,
    size - number of samples in each of these vectors,
    sum() - summing operation,
    * - element-wise multiplication (Hadamar / Schur product).

    Parameters
    ----------
    v_t : ndarray or list
        An array of voltage samples in time domain.
    i_t : ndarray or list
        An array of current samples in time domain.

    Returns
    -------
    p_avg : float
        The average real power in a respective component.
    """
    if not isinstance(v_t, np.ndarray):     v_t = np.array(v_t)
    if not isinstance(i_t, np.ndarray):     i_t = np.array(i_t)
    return (v_t*i_t).mean()


def rms(arr):
    """
    Get the root mean square value of the numbers in a given array.
    rms = sqrt(sum(arr^2)/size), where
    arr - array of real numbers,
    size - size of the array,
    sum() - summing operation,
    sqrt() - square root operation.

    Parameters
    ----------
    arr : ndarray or list
        An array of real numbers.

    Returns
    -------
    rms : float
        A root mean square value.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return np.sqrt((arr**2).sum()/arr.size)


def mpoc(v_ad_max, i_ad_rms, p_out_h1):
    """
    Get a modified power output capability (MPOC or "cpmr") value.
    mpoc = p_out_h1/(v_ad_max*i_ad_rms), where
    p_out_h1 - power of the 1st harmonic in the load network,
    v_ad_max - the maximum voltage on the active device (AD),
    i_ad_rms - root mean square current through the AD.

    Parameters
    ----------
    v_ad_max : float
        The maximum voltage on an AD.
    i_ad_rms : float
        Root mean square current through an AD.
    p_out_h1 : float
        The 1st harmonic of real (active) output power.

    Returns
    -------
    mpoc : float
        An MPOC value.

    Notes
    -----
    Use MPOC to assess comparative efficiency of an AD usage in different
    power amplifiers. Since an AD may cost a considerable amount of money,
    it is important to use it as effective as it is possible.

    References
    ----------
    "High-Efficiency Class E, EF2, and E/F3 Inverters", Zbigniew Kaczmarczyk,
    2006.
    """
    return p_out_h1/(v_ad_max*i_ad_rms)
