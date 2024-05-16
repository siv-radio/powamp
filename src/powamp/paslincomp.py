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
Passive linear components.
"""

from collections import namedtuple

__all__ = ['Capacitor', 'Inductor']


class Capacitor:
    """
    A capacitor that consists of a capacitance and a resistance connected in
    parallel.
    """

    Params = namedtuple('CaprParams', 'c g')

    def __init__(self):
        self.__c = 1e-12
        self.__g = 1e-12

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def set_c(self, val):
        self.__c = float(val)
        return self

    def get_c(self):
        return self.__c

    def set_g(self, val):
        self.__g = float(val)
        return self

    def get_g(self):
        return self.__g

    def set_params(self, *, c=None, g=None):
        if c is not None:   self.__c = self.set_c(c)
        if g is not None:   self.__g = self.get_g(g)
        return self

    def get_params(self):
        return self.Params(c=self.__c, g=self.__g)

    def get_y(self, w):
        return self.__g + 1j*w*self.__c

    def get_z(self, w):
        return 1.0/self.get_y(w)


class Inductor:
    """
    An inductor that consists of an inductance and a resistance connected in
    series.
    """

    Params = namedtuple('IndrParams', 'l r')

    def __init__(self):
        self.__l = 1e-12
        self.__r = 1e-12

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def set_l(self, val):
        self.__l = float(val)
        return self

    def get_l(self):
        return self.__l

    def set_r(self, val):
        self.__r = float(val)
        return self

    def get_r(self):
        return self.__r

    def set_params(self, *, l=None, r=None):
        if l is not None:   self.__l = self.set_l(l)
        if r is not None:   self.__r = self.get_r(r)
        return self

    def get_params(self):
        return self.Params(l=self.__l, r=self.__r)

    def get_y(self, w):
        return 1.0/self.get_z(w)

    def get_z(self, w):
        return self.__r + 1j*w*self.__l
