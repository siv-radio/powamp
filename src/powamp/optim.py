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
Optimization tools for the power amplifier library.
"""

from collections import namedtuple

__all__ = ['cmp_lt', 'cmp_gt', 'optimize', 'FResult']

Optimization = namedtuple(
    'Optimization', 'success status message x f fstate nfev nit')

FResult = namedtuple('FResult', 'success value state')
# Note: "FResult" can be used in a goal function as an output data structure.


def cmp_lt(lhs, rhs):
    """
    "Lower than" (<) comparator.
    Useful to minimize a goal function.

    Parameters
    ----------
    lhs : comparable
        Left hand side value.
    rhs : comparable
        Right hand side value.

    Returns
    -------
    result : bool
        The result of a comparison. "True" if (lhs < rhs), "False" otherwise.
    """
    return lhs < rhs


def cmp_gt(lhs, rhs):
    """
    "Greater than" (>) comparator.
    Useful to maximize a goal function.

    Parameters
    ----------
    lhs : comparable
        Left hand side value.
    rhs : comparable
        Right hand side value.

    Returns
    -------
    result : bool
        The result of a comparison. "True" if (lhs > rhs), "False" otherwise.
    """
    return lhs > rhs


# To grasp the basic idea of the algorithm below with ease, see
# "Find the maximum element in an array which is first increasing and then
# decreasing", "Method 3 (Binary Search – Iterative Solution)",
# www.geeksforgeeks.org, 2023.08.14.
# https://www.geeksforgeeks.org/find-the-maximum-element-in-an-array-which-is-first-increasing-and-then-decreasing/
# Note: the algorithm below cannot correctly process the case when there are
# several consecutive "x" values that give the same "f(x)" value.
class Optimizer:
    """
    Optimization of a goal function of one discrete argument.
    """

    def __init__(self):
        self.func = None  # Goal function to optimize.
        self.x_begin = 0  # Left bound "x" value.
        self.x_end = 0  # The 1st "x" value outside the right bound.
        self.x_init = 0  # Initial condition.
        self.x_cur = None  # Current "x" value.
        self.maxstep = None  # Maximum "x" step size.
        self.comp = None  # Comparator.
        self.f_cache = dict()  # Results of function evaluations.
        self.nfev = 0  # Number of function evaluations.
        self.nit = 0  # Number of algorithm iterations.

    def __copy__(self):
        raise TypeError("A 'copy' operation is not supported.")

    def run(self):
        x_begin = self.x_begin
        x_end = self.x_end
        x_len = x_end - x_begin  # Number of points in an "x" array.

        # Check if an "x" range is empty. The fastest answer if it is.
        if x_len == 0:          return self.__result_empty_range()

        # Starting point (initial guess).
        if self.x_init is None: x_mid = x_begin + (x_end - x_begin)//2
        else:                   x_mid = self.x_init

        for _ in range(x_len):  # To limit the cycle.
            self.nit += 1  # Increase the counter of algorithm iterations.
            # Termination condition if there are less than 4 elements left.
            if x_end - x_begin < 4:
                return self.__direct_search(x_begin=x_begin, x_end=x_end)
            f_mid = self.__get_f_res(x_mid)
            if not f_mid.success:
                return self.__result_func_fail(x=x_mid, f=f_mid)
            # Left point (x_left).
            x_left = x_mid - 1
            f_left = self.__get_f_res(x_left)
            if not f_left.success:
                return self.__result_func_fail(x=x_left, f=f_left)
            if self.comp(f_mid.value, f_left.value):  # (f_left > f_mid) by default.
                # Right point (x_right).
                x_right = x_mid + 1
                f_right = self.__get_f_res(x_right)
                if not f_right.success:
                    return self.__result_func_fail(x=x_right, f=f_right)
                if self.comp(f_mid.value, f_right.value):  # f_left > f_mid < f_right.
                    # The minimum has been found.
                    return self.__result_success(x=x_mid, f=f_mid)
                else:  # f_left > f_mid >= f_right. Move to the right.
                    x_begin = x_mid + 1
                    direction = 'r'
            else:  # f_left <= f_mid. Move to the left.
                x_end = x_mid
                direction = 'l'
            # Middle point (x_mid).
            # If there is no maximum step size limit:
            # x_mid = x_begin + (x_end - x_begin)//2
            # Otherwise, a special function is used.
            x_mid = self.__go_mid(x_begin=x_begin, x_end=x_end, direction=direction)

        return self.__result_no_solution(x=self.x_cur, f=self.__get_f_res(self.x_cur))

    # Auxiliary methods.

    def __get_f_res(self, x):
        self.x_cur = x  # Current "x" value.
        # Calculate an f(x) value, cache it, and increase the counter.
        f_res = self.f_cache.get(x)
        if f_res is None:
            f_res = self.func(x)
            f_res = FResult(success=f_res[0], value=f_res[1], state=f_res[2])
            self.f_cache[x] = f_res
            self.nfev += 1  # Increase the counter of function calls.
        return f_res

    def __direct_search(self, x_begin, x_end):
        # Use it when (x_end - x_begin < 4).
        x_root = None
        f_root = FResult(success=None, value=None, state=False)
        for x in range(x_begin, x_end):
            f_res = self.__get_f_res(x)
            if not f_res.success:
                return self.__result_func_fail(x=x, f=f_res)
            if not f_root.success:  # For the 1st step when there is no root.
                x_root = x
                f_root = f_res
            else:
                if self.comp(f_res.value, f_root.value):  # (f_val < f_root) by default.
                    x_root = x
                    f_root = f_res
        return self.__result_success(x=x_root, f=f_root)

    def __go_mid(self, x_begin, x_end, direction):
        # Get the middle point with a maximum step size restriction if it is
        # defined.
        # x_mid = x_begin + (x_end - x_begin)//2
        x_step = (x_end - x_begin)//2  # Step.
        if self.maxstep is not None and self.maxstep < x_step:  # If "x_step" is too big.
            if direction == 'l':
                return x_end - self.maxstep - 1
                # A sequence to check the correctness.
                # x:  0   1   2   3   4   5   6
            elif direction == 'r':
                return x_begin + self.maxstep
            else:
                raise ValueError("Unknown direction: {0}.".format(direction))
        return x_begin + x_step

    # Results.

    def __result_success(self, x, f):
        # Generate output if an optimization succeeded.
        return Optimization(
            success=True, status=0, message=
            "A soulution has been successfully found.",
            x=x, f=f.value, fstate=f.state, nfev=self.nfev, nit=self.nit)

    def __result_no_solution(self, x, f):
        # Generate output if no solution has been found.
        return Optimization(
            success=False, status=1, message=
            "No solution has been found.\n"
            "The iteration limit has been achieved.",
            x=x, f=f.value, fstate=f.state, nfev=self.nfev, nit=self.nit)

    def __result_func_fail(self, x, f):
        # Generate output if f(x) calculation is failed.
        return Optimization(
            success=False, status=2, message=
            "Optimization process has been interrupted.\n"
            "Cannot make the next step.\n"
            "Cannot calculate a value of the goal function.\n"
            "See the goal function parameters to learn more.",
            x=x, f=f.value, fstate=f.state, nfev=self.nfev, nit=self.nit)

    def __result_empty_range(self):
        # Generate output if "x" range is empty. I. e. (x_end - x_begin == 0).
        return Optimization(
            success=False, status=3, message=
            "No optimization produced.\n"
            "The range of 'x' variable is empty.",
            x=None, f=None, fstate=None, nfev=self.nfev, nit=self.nit)

    # The end of "Optimizer" class.


# It is convenient to use a wrapper around an "Optimizer" object.
def optimize(*, func, x_begin, x_end, x_init=None, maxstep=None, comp=cmp_lt):
    """
    Find a discrete "x" value that provides the optimum (minimum by default) of
    an "f(x)" function.
    It uses binary search method.
    It uses a caching mechanism to reduce the number of "f(x)" calls.

    Constraints:
    1. "x_begin" is less than "x_end".
        x_begin < x_end.
    2. No "x" values that provide the same "f(x)" values.
        f(x1) != f(x2) for any x1 != x2.
    3. "f(x)" can be of these types (in case of minimization):
        3.1. Monotonously increasing.
            f(x1) < f(x2) for any x1 < x2.
        3.2. Monotonously decreasing.
            f(x1) > f(x2) for any x1 < x2.
        3.3. Monotonously decreasing and then monotonously increasing.
            f(x1) > f(x2) for any x1 < x2 that belongs to [x_begin, x_root] and
            f(x1) < f(x2) for any x1 < x2 that belongs to [x_root, x_end),
            where x_root - a value that provides the minimum of f(x).
    Otherwise, there is no guarantee that a solution will be found.

    Parameters
    ----------
    func : callable
        A goal function to optimize with a signature:
        f(x) -> tuple(value, state, success)
        x : int
            An argument by which optimization produces.
        value : comparable by default
            An optimal value of a function call.
        state : any type
            Any additional information that relates to a certain function call.
            For example, it can store some internal parameters of a goal
            function that can be written in a result of an "optimize" call
            alongside with "x" and "f" values. It is helpful when a multilevel
            optimization is used.
        success : bool
            A flag that shows whether a goal function call is successful or
            not. It is helpful when a multilevel optimization is used.
    x_begin : int
        The first (left) value in an "x" range.
    x_end : int
        The first (right) value outside an "x" range.
    x_init : int, optional
        Initial guess about an "x" value.
    maxstep : int, optional
        The maximum "x" step. It limits the maximum "x" change between adjacent
        algorithm iterations. It is useful when "f(x)" has an internal
        algorithm that does not guarantee a correct solution and its result
        depends on an initial estimation. The closer two "x" values
        will be, the more chance that "f(x)" will be found at the next "x"
        value.
        If it is "None", no restrictions on a step size.
    comp : callable, optional
        Comparator for values of a goal function. It has a signature:
        comp(lhs, rhs) -> result
        lhs : comparable by default
            Left hand side argument.
        rhs : comparable by default
            Right hand side argument.
        result : bool
            The result of a comparison of "lhs" and "rhs" values.
        Comparable values here and in an "f(x)" result can be not actually
        comparable if a custom comparator avoids comparing operations inside.

    Returns
    -------
    result : Optimization
        A "namedtuple" that contains following fields:
        success : bool
            A flag that is "True" if a solution is found. "False" otherwise.
        status : int
            An integer flag. Set to "0" if a solution was found,
            otherwise refer to "message" for more information.
        message : str
            Contains a description of an algorithm launch.
        x : int
            A value that provides the optimum (minimum by default) of an "f(x)"
            function.
        f : comparable (float, int, etc.)
            The optimum (minimum by default) of an "f(x)" function.
        fstate : any type, None by default
            Additional information about an optimum goal function call.
        nfev : int
            Number of function calls.
        nit : int
            Number of algorithm iterations.

    Notes
    -----
    There are 4 different result variants:
    1. success = True, state = 0. An optimization is successful.
       "x", "f", and "fstate" contain values that relate to the solution.
    2. success = False, state = 1. An optimization has failed. The maximum
       number of iterations is exceeded, but no solution found. Possible reason
       is an inappropriate function behaviour that does not conform to the
       requirements.
       "x", "f", and "fstate" contain values that relate to the last function
       call.
    3. success = False, state = 2. A function call has failed.
       "x", "f", and "fstate" contain values that relate to the failed function
       call.
    4. success = False, state = 3. The "x" range is empty.
       "x", "f", and "fstate" contain "None".
    """
    opt = Optimizer()
    opt.func = func
    opt.x_begin = x_begin
    opt.x_end = x_end
    opt.x_init = x_init
    opt.maxstep = maxstep
    opt.comp = comp
    return opt.run()
    # The end of "optimize" function.
