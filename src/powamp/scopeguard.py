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
ScopeGuard pattern.

To learn more about ScopeGuard pattern, see
1. "Hands-On Design Patterns with C++", Fedor G. Pikus, 2019, chapter 11,
   "ScopeGuard".
2. An advanced implementation of this pattern in C++
   https:/​/​github.​com/​facebook/​folly/​blob/​master/​folly/​ScopeGuard.​h
3. "A "scopeguard" for Python", 2010.03.03,
   https://www.thecodingforums.com/threads/a-scopeguard-for-python.716620/

It is also important to know about the context manager protocol in Python. See
"Context Managers and Python's with Statement" by Leodanis Pozo Ramos,
on 2023.08.21, https://realpython.com/python-with-statement/

About "copy" and "deepcopy" overloading, see
1. "How to override the copy/deepcopy operations for a Python object",
   2009.09.30,
   https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
2. "Deep and Shallow Copies of Objects", Aquiles Carattino, 2019.02.04,
   https://pythonforthelab.com/blog/deep-and-shallow-copies-of-objects/

It is not possible to overload the assignment operator. See
"Is it possible to overload Python assignment", 2012.06.13,
https://stackoverflow.com/questions/11024646/is-it-possible-to-overload-python-assignment

2023.08.30
I do not know how to catch an external exception in a destructor. So that I
could not implement "ScopeGuardAuto" that does not require the usage of a
"with" statement.

2023.08.31
I recommend to use all of these classes as context managers. I. e. with a
"with" statement.
"""


class ScopeGuardManual:
    """
    ScopeGuard with manual control.
    It can be used with a "with" statement or separately.
    In both cases it requires to use a "commit" method call to prove that the
    ScopeGuard object can be deactivated.
    """

    def __init__(self, func):
        """
        Construct a ScopeGuard object.

        Parameters
        ----------
        func : callable
            The main usage of that function is to restore the initial state of
            an object or some other data. However, it can do any other required
            action.
            It can be a free function, lambda, or functional object.
        """
        if not callable(func):
            raise TypeError(
                "Cannot set a function that restores the initial state.\n"
                "An object must be callable.")
        self.__func = func
        self.__active = True

    def __del__(self):
        """
        An object's destructor.
        If a ScopeGuard object has not been commited, it will call a specified
        function. Otherwise, it does nothing.
        """
        self.__try_reset()

    def __enter__(self):
        """
        A part of the context manager protocol.

        Returns
        -------
        "self" reference to the caller object.
        """
        return self

    def __exit__(self, *exc):
        """
        A part of the context manager protocol.
        All the parameters here are unused.
        If there was no commitment call, it will call a specified function.
        If an exception within the body of a "with" statement occures, it will
        propagate outside the statement.

        Parameters
        ----------
        exc : tuple
            An exception parameters.
        """
        self.__try_reset()

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def __deepcopy__(self):
        """
        A "deepcopy" operation is disabled.
        """
        raise TypeError("A 'deepcopy' operation is not supported.")

    def commit(self):
        """
        Deactivate a specified function call. After calling this method,
        nothing will happen after the ScopeGuard object is out of its scope.
        """
        self.__active = False

    def __try_reset(self):
        """
        Try to call a specified function.
        If the ScopeGuard object has not been commited, it will call the
        function it stores. Otherwise, nothing will happen.
        """
        if self.__active:
            try:
                # Disarm to aviod the function execution during a deletion
                # operation if it has been executed during an exit operation.
                self.__active = False
                self.__func()
            except:
                raise RuntimeError(
                    "An error occurred during a reset procedure.\n"
                    "The initial state may not be restored properly.")

    # The end of "ScopeGuardManual" class.


# Note:
# I actually do not know how to detect whether an exception is already
# propagating or not in a destructor without using "try-except" block outside
# the object.
# If it is possible, then it is possible to implement it and use this
# ScopeGuard version without a "with" statement.
class ScopeGuardAuto:
    """
    ScopeGuard with automatic control driven by an exception.
    It can be used only as a context manager with a "with" statement.
    It is activated automatically after an exception has been thrown inside a
    "with" scope.
    """

    def __init__(self, func):
        """
        Construct a ScopeGuard object.

        Parameters
        ----------
        func : callable
            The main usage of that function is to restore the initial state of
            an object or some other data. However, it can do any other required
            action.
            It can be a free function, lambda, or functional object.
        """
        if not callable(func):
            raise TypeError(
                "Cannot set a function that restores the initial state.\n"
                "An object must be callable.")
        self.__func = func

    def __enter__(self):
        """
        A part of the context manager protocol.

        Returns
        -------
        "self" reference to the caller object.
        """
        return self

    # To make it possible to act automatically, it must somehow detect
    # whether there is an exception or not.
    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        A part of the context manager protocol.
        Only the first parameter is used.
        If an exception within the body of a "with" statement occures, this
        method will automatically call a specified funtion. The exception will
        propagate outside the statement.

        Parameters
        ----------
        exc_type : type
            The type of an exception being handled.
        exc_value : BaseException
            An exception instance.
        exc_tb : traceback
             A traceback object which encapsulates the call stack at the point
             where the exception originally occurred.
        """
        if exc_type is not None:
            self.__reset()
            return False

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def __deepcopy__(self):
        """
        A "deepcopy" operation is disabled.
        """
        raise TypeError("A 'deepcopy' operation is not supported.")

    def __reset(self):
        """
        Call a specified function.
        """
        try:
            self.__func()
        except:
            raise RuntimeError(
                "An error occurred during a reset procedure.\n"
                "The initial state may not be restored properly.")

    # The end of "ScopeGuardAuto" class.
