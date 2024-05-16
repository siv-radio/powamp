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

from pathlib import Path

from setuptools import setup


# The name of the package.
PKG_NAME = "powamp"

# The absolute path to the current directory.
here = Path(__file__).parent.absolute()

with open(here.joinpath("README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(here.joinpath("requirements.txt"), encoding="utf-8") as file:
    requirements = file.read().splitlines()

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# https://realpython.com/python-exec/
with open(here.joinpath("src", PKG_NAME, "version.py"), encoding="utf-8") as file:
    global_params = {'__builtins__': None}
    local_params = dict()
    exec(file.read(), global_params, local_params)
    PKG_VERSION = local_params['PKG_VERSION']


setup(
    name=PKG_NAME,
    version=PKG_VERSION,
    description="PowAmp - a library to perform calculations of power amplifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Igor Sivchek",
    author_email="sivradiotech@gmail.com",
    license="AGPL-3.0-only",
    url="https://github.com/siv-radio/powamp",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Education"
    ],
    keywords=["power amplifier", "harmonic balance", "optimization", "class E", "class EF2", "class EF3"],
    python_requires=">=3.7",
    package_dir = {'': 'src'},
    packages = ['examples', 'powamp'],
    install_requires=requirements
)
