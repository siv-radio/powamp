# PowAmp &ndash; Power Amplifier Library for Python
Date: 2024.05.16

*"For every complex problem, there's a solution that is simple, neat, and wrong."*  
Henry Louis Mencken, journalist, essayist

The solution presented here is not simple and neat. In theory, it has enough degree of freedom to describe some particular physical systems in a right way in order to solve respective problems.

## Overview
The library is aimed to facilitate research and development processes of some well known yet complex power amplifiers (PA). At the first release, it consists of configurable class E and EF<sub>x</sub> PA models with lumped elements. Built-in PA tuners provide the ability to find optimum working parameters of the PAs. The models are based on the harmonic balance (HB) analysis, which makes it possible to quickly calculate their steady-state responses without oversimplification of the real physical objects that they represent.

## 1. License
At the first release, the library is distributed under [AGPL v3 only](https://www.gnu.org/licenses/agpl-3.0.html) &ndash; a very strong copyleft license. It obligates developers to provide to users the source code of works based on this library.

## 2. Versioning
A version started with "0" here means that this library is at an experimental stage of development. So that all these versions will be published with a pre-release status (like "beta") [[link](https://peps.python.org/pep-0440/)]. If a next version "0.Yb\*" ("b" is short for "beta") emerges, there is no guaranty that it will have a backward compatibility with the previous version "0.Xb\*". It includes names, presence of attributes, the project structure, etc. Those who decided to use this library in their project, should work with the chosen library version "0.Xb\*" or be ready to redevelop their own code.

If the first non-experimental release occurs, it will have version "1.0.0". It will conform usual semantic rules "\<major\>.\<minor\>.\<patch\>" [[link](https://semver.org/)], which means successiveness of the application programming interface (API) within the same major version.

A lot of things should be done before this library can have a stable interface and a wide enough set of useful features.

There is no guarantee of the consistency of protected (the names started with an underscore "`_`" symbol) and private (the names started with two underscores "`__`") attributes. They are not considered here as part of the interface. It is recommended to use only public attributes.

## 3. Contributing
While this project is open source on its start, it is not aimed at a direct cooperation with other developers at least at the early (experimental) stage. There is a plan to implement several important features that can significantly impact on the library interface and its structure. These changes are not quantitative and beyond a straight extension of the library by adding new power amplifier models based on the current math. However, if someone encounters an error, they can use the [issue tracker](https://github.com/siv-radio/powamp/issues) to inform about it. Suggestions on improvement of this project are also valuable.

## 4. Open source software burnout
There is no guarantee that the development of the project will be continued and the project will not be abandoned. Any version can be the last one.

## 5. Introduction
There are many mathematical models of different power amplifiers (PAs). These models have different complexity and accuracy. On one side, we have a simple model of a class A PA, which does not require solving of differential equations and can be written on a tissue paper. On the other side, there is a class EF<sub>2</sub> PA model, which requires to use some advanced mathematical techniques and, probably, a computer. In the present-day, most of analytical PA models are built to describe a steady-state response to make it possible to find required parameters of an electrical circuit to achieve desirable electrical characteristics, while a transient process can be simulated on a computer and provides more precision.

Each mathematical model has its constraints, which limit its application. Those simplifications that lay in the foundation of the model must be satisfied in practice as close as possible. For example, there are many mathematical models of class E PAs, but usually they oversimplify some significant effects, which are observed in practice. Here are some known drawbacks of these models: no active device (AD) losses; inapplicable load network (LN) (a series resonant circuit, which gives very particular voltage and current waveforms at the PA output, while a task requires to have a sinusoidal output voltage); infinite DC-feed inductance. Some mathematical models are developed to take into account one of these aspects, but the author of this document does not know any analytical model that allows for all of these effects simultaneously. This is because it is hard to make even a simple model of this PA based on an analytical solution of a system of differential equations. Usage of numerical methods to solve these equations requires significant computational resources, because it is necessary to find a steady-state response (i. e. it is necessary to evaluate the entire transient process, which is mostly useless by itself).

The harmonic balance (HB) analysis [*"Nonlinear Microwave and RF Circuits", Stephen A. Maas, 2nd ed., 2003,* [link](https://us.artechhouse.com/Nonlinear-Microwave-and-RF-Circuits-Second-Edition-P1097.aspx)] provides an opportunity to build a complex mathematical model of a PA and find a solution quickly. It is very useful for optimization of PAs. A radio engineer can be acquainted with this method since it is implemented in some electrical circuit simulators like [Microwave Office](https://www.cadence.com/en_US/home/tools/system-analysis/rf-microwave-design/awr-microwave-office.html), [Advanced Design Systems](https://www.keysight.com/us/en/products/software/pathwave-design-software/pathwave-advanced-design-system/pathwave-ads-software-bundles.html) (ADS), [HSPICE](https://www.synopsys.com/implementation-and-signoff/ams-simulation/primesim-hspice.html), and [Xyce](https://xyce.sandia.gov/). The PowAmp library has the HB analysis in its base.

It would be incorrect to say that these HB models are analytical, because they are based on matrices and Newton method of solving a system of nonlinear equations. However, it would also be not true that these models are just numerical, because they are based on a particular non-universal matrix formula that is specifically made to build mathematical models of power amplifiers. Based on that, it is right to say that these models are somewhere in between and have a hybrid type &ndash; analytical-numerical ("anamerical"?).

## 6. The place of this library among other radio engineering tools
**Analytical models** have restricted forms and complexity. It usually takes a lot of time to build and verify them. They provide the fastest calculations. They are also the most comprehensive. It is very helpful for a developer to build their own analytical power amplifier (PA) model, even if other researchers did it many times before and published their results. Another important function of these models is that they can provide an initial estimation or parameters for work and optimization of more complex and realistic models in electrical circuit simulators.

**Computer algebra systems**, like [Mathcad](https://www.mathcad.com/en/), are convenient to work on analytical models. They increase working speed and reduce the number of errors that occur during the process of a model development. In fact, they can replace a working notebook. The main restriction is that they are not so convenient to implement complex models and research scenarios that consist of many steps and require respective programming procedures.

**[MATLAB](https://www.mathworks.com/products/matlab.html) / [GNU Octave](https://octave.org/)** are not very convenient to work on analytical models, although they provide this ability. However, they are relatively convenient to implement an existing analytical model and do research on it. It is possible to perform complex research scenarios. MATLAB Simulink can be used to build models of electrical circuits and use them in complex scenarios. A significant drawback of these programs is that they have an object-oriented programming (OOP) language that is far from perfection.

**Electrical circuit simulators** provide the ability to make models with unrestricted forms and complexity using graphical user interface (GUI) or a special SPICE-like programming language to define an electrical circuit and set simulation conditions. The simulators provide the fastest model building. Calculations of steady-state responses using the harmonic balance (HB) analysis are fast. The models are the least comprehensive. Operations and research opportunities on these models are restricted by simulator's abilities. For example, in case of ADS they can be relatively wide and convenient.

**PowAmp library** is written in Python, which is a full-fledged high level programming language of common purpose. The development, implementation, and debugging of a new PA model is hard. The models provide fast calculations and built-in optimizations. They are not much more comprehensive than models from electrical circuit simulators, although it is possible to conceive the source code. The PA models work out of the box, but the library requires a user to have at least basic programming skills in Python and study the application programming interface (API) of the library. The models provide the maximum flexibility of research scenarios using all Python OOP power.

It is nice for a researcher to make their own simplified mathematical model of a PA in order to understand its physics and limitations. This library can be used to find optimum parameters of some complex PA models and do complex research on them. A reliable electrical circuit simulator can be used on different purposes. First, to compare and check the results obtained from a user's analytical model or a model from this library. Second, to make an all-embracing computer model of an interesting PA and carry out some research on it, which is convenient to do using the simulator or can hardly be done otherwise.

This library is not made to replace analytical models, MATLAB, or ADS. It is made to supplement a toolbox of a modern radio engineer / scientist.

## 7. Installation instruction
*A programming tool with its description should not be a riddle from a David Lynch's movie.*

### 7.1. Notion
The author of this library is a scientist in radio technical hardware who does not have formal education in computer sciences. Programming skills were obtained mostly because of personal interest. This background gives understanding that it would not be right to tell other non-computer engineers and scientists that they can "simply" install this package using "pip" or "conda" into their virtual environment. Probably, these potential users just close the project page and forget about it (the author would definitely do that in their place). For many of them, it is a big feat to divide their MATLAB script into several files with functions to make the program more readable and flexible. Programming is not what they are paid for; a programming language for them is just a tool to make some valuable mathematical calculations. Due to that notion, a more detailed instruction with some crucial explanations has been made.

### 7.2. Choice of a working environment
This library is developed using Anaconda3 2019.03 for Windows 64-bit. Spyder IDE v3.3.3 (a part of the Anaconda distribution) was used as the main development environment. Since 2020, Anaconda infrastructure is not totally free of charge, but it is still free in some use cases [[link](https://www.anaconda.com/blog/is-conda-free)]. It gives user experience that is partly close to what one can acquire using MATLAB. A huge difference between them is that MATLAB is commercial software produced by one company, while Python with third-party libraries and tools are mostly pieces of free open-source software that are made by many different people and organizations.

MATLAB is an all-in-one tool. In most cases, a user does not have the necessity to use third-party libraries. The built-in libraries do not produce conflicts with each other. The "dependency hell" is unknown for MATLAB users. The documentation is provided in two places (offline as part of a MATLAB installation and on their official website) in one style.
Usually, Anaconda3 requires to use third-party libraries, which should be installed from related channels called "anaconda" or "defaults", "conda-forge", "bioconda", and others. Each stable Anaconda3 release has a variety of libraries that do not conflict with each other [[link](https://docs.anaconda.com/free/anaconda/index.html)]. However, if a user wants to use another version of a library than Anaconda3 provides or a library that is absent there, the user will eventually encounter the "dependency hell". As a cure for this poison of programmers' lives, the concept of virtual environments exists.

#### 7.2.1. The dependency hell
The premise of the "dependency hell" is that each programming library and software, which relies on it, require to use a certain range of versions of programming languages in which the library is written and a certain range of versions of each third-party library that this library requires. Then, let us imagine that there are 1000 versions of 100 libraries and each one could be written in one of ten different versions of the same programming language, and also has ten arbitrary dependencies (in average) on other libraries. In many cases, using a particular library version will not allow to use many other versions of many other libraries. This abstract example can help realize that it can be very hard to resolve potential conflicts between different libraries and provide the possibility to use all of them in one user's project. Fortunately, in many cases, it is not necessary to use 100 libraries in one place.

#### 7.2.2. Packages
The term "programming library" has too narrow sense to describe the whole variety of third-party software that can be installed and used during the process of software development. It is better to use a more wide term "package", which can imply an application like a Python interpreter itself. This term is widely used in the Python community. A package can contain not only a piece of software, but it also has an instruction for automatic installation. If someone needs a Python library, they should look for a respective package.

#### 7.2.3. Virtual environments
In spite of having all potentially useful Python third-party libraries in one environment, a developer can use an isolated environment containing only those libraries that are necessary in a certain project. There are two major alternatives of Python package management. They are based on console / terminal applications and online repositories that store packages.

A bundle of "venv" and "pip" [[link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)] is standard and most popular. "venv" provides the ability to create and manage virtual environments, while "pip" is an application for installation and management of packages (libraries, applications, etc.). This couple has two big practical drawbacks. A virtual environment created with "venv" is based on an external Python installation. It means that all virtual environments created by "venv" will use the related particular Python interpreter. It limits the ability to use different Python versions, for example, to check that a piece of software, on which a user works, will work with different Python versions. The second problem is that the modern "pip" have only a basic mechanism of dependency resolution (initially it did not truly have it at all) [[link](https://pip.pypa.io/en/stable/topics/dependency-resolution/)]. The "pip" infrastructure is not optimized for solving this task. It can consume a lot of time and Internet traffic to find which versions of packages have to be installed alongside with the packages that a user wants to have.

An alternative to these tools is "conda" [[link](https://github.com/conda/conda)]. As a virtual environment manager it is not limited by usage in only Python projects [[example](https://interrupt.memfault.com/blog/conda-developer-environments)]. A virtual environment created by "conda" can contain its own Python version or do not contain any. In addition, modern "conda" versions use an [advanced dependency solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). Online repositories here are called "channels" and can be public or private. Users can freely create their own channels. "defaults" is the main Anaconda3 channel maintained by its developers. This channel can be paid in some cases. "conda-forge" is a big free community driven repository.

There are also other alternatives like "mamba" and "micromamba" [[link](https://github.com/mamba-org/mamba)], but a full review of existing Python infrastructure solutions is out of scope of this document.

It is better to install all necessary packages simultaneously, rather than install them consequently one-by-one (because it is easier to resolve the problems with dependencies one time, rather than install one package with its dependencies and find that another package cannot be installed in this configuration) [[link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)]. The author recommends to use a virtual environment to install this library in.

This library is made using non-last versions of Python and third-party libraries. However, the last available version of Anaconda3 can be used. It would be better for user experience to create a separate virtual environment for working with this library. A user has two options here:
1. Conservative. Install those old versions of Python and third-party packages that were used during the development of this library. In such environment, the package will work not worse than it does on the author's computer. Modern versions of Python and third-party packages can be used in other virtual environments.
2. Advanced. Use modern versions of Python and third-party packages to work with this library. In theory, there can be some unexpected problems.

#### 7.2.4. Minimal set of packages
Of course, a beginner will ask a question "What libraries can be useful in my project?". If this beginner is a radio engineer / scientist, probably, it would be nice to have [NumPy](https://numpy.org/) (math, arrays), [SciPy](https://scipy.org/) (has functionality similar to MATLAB Optimization Toolbox), and a plotting library. [Matplotlib](https://matplotlib.org/) imitates the MATLAB plotting subsystem with all its vast customization and entanglement. There are also some alternative plotting libraries and applications. One of them that should be mentioned here is "[gnuplot](https://www.gnuplot.info/)" &ndash; a program to make scientific plots. "gnuplot" can be used directly from a Python interpreter via a third-party programming interface library or it can be ran separately. [Microsoft Excel](https://www.microsoft.com/en-us/microsoft-365/excel) can be used to configure plots using a graphical user interface. To use a separate program for displaying plots, the data from a Python program must be formatted and exported into a file. The author chose Matplotlib because got accustomed to that pain, but it is not a recommendation for others. In addition, the beginner may want to install some particular libraries that are necessary to solve their problems. PowAmp library relates to this category. Another useful category of programs is unit test frameworks. The ability to run all necessary tests in one click and watch into a terminal for the results, instead of manually running snippets of code from different files, makes a programmer's life easier. "[pytest](https://pytest.org/en/)" is the de-facto standard unit test framework for Python programmers.

#### 7.2.5. Intermediate recommendations
1. Install Anaconda3.
2. Create a "conda" virtual environment. Download and install a "conda" package of PowAmp library. The installation will cause installation of the required dependencies: numpy, scipy, matplotlib. In addition, install "pytest", "spyder-kernels" (necessary to work with the virtual environment from Spyder IDE), and whatever is needful. Try to install these packages simultaneously.
3. Configure Spyder IDE and create a Spyder project with this library.

### 7.3. Formats of packages
There are two types of PyPI ("[pip](https://pip.pypa.io/en/stable/)") packages:
1. Source distribution "sdist" archive with "`.tar.gz`" extension [[link](https://packaging.python.org/en/latest/specifications/source-distribution-format/)]. It contains source code and an instruction on how to build it. Since it is source code, this package is system independent (if the code can be built for a related computer architecture). A package installation using this type of distribution requires a building procedure and therefore requires related programming tools. For example, C/C++ compilers can be necessary to build some packages.
2. Binary distribution "wheel" archive with "`.whl`" extension, which is actually a ZIP archive [[link](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)]. It contains already built code. Potentially, it can depend on a computer architecture including an operating system. For example, it can contain pre-built C/C++ libraries. In case of pure Python packages, the content of a "wheel" package can be very similar to that of a related "sdist" package and include the same source code, which makes the "wheel" package system independent. It is better to use a "wheel" package if it is provided, because it does not require a building procedure.

There are also two types of "[conda](https://docs.conda.io/en/latest/)" packages:
1. A tarball archive with "`.tar.bz2`" extension. This is an old format of "conda" package distributives.
2. A ZIP container with "`.conda`" extension. It is a newer format with better compression [[link](https://docs.conda.io/projects/conda-build/en/stable/resources/package-spec.html)] and faster processing [[link](https://www.anaconda.com/blog/understanding-and-improving-condas-performance)].

These "conda" packages can have different content including: source code, documentation, data sets, and binary files. Therefore, they can be system dependent or independent. System independent packages has "noarch" (no architecture) affix in their names [[link](https://www.anaconda.com/blog/condas-new-noarch-packages)]. Pure Python packages can be "noarch" if they meet a specific set of criteria [[link](https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/packages.html)].

In practice, a "conda" package may have content similar to a related "wheel" package. For example, in case of "powamp" packages, the same "[`setup.py`](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)" script for "[setuptools](https://setuptools.pypa.io/en/latest/)" is used to build them from the source code. The main difference is that a "conda" package has its own [metadata](https://docs.conda.io/projects/conda-build/en/stable/resources/package-spec.html) that is necessary to work with "conda" infrastructure. The building of a "conda" package requires to use an additional configuration file "[`meta.yaml`](https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html)" for "[conda-build](https://docs.conda.io/projects/conda-build/en/stable/)" application.

General recommendation here is to use pre-built packages whenever it is possible to avoid unnecessary issues that someone else already solved.

### 7.4. Step-by-step instruction
At the first release moment, this library is absent in online Python package repositories. However, packages for "pip" and "conda" are built and available for manual downloading and installation from a local source.

These instructions are made detailed and comprise extra parts to make Python novices able to use this library. Those who already have experience in Python, prefer a certain virtual environment manager, a package manager, and an integrated development environment (IDE), obviously, can omit these unnecessary steps and replace them with what they wish.

#### 7.4.1. Using "conda" (recommended)
1. Download and install "conda" compatible tools like Anaconda3 or Miniconda.  
   Current version of Anaconda [[link](https://www.anaconda.com/download/success)].  
   Anaconda archive [[link](https://repo.anaconda.com/archive/)].  
   Compatibility of Anaconda with old operating systems [[link](https://docs.anaconda.com/free/anaconda/install/old-os/)].  
   Packages [[link](https://anaconda.org/)].

2. Download and install the "powamp" package in a local repository.  
   This instruction describes the procedure of a local repository creation. If you already have your own local repository, you can use it on that purpose instead.

   2.1. Download a "powamp" package for "conda" &ndash; a "`powamp*.tar.bz2`" or "`powamp*.conda`" archive.

   2.2. Create a directory on a computer for manually downloaded packages. For example,
   ```
   C:\ProgramData\Anaconda3\pkgs-usr
   ```

   2.3. Create a subdirectory named "`noarch`" (no architecture) in that directory.
   ```
   C:\ProgramData\Anaconda3\pkgs-usr\noarch
   ```
   This directory is used for packages that do not contain code that depends on a computer architecture (including an operating system).

   2.4. Place the "powamp" package into the "`noarch`" directory.

   2.5. Run "Anaconda Prompt".

   2.6. Make a local "conda" repository in the defined directory.
   ```
   conda index C:/ProgramData/Anaconda3/pkgs-usr
   ```
   After that, "conda" can use "`pkgs-usr`" directory as a local repository for searching and installation packages out of there.  
   Note: it is correct to use "`/`" in addresses here like in UNIX instead of "`\`" that is a norm in Windows.

   To make sure that the local repository is successfully created, use this command
   ```
   conda search --channel C:/ProgramData/Anaconda3/pkgs-usr --override-channels
   ```
   It looks for packages in the channel (i. e. repository) with a given address. "`--override-channels`" command tells to use only given channels and not to look in others.
   Normally, you will see that the "powamp" package is available in the "pkgs-usr" channel.
   You can add other packages into this directory to install them from the local source.

3. Creation of a virtual environment with required packages.

   3.1. Create and activate a "conda" virtual environment named "powamp-prj" (power amplifier project) with necessary packages.
   ```
   conda create -n powamp-prj python=3.7.3 pytest==4.3.1 spyder-kernels==0.4.2 powamp --channel file:///C:/ProgramData/Anaconda3/pkgs-usr --channel defaults --override-channels
   ```
   It defines versions of Python, "pytest" (optional, unit test framework) and "spyder-kernel" (necessary only for working with Spyder IDE and can be omitted if you prefer to use another IDE) to install. Other packages including "numpy", "scipy", and "matplotlib" are "powamp" dependencies; they will be installed automatically with other higher order dependencies.  
This command declares the list of channels (repositories) from which packages will be installed. "`--channel file:///C:/ProgramData/Anaconda3/pkgs-usr`" is used to get the "powamp" package only. "`--channel defaults`" is used for installation of all other packages. You can try to use "`--channels conda-forge`" instead, if for some reasons you do not want to use the default "conda" channel. "`--override-channels`" prevents to use channels other than mentioned.  
   If you want to have other packages in this working virtual environment, add them in the command.

   3.2. Activate the "powamp-prj" virtual environment to work in it.
   ```
   conda activate powamp-prj
   ```
   After that, "conda" will work inside this environment. Related Python interpreter will also be used.  
   If you run
   ```
   where python
   ```
   command ("`which python`" in case of OS X and Linux systems) in the "base" environment, you will see the path to the system Python interpreter. Like "`C:\ProgramData\Anaconda3\python.exe`". After activation of the virtual environment, the same command will give this output "`C:\ProgramData\Anaconda3\envs\powamp-prj\python.exe`".

   To deactivate this environment and return into the "base" environment when it is necessary, type
   ```
   conda deactivate
   ```
   Note: it is unnecessary to manually activate a virtual environment using "conda" each time when you are going to work in it using Spyder IDE.

4. Configure Spyder IDE [[link](https://github.com/spyder-ide/spyder/wiki/Working-with-packages-and-environments-in-Spyder)] and create a project (optional).  
   This instruction is written for Spyder v3.3.3.

   4.1. Select the Python interpreter from your working virtual environment.  
`Tools -> Preferences -> Python interpreter -> Python interpreter`. Choose "`Use the following Python interpreter`". Point to the Python interpreter of your virtual environment.  
   In IPython console sub-window, choose the gear symbol called "`Options`", select "`Restart kernel`" and confirm the restarting.  
   If you type into the IPython console
   ```
   import sys
   print(sys.executable)
   ```
   it will show you the path to the Python interpreter executable file. In this example,
   ```
   C:\ProgramData\Anaconda3\envs\powamp-prj\pythonw.exe
   ```

   4.2. Create a new project in which you would like to use this library (not actually necessary, but convenient).  
   The project name and path here are just examples.  
   `Project -> New Project...`  
   Select "`New directory`".  
   `Project name: PowAmp-rsch`  
   `Location: D:\Programming`  
   `Project type: Empty project`  
   Click the "`Create`" button.  
   After that, you will see that the working directory is "`D:\Programming\PowAmp-rsch`" (power amplifier research). And in the related directory a new subdirectory called "`.spyproject`" appeared.  
   If you go into  
   `Tools -> PYTHONPATH manager`  
   you will see that the path "`D:\Programming\PowAmp-rsch`" is selected [[link](https://www.geeksforgeeks.org/pythonpath-environment-variable-in-python/)].  
   To check that this library is available in this project, type
   ```
   import powamp
   ```
   The module will appear in the "`Variable explorer`" sub-window.

   4.3. For a quick introduction, the author recommends to copy the scripts from the "`examples`" directory (see for them in a package installation directory or among the source files of the library) into the "`PowAmp-rsch`" directory and use them as a starting point for your own project.

   If you are familiar with MATLAB / GNU Octave, you may notice that the use of Python is not so convenient. However, it can be compensated by Python flexibility, and these long instructions and reading of endless StackOverflow threads are worth it in the author's opinion.

#### 7.4.2. Using "pip"
This instruction contains only basic points. It implies that a user independently chooses a Python interpreter and other working instruments. The instruction itself is made using a "conda" virtual environment in which Python v3.7.3 is installed as a "conda" package. After that, a virtual environment is created using "venv", and other packages are installed there using "pip". There is a problem with creation of a virtual environment using "venv" in that version of Anaconda3 [[link](https://github.com/ContinuumIO/anaconda-issues/issues/10822)]. Thanks to basavarajsh98 suggestion, it is solved by copying "`python.exe`" and "`pythonw.exe`" files into the place where they were missed.

To build some "pip" packages you may need to have Microsoft Visual Studio Build Tools [[link](https://visualstudio.microsoft.com/visual-cpp-build-tools/)]. While making this instruction, it was necessary to build "spyder-kernels" v0.4.2.

1. Choose and install a Python interpreter onto your computer if you do not already have one.
2. Create and activate a virtual environment using "venv".
   ```
   python -m venv D:\Programming\venv-pip\powamp-prj\
   ```
   Activate the virtual environment.
   ```
   D:\Programming\venv-pip\powamp-prj\Scripts\activate
   ```
   Now, the virtual environment is activated. A new virtual environment already contains "pip" and "setuptools" (a back-end library to build "pip" packages from sources).  
   To be sure that "pip" is there, use
   ```
   python -m pip --version
   ```
   To deactivate the environment when it is necessary, use
   ```
   D:\Programming\venv-pip\powamp-prj\Scripts\deactivate
   ```
3. Install "spyder-kernels" if you are going to use Spyder IDE. The necessary version here can be different from the mentioned below; see the requirements of your Spyder IDE version.
   ```
   python -m pip install spyder-kernels==0.*
   ```
4. Download a "`powamp*.whl`" ("wheel") package and place it in a local directory that will be used as a local repository. For example,
   ```
   D:\Programming\pkgs-pip
   ```
5. Install the "powamp" package with its dependencies.
   ```
   python -m pip install powamp --find-links file:///D:/Programming/pkgs-pip/
   ```
   "`--find-links`" key is used to add an additional URL for finding packages, i. e. "powamp". Its dependencies will be downloaded automatically from the PyPI repository.

   After installation, you may check the list of packages in this virtual environment using
   ```
   pip list
   ```
6. Configuring of Spyder IDE here is similar to that for a "conda" virtual environment. The difference is that you should use the Python interpreter from "powamp-prj" virtual environment created by "venv". In this example, the absolute path is
   ```
   D:\Programming\venv-pip\powamp-prj\Scripts\python.exe
   
   ```

If everything is fine, the library is ready to use.

#### 7.4.3. Barbarian way
In short, just copy the "`powamp`" directory with all its content into your working directory. Install the required dependencies separately.  
It works if a library, like a pure Python one, does not actually require a building procedure to make binary files.  
This way can be used to assess the library quickly, but is not recommended for regular use.

1. Install Anaconda3.
2. Using Anaconda Navigator, install "numpy", "scipy", and "matplotlib" packages in the virtual environment in which you are going to work.
3. Create a project using Spyder IDE.
4. Copy the "`powamp`" directory into your project directory.

Now you are able to import "powamp" package as a [regular package](https://docs.python.org/3/reference/import.html) directly inside the project directory.

## 8. Library structure
This library mostly relies on the object-oriented programming (OOP) paradigm.

Examples that show how to use this library can be found using "`get_path`" function.

A power amplifier (PA) model can be created by passing its name into a special function "`make_powamp`".

Parameters of a PA model can be set and got using separate setters "`set_<parameter-name>`" and getters "`get_<parameter-name>`", or common setter "`set_params`" and getter "`get_params`".

An active device (AD) and a load network (LN) are configurable. It means that their structure and parameters depend on a chosen component name. "`config_ad`", "`get_ad_config`", "`config_load`", "`get_load_config`" methods are used on that purpose.

Each PA model has its own tuning methods "`tune_<something/somehow>`" that produce respective calculations and output data.

All steady-state voltages and currents in a PA model (excluding its LN) can be calculated during a full simulation procedure, which can be ran using "`simulate`" method. It produces a simulation data object with its own set of methods. These methods provide the ability to: get parameters of the related PA model; get voltages and currents in time and frequency domains; calculate some derivative electrical characteristics; save the PA model parameters and simulation data to a disk.

## 9. Available power amplifier models
The library contains 2 power amplifier (PA) models:
1. "`class-e:le`" &ndash; class E PA with lumped elements. The model has 2 main and 1 auxiliary tuning procedures. It can be tuned to work at one frequency or to work in a frequency range with a particular load network (LN), which has a specific input impedance law in frequency domain. (An LN model for that is not implemented.)
2. "`class-ef:le`" &ndash; class EF<sub>x</sub> PA with lumped elements. The model has 2 main and 2 auxiliary tuning procedures. It can be tuned to work at one frequency with the maximum modified power output capability (MPOC) [*"High-Efficiency Class E, EF<sub>2</sub>, and E/F<sub>3</sub> Inverters", Zbigniew Kaczmarczyk, 2006,* [link](https://ieeexplore.ieee.org/document/1705650)], which makes this PA very effective in terms of output power taken from a particular active device (transistor, vacuum tube, etc.).

There are 3 active device (AD) models:
1. "`bidirect:switch`" &ndash; a linear AD that can equally conduct current in both directions during its duty cycle.
2. "`forward:switch`" &ndash; a nonlinear AD that can conduct only forward current during its duty cycle.
3. "`freewheel:switch`" &ndash; a nonlinear AD that can conduct forward current during its duty cycle and can also conduct backward current via a freewheeling diode at any respective moment.

There are 7 passive linear LN models:
1. "`zbar`" &ndash; an almost ideal filter of the 1st harmonic of voltage or current.
2. "`sercir:le`" &ndash; a series resonant circuit with lumped elements.
3. "`parcir:le`" &ndash; a parallel resonant circuit with lumped elements.
4. "`teenet:le`" &ndash; a Tee network with lumped elements.
5. "`pinet:le`" &ndash; a Pi network with lumped elements.
6. "`zlaw`" &ndash; a LN defined by its input impedance law in frequency domain.
7. "`custom`" &ndash; a user defined LN model.

Any of these AD and LN models can be combined with any PA model.
The models take into account losses in their electrical components. Therefore, it is possible to assess the efficiency of a PA.

## 10. Working with the library
At the first release moment, there is no separate documentation. However, the code is documented and there are several scripts with basic examples of usage.

If you are going to use the library for the first time, you can copy interesting examples from the "`examples`" directory into your working directory. The examples demonstrate how to work with the application programming interface (API) and some library features. These examples can be considered as an initial point for your further work. You can choose an interesting example, which you think is the closest to your needs, study the API, remove unnecessary code snippets from the script, and add some extra code to achieve your research goals.

## Conclusion
The library contains models of class E and EF<sub>x</sub> power amplifiers (PAs) with lumped elements, 3 abstract active device models, and 7 models of passive linear load networks. The PA models have built-in tuning methods that can provide parameters to achieve some optimum working conditions with ease.

The models are based on the harmonic balance analysis, which makes it possible to quickly calculate a steady-state response of a PA. The number of physical effects that can be taken into account in these models is significantly bigger than in case of analytical models of the PAs based on systems of differential equations.

The application programming interface (API) of the library has been developed to hide the complexity of the mathematical models and provide to a user a higher level abstraction in order to facilitate focusing on PA application problems.

This library can supplement existing radio engineering / scientific tools and occupy a slot between analytical mathematical models and electrical circuit simulators. It also can be used to implement complex user defined research scenarios.
