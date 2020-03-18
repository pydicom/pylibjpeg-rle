
import os
import sys
from pathlib import Path
import platform
import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
import subprocess
from distutils.command.build import build as build_orig
import distutils.sysconfig


RLE_SRC = os.path.join('rle', 'src', 'rle')


# Workaround for needing cython and numpy
# Solution from: https://stackoverflow.com/a/54128391/12606901
class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False

        import numpy
        for ext in self.distribution.ext_modules:
            if ext in extensions:
                ext.include_dirs.append(numpy.get_include())


def get_mscv_args():
    """Return a list of compiler args for MSVC++'s compiler."""
    flags = [
        '/GS',  # Buffer security check
        '/W3',  # Warning level
        '/wd"4335"',  # Ignore warning 4335
        '/Zc:wchar_t',  # Use windows char type
        '/Zc:inline',  # Remove unreferenced function or data (...)
        '/Zc:forScope',
        '/Od',  # Disable optimisation
        '/Oy-',  # (x86 only) don't omit frame pointer
        '/openmp-',  # Disable #pragma omp directive
        '/FC',  # Display full path of source code files
        '/fp:precise',  # Floating-point behaviour
        '/Gd',  # (x86 only) use __cdecl calling convention
        '/GF-',  # Disable string pooling
        '/GR',  # Enable run-time type info
        '/RTC1',  # Enable run-time error checking
        '/MT',  # Create multithreading executable
        # /D defines constants and macros
        '/D_UNICODE',
        '/DUNICODE',
    ]
    # Set the architecture based on system architecture and Python
    is_x64 = platform.architecture()[0] == '64bit'
    if is_x64 and sys.maxsize > 2**32:
        flags.append('/DWIN64=1')
    else:
        # Architecture is 32-bit, or Python is 32-bit
        flags.append('/DWIN32=1')

    return flags


def get_source_files():
    """Return a list of paths to the source files to be compiled."""
    source_files = [
        'rle/_libjpeg.pyx',
        os.path.join(RLE_SRC, 'decode.cpp'),
    ]
    #for fname in Path(RLE_SRC).glob('*/*'):
    #    if '.cpp' in str(fname):
    #        source_files.append(str(fname))

    return source_files


# Compiler and linker arguments
extra_compile_args = []
extra_link_args = []
if platform.system() == 'Windows':
    os.environ['LIB'] = os.path.abspath(
        os.path.join(sys.executable, '../', 'libs')
    )
    extra_compile_args = get_mscv_args()


extensions = [
    Extension(
        '_rle',
        get_source_files(),
        language='c++',
        include_dirs=[
            RLE_SRC,
            distutils.sysconfig.get_python_inc(),
            # Numpy includes get added by the `build` subclass
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

VERSION_FILE = os.path.join('rle', '_version.py')
with open(VERSION_FILE) as fp:
    exec(fp.read())

with open('README.md', 'r') as fp:
    long_description = fp.read()

setup(
    name = 'pylibjpeg-rle',
    description = (
        "A Python wrapper for libjpeg, with a focus on use as a plugin for "
        "for pylibjpeg"
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    version = __version__,
    author = "scaramallion",
    author_email = "scaramallion@users.noreply.github.com",
    url = "https://github.com/pydicom/pylibjpeg-rle",
    license = "MIT",
    keywords = (
        "dicom pydicom python medicalimaging radiotherapy oncology imaging "
        "radiology nuclearmedicine rle pylibjpeg"
    ),
    classifiers = [
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        #"Development Status :: 4 - Beta",
        #"Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
    ],
    packages = find_packages(),
    package_data = {'': ['*.txt', '*.cpp', '*.h', '*.hpp', '*.pyx']},
    include_package_data = True,
    zip_safe = False,
    python_requires = ">=3.6",
    setup_requires = ['setuptools>=18.0', 'cython', 'numpy>=1.16.0'],
    install_requires = ["numpy>=1.16.0"],
    cmdclass = {'build': build},
    ext_modules = extensions,
    # Plugin registrations
    entry_points={
        'pylibjpeg.pixel_data_decoders': [
            "1.2.840.10008.1.2.5 = rle:decode_pixel_data",
        ],
    },
)
