
import os
import sys
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

import subprocess
from distutils.command.build import build as build_orig
import distutils.sysconfig



VERSION_FILE = os.path.join('rle', '_version.py')
with open(VERSION_FILE) as fp:
    exec(fp.read())

with open('README.md', 'r') as fp:
    long_description = fp.read()

setup(
    name = 'pylibjpeg-rle',
    description = (
        "Python bindings for a fast RLE decoder, with a focus on use as a "
        "plugin for pylibjpeg"
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    version = __version__,
    author = "scaramallion",
    author_email = "scaramallion@users.noreply.github.com",
    url = "https://github.com/scaramallion/pylibjpeg-rle",
    license = "MIT",
    keywords = (
        "dicom pydicom python medicalimaging radiotherapy oncology imaging "
        "radiology nuclearmedicine rle pylibjpeg rust"
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
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
    ],
    packages = find_packages(),
    package_data = {'': ['*.txt', '*.rs', '*.pyx']},
    include_package_data = True,
    zip_safe = False,
    python_requires = ">=3.6",
    setup_requires = ['setuptools>=18.0', 'setuptools-rust'],
    install_requires = ["numpy"],
    extras_require = {
        'tests': ["pytest", "pydicom", "numpy"],
        'benchmarks': ["pydicom", "numpy", "asv"],
    },
    rust_extensions = [RustExtension('rle._rle', binding=Binding.PyO3)],
    # Plugin registrations
    entry_points={
        'pylibjpeg.pixel_data_decoders': [
            "1.2.840.10008.1.2.5 = rle:decode_pixel_data",
        ],
    },
)
