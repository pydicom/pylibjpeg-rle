
from pathlib import Path
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


PACKAGE_DIR = Path(__file__).parent / "rle"


with open(PACKAGE_DIR  / '_version.py') as f:
    exec(f.read())

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name = 'pylibjpeg-rle',
    description = (
        "Python bindings for a fast RLE decoder/encoder, with a focus on "
        "use as a plugin for pylibjpeg"
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    version = __version__,
    author = "scaramallion",
    author_email = "scaramallion@users.noreply.github.com",
    url = "https://github.com/pydicom/pylibjpeg-rle",
    license = "MIT",
    keywords = "dicom pydicom python rle pylibjpeg rust",
    classifiers = [
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
    python_requires = ">=3.7",
    install_requires = ["numpy>=1.20"],
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
        'pylibjpeg.pixel_data_encoders': [
            "1.2.840.10008.1.2.5 = rle:encode_pixel_data",
        ],
    },
)
