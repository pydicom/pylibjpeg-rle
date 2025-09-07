<p align="center">
<a href="https://github.com/pydicom/pylibjpeg-rle/actions?query=workflow%3Aunit-tests"><img alt="Build status" src="https://github.com/pydicom/pylibjpeg-rle/workflows/unit-tests/badge.svg"></a>
<a href="https://codecov.io/gh/pydicom/pylibjpeg-rle"><img alt="Test coverage" src="https://codecov.io/gh/pydicom/pylibjpeg-rle/branch/main/graph/badge.svg"></a>
<a href="https://pypi.org/project/pylibjpeg-rle/"><img alt="PyPI versions" src="https://img.shields.io/pypi/v/pylibjpeg-rle"></a>
<a href="https://www.python.org/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/pylibjpeg-rle.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## pylibjpeg-rle

A fast DICOM ([PackBits](https://en.wikipedia.org/wiki/PackBits)) RLE plugin for [pylibjpeg](https://github.com/pydicom/pylibjpeg), written in Rust with a Python wrapper.

Linux, MacOS and Windows are all supported.

### Installation
#### Installing the current release
```bash
pip install pylibjpeg-rle
```
#### Installing the development version

Make sure [Python](https://www.python.org/), [Git](https://git-scm.com/) and
[Rust](https://www.rust-lang.org/) are installed. For Windows, you also need to install
[Microsoft's C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
```bash
git clone https://github.com/pydicom/pylibjpeg-rle
cd pylibjpeg-rle
python -m pip install .
```

### Supported Transfer Syntaxes

| UID                 | Description  | Decoding | Encoding |
| ---                 | ---          | ---      | ---      |
| 1.2.840.10008.1.2.5 | RLE Lossless | Yes      | Yes      |

### Usage
#### Decoding
##### With pylibjpeg

```python
from pydicom import dcmread
from pydicom.data import get_testdata_file

ds = dcmread(get_testdata_file("OBXXXX1A_rle.dcm"))
arr = ds.pixel_array
```

##### Standalone with pydicom
Alternatively you can use the included functions to decode a given dataset:
```python
from rle import pixel_array, generate_frames

# Return the entire Pixel Data as an ndarray
arr = pixel_array(ds)

# Generator function that only processes 1 frame at a time,
# may help reduce memory usage when dealing with large Pixel Data
for arr in generate_frames(ds):
    print(arr.shape)
```

#### Encoding
##### Standalone with pydicom

Convert uncompressed pixel data to RLE encoding and save:
```python
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.uid import RLELossless

from rle import pixel_data

# Get the uncompressed pixel data
ds = dcmread(get_testdata_file("OBXXXX1A.dcm"))
arr = ds.pixel_array

# RLE encode and encapsulate `arr`
ds.PixelData = pixel_data(arr, ds)
# Set the correct *Transfer Syntax UID*
ds.file_meta.TransferSyntaxUID = RLELossless
ds.save_as('as_rle.dcm')
```

### Benchmarks
#### Decoding

Time per 1000 decodes, pydicom's default RLE decoder vs. pylibjpeg-rle:

| Dataset                     | Pixels  | Bytes   | pydicom | pylibjpeg-rle |
| ---                         | ---     | ---     | ---     | ---           |
| OBXXXX1A_rle.dcm            | 480,000 | 480,000 |  5.7 s  |        1.1 s  |
| OBXXXX1A_rle_2frame.dcm     | 960,000 | 960,000 | 11.5 s  |        2.1 s  |
| SC_rgb_rle.dcm              |  10,000 |  30,000 | 0.28 s  |        0.19 s |
| SC_rgb_rle_2frame.dcm       |  20,000 |  60,000 | 0.45 s  |        0.28 s |
| MR_small_RLE.dcm            |   4,096 |   8,192 | 0.46 s  |        0.15 s |
| emri_small_RLE.dcm          |  40,960 |  81,920 | 1.8 s   |        0.67 s |
| SC_rgb_rle_16bit.dcm        |  10,000 |  60,000 | 0.48 s  |        0.25 s |
| SC_rgb_rle_16bit_2frame.dcm |  20,000 | 120,000 | 0.86 s  |        0.39 s |
| rtdose_rle_1frame.dcm       |     100 |     400 | 0.16 s  |        0.13 s |
| rtdose_rle.dcm              |   1,500 |   6,000 | 1.0 s   |        0.64 s |
| SC_rgb_rle_32bit.dcm        |  10,000 | 120,000 | 0.82 s  |        0.35 s |
| SC_rgb_rle_32bit_2frame.dcm |  20,000 | 240,000 | 1.5 s   |        0.60 s |

#### Encoding

Time per 1000 encodes, pydicom's default RLE encoder vs. pylibjpeg-rle and [python-gdcm](https://github.com/tfmoraes/python-gdcm):

| Dataset            | Pixels  | Bytes   | pydicom | pylibjpeg-rle | python-gdcm |
| ---                | ---     | ---     | ---     | ---           | ---         |
| OBXXXX1A.dcm       | 480,000 | 480,000 | 30.6 s  |       1.4 s   | 1.5 s       |
| SC_rgb.dcm         |  10,000 |  30,000 |  1.9 s  |       0.11 s  | 0.21 s      |
| MR_small.dcm       |   4,096 |   8,192 |  3.0 s  |       0.11 s  | 0.29 s      |
| SC_rgb_16bit.dcm   |  10,000 |  60,000 |  3.6 s  |       0.18 s  | 0.28 s      |
| rtdose_1frame.dcm  |     100 |     400 | 0.28 s  |       0.04 s  | 0.14 s      |
| SC_rgb_32bit.dcm   |  10,000 | 120,000 |  7.1 s  |       0.32 s  | 0.43 s      |
