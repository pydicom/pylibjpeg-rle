
## pylibjpeg-rle

A fast DICOM ([PackBits](https://en.wikipedia.org/wiki/PackBits)) RLE plugin for [pylibjpeg](https://github.com/pydicom/pylibjpeg), written in Rust with a Python 3.6+ wrapper.

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
python -m setup.py develop
```

### Supported Transfer Syntaxes

| UID                 | Description  | Decoding | Encoding |
| ---                 | ---          | ---      | ---      |
| 1.2.840.10008.1.2.5 | RLE Lossless | Yes      | Yes      |

### Benchmarks
#### Decoding

Time per 1000 decodes, pydicom's NumPy RLE handler vs. pylibjpeg-rle

| Dataset                     | Pixels  | Bytes   | NumPy  | pylibjpeg-rle |
| ---                         | ---     | ---     | ---    | ---           |
| OBXXXX1A_rle.dcm            | 480,000 | 480,000 | 4.89 s |        0.79 s |
| OBXXXX1A_rle_2frame.dcm     | 960,000 | 960,000 | 9.89 s |        1.65 s |
| SC_rgb_rle.dcm              |  10,000 |  30,000 | 0.20 s |        0.15 s |
| SC_rgb_rle_2frame.dcm       |  20,000 |  60,000 | 0.32 s |        0.18 s |
| MR_small_RLE.dcm            |   4,096 |   8,192 | 0.35 s |        0.13 s |
| emri_small_RLE.dcm          |  40,960 |  81,920 | 1.13 s |        0.28 s |
| SC_rgb_rle_16bit.dcm        |  10,000 |  60,000 | 0.33 s |        0.17 s |
| SC_rgb_rle_16bit_2frame.dcm |  20,000 | 120,000 | 0.56 s |        0.21 s |
| rtdose_rle_1frame.dcm       |     100 |     400 | 0.12 s |        0.13 s |
| rtdose_rle.dcm              |   1,500 |   6,000 | 0.53 s |        0.26 s |
| SC_rgb_rle_32bit.dcm        |  10,000 | 120,000 | 0.56 s |        0.19 s |
| SC_rgb_rle_32bit_2frame.dcm |  20,000 | 240,000 | 1.03 s |        0.28 s |

#### Encoding

Time per 1000 encodes, pydicom's NumPy RLE handler vs. pylibjpeg-rle

| Dataset            | Pixels  | Bytes   | NumPy  | pylibjpeg-rle |
| ---                | ---     | ---     | ---    | ---           |
| OBXXXX1A.dcm       | 480,000 | 480,000 | 30.7 s |       1.36 s  |
| SC_rgb.dcm         |  10,000 |  30,000 | 1.80 s |       0.09 s  |
| MR_small.dcm       |   4,096 |   8,192 | 2.29 s |       0.04 s  |
| SC_rgb_16bit.dcm   |  10,000 |  60,000 | 3.57 s |       0.17 s  |
| rtdose_1frame.dcm  |     100 |     400 | 0.19 s |       0.003 s |
| SC_rgb_32bit.dcm   |  10,000 | 120,000 | 7.20 s |       0.33 s  |

### Usage
#### Decoding
##### With pylibjpeg

Because pydicom defaults to the NumPy RLE decoder, you must specify the use
of pylibjpeg when decompressing:
```python
from pydicom import dcmread
from pydicom.data import get_testdata_file

ds = dcmread(get_testdata_file("OBXXXX1A_rle.dcm"))
ds.decompress("pylibjpeg")
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

```python
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.encaps import encapsulate
from pydicom.uid import RLELossless

from rle import encode_array

# Get an uncompressed numpy ndarray
ds = dcmread(get_testdata_file("OBXXXX1A.dcm"))
arr = ds.pixel_array

# This is a bit silly -> switch ds in, kwargs pure optional
# `encode_array` is a generator function that yields encoded frames
ds.PixelData = encapsulate([enc for enc in encode_array(arr, **{'ds': ds})])
ds.file_meta.TransferSyntaxUID = RLELossless

```
