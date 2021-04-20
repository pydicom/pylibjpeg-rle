
## pylibjpeg-rle

A fast DICOM RLE decoding plugin for pylibjpeg, written in Rust with a Python 3.6+ wrapper.

Linux, OSX and Windows are all supported.

### Installation
#### Installing the current release
```bash
pip install pylibjpeg-rle
```
#### Installing the development version

Make sure [Python](https://www.python.org/) and [Git](https://git-scm.com/) and
[Rust](https://www.rust-lang.org/) are installed. For Windows, you also need to install
[Microsoft's C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
```bash
git clone https://github.com/pydicom/pylibjpeg-rle
python -m pip install setuptools-rust
python -m pip install pylibjpeg-rle
```

### Supported Transfer Syntaxes
#### Decoding
| UID                 | Description  |
| ---                 | ---          |
| 1.2.840.10008.1.2.5 | RLE Lossless |

### Benchmark
#### Against the default pydicom RLE decoder (based on NumPy)

| Dataset               | Default  | pylibjpeg-rle |
| ---                   | ---      | ---           |
| OBXXXX1A_rle.dcm      | | |
| SC_rgb_rle.dcm        | | |
| MR_small_RLE.dcm      | | |
| SC_rgb_rle_16bit.dcm  | | |
| emri_small_RLE.dcm    | | |
| SC_rgb_rle_32bit.dcm  | | |
| rtdose_rle.dcm        | | |

### Usage
#### With pylibjpeg and pydicom

Because pydicom defaults to the NumPy RLE decoder, you must specify which
RLE handler you want to use when decompressing the *Pixel Data*:
```python
from pydicom import dcmread
from pydicom.data import get_testdata_file

ds = dcmread(get_testdata_file("OBXXXX1A_rle.dcm"))
ds.decompress("pylibjpeg")
arr = ds.pixel_array
```
