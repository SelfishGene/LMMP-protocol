# Large Multimodal Model Protocol (LMMP)

A protocol for encoding images into text representations for multimodal Large Language Models. 
LMMP converts images to text strings using a image2string() function and reconstructs images from text strings using a string2image() function.  
LMMP uses a hierarchical patch-based approach combining RGB encoding and patch-based encoding from a precalcuated dictionary of patches to create compact, token-efficient yet human readable string image representations.

### NOTE: this repository is a work in progress and still under development

## Protocol Details

### Encoding Format

The protocol uses a structured text format with hierarchical organization:

```
<image>
<s512x512>
<h_p64f100qRGB> <h_p32f60qRGB> <h_p16f40q1024>

<p64qRGB> <h0w0> <0xA1B2C3>
<p64qRGB> <h0w64> <0xD4E5F6>
<p64qRGB> <h0w128> <0x789ABC>
...
<p64qRGB> <h448w448> <0x123456>

<p32qRGB> <h32w64> <0x7A8B9C>
<p32qRGB> <h128w192> <0xD4E5F6>
...
<p32qRGB> <h416w448> <0x123456>

<p16q1024> <h256w256> <0x0A1B>
<p16q1024> <h256w272> <0x2C3D>
...
<p16q1024> <h432w448> <0x4E5F>

</image>
```

Format components:
1. Header tags:
   - `<image>`: Container tag
   - `<sWxH>`: Image dimensions (width x height)
   - `<h_pSfPqT>`: Hierarchy level descriptor
     - S: Patch size in pixels  
       example: `p64` for 64x64 patches
     - P: Percentage of patches to encode  
       example: `f100` for full coverage, `f80` for 80% coverage, etc.
     - T: Encoding type (RGB or dictionary size)  
       example: `qRGB` for RGB, `q1024` for 1024-patch dictionary (32x32 grid of patches)

2. Patch entries:
   - `<pSqT>`: Patch descriptor (patch size and quantization type/level)
   - `<hYwX>`: Position coordinates of patch in image (top-left corner coordinates in pixels)
   - `<0xVALUE>`: Encoded value
     - RGB: 6-digit hex color (2 char hex is used for each channel R, G, B (0-255))  
       example: `<0xFF00FF>` for R=255, G=0, B=255 and encodes the color magenta
     - Dictionary: 4-digit patch index hex (first two chars: row, last two chars: column)  
       example: `<0x071B>` for patch at row 7, column 27

### Protocol Parameters

#### Patch Sizes
- Controls the granularity of image decomposition
- Typical ranges:
  - Large patches (64-128px): Capture broad features and backgrounds
  - Medium patches (32px): Balance between detail and compression
  - Small patches (8-16px): Fine details and sharp edges
- Impact on token count is quadratic with size reduction

#### Patch Coverage (Fractions)
- Determines percentage of patches encoded at each level
- Parameters: any value between 0 and 1 in 0.01 increments
- Selection based on Mean Absolute Error (MAE) between original and reconstructed patches

#### Quantization Methods
1. RGB Averaging (`'qRGB'`):
   - Direct color representation
   - Best for:
     - Color-critical regions
     - Gradients and smooth transitions
     - Areas with unique colors

2. Dictionary-based patching (`'qN'`):
   - Matches patches against pre-computed set of patches
   - Available levels of quantization (number of patches in dictionary): 64, 256, 1024, 4096

## Protocol Core Concepts

LMMP operates on several principles:
1. Protocol is simple and human readable so that both LLMs and humans can easily understad the encoding
2. Protocol is general and can represent any image at any size and what level of granularity is required
2. Hierarchical decomposition of images into patches of varying sizes - coarse to fine
3. Selective encoding of patches based on visual importance - reduces token count but also kind of an augmentation
4. Dual encoding methods (RGB and dictionary-based)

## Quick Start

1. Clone the repository:
```bash
pip install git+https://github.com/SelfishGene/LMMP-protocol.git
```

2. Basic usage:
```python
from lmmp import LargeMultimodalModelProtocol

# Initialize with patch dictionary
LMMP = LargeMultimodalModelProtocol()

# Configure encoding parameters
patch_sizes = [64, 32, 16]
patch_fractions = [1.0, 0.8, 0.6]
patch_quantizations = ['rgb', 'rgb', 1024]

# Load image
image_PIL = Image.open('/path/to/image.jpg')

# Encode and decode image
encoded_str = LMMP.image2string(image_PIL, patch_sizes, patch_fractions, patch_quantizations)
image_rec_PIL = LMMP.string2image(encoded_str)
```
## Example encodings with various parameters
![Various encodings examples](https://raw.githubusercontent.com/SelfishGene/LMMP-protocol/main/figures/LMMP_encoding_params_FLUX1_dev_image_0000062.png)

## Requirements

- Python 3.8+
- PIL/Pillow
- NumPy
- scikit-learn
- UMAP-learn
- google.generativeai (for Gemini integration)
- python-dotenv

## Repository Structure

```
LMMP-protocol/
├── lmmp/                                  # LMMP package
│   ├── __init__.py                        # Package initialization
│   ├── multimodal_protocol.py             # Core protocol implementation
│   └── patch_grid_p128_g64_8192x8192.jpg  # Pre-computed patch dictionary
├── usage/                                 
│   ├── example_usage.py                   # Usage examples and visualization
│   ├── protocol_in_context_learning.py    # Usage examples for in-context learning of (text->image) or (image->text) 
│   ├── api_key_manager.py                 # API key management utilities
│   └── create_patch_grid_images.py        # Patch dictionary generation tool
├── figures/                               
│   └── LMMP_encoding_params_*.png         # Sample images with various encoding parameters
├── README.md                              # Documentation and usage guide
└── setup.py                               # Installation script
```

### Patch Dictionary Generation

The protocol uses a pre-computed patch dictionary organized in a grid layout:
- Generated from a dataset of several thousands images
- Ordered using UMAP for smooth transitions between patches and enable patching granularity to be selected at runtime

![Patch Grid Dictionary Bank](https://github.com/SelfishGene/LMMP-protocol/blob/main/lmmp/patch_grid_p128_g64_8192x8192.jpg)

## Contributing

Contributions welcome!

## License

MIT License

## Acknowledgements

If you use this protocol in your research, please cite it as follows:

```
@misc{david_beniaguev_2024_LMMP,
    title={Large Multimodal Model Protocol (LMMP)},
    author={David Beniaguev},
    year={2024},
    url={https://github.com/SelfishGene/LMMP-protocol},
    publisher={GitHub},
}
```
