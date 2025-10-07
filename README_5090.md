# RFdiffusion on RTX 5090 (Blackwell Architecture)

This guide documents the workaround to run RFdiffusion on NVIDIA RTX 50-series GPUs (Blackwell architecture), which require CUDA 12.6+ and PyTorch 2.7+.

## The Problem

RFdiffusion depends on DGL (Deep Graph Library), which has the following compatibility issues:
- **DGL 2.4** (latest available) only officially supports PyTorch ≤ 2.4
- **PyTorch 2.7+** is required for RTX 5090 Blackwell GPU support
- Official DGL builds don't support CUDA 12.6+/PyTorch 2.7

## The Solution

The workaround involves using PyTorch 2.7+ with CUDA 12.8 while installing DGL 2.4 compiled for PyTorch 2.4/CUDA 12.4. Despite the version mismatch warning, this configuration is functional.

### Credit
This solution is based on community workarounds documented in:
- **GitHub Issue**: [RosettaCommons/RFdiffusion#349 - how to install RFdiffusion in Cuda12.8 and RTX50series](https://github.com/RosettaCommons/RFdiffusion/issues/349)
- Contributed by [@DTZhou1996](https://github.com/DTZhou1996) and [@kerrding](https://github.com/kerrding)
- Tested and validated on RTX 5090 with PyTorch 2.7/2.8

## Installation Steps

### 1. Create Fresh Conda Environment

```bash
conda create -n SE3nv python=3.11 pip -y
conda activate SE3nv
```

### 2. Install PyTorch 2.7 with CUDA 12.8

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Important:** PyTorch 2.7+ is required for RTX 5090 support.

### 3. Install DGL 2.4 (without dependencies)

```bash
pip install --no-deps dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

**Critical:** Use `--no-deps` to prevent pip from downgrading PyTorch to 2.4.

### 4. Install DGL's Dependencies

```bash
pip install pandas pydantic scipy psutil pyyaml requests tqdm packaging
```

### 5. Install SE3Transformer Dependencies

```bash
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../..
```

### 6. Install RFdiffusion

```bash
pip install -e .
```

### 7. Install Additional Dependencies

```bash
pip install omegaconf hydra-core icecream biopython pyrsistent
```

### 8. Apply e3nn Fix for PyTorch 2.7

PyTorch 2.6+ changed the default `weights_only` parameter in `torch.load()` from `False` to `True`, which breaks e3nn's loading of constants.

**Fix:** Edit the e3nn file:
```bash
nano ~/miniforge3/envs/SE3nv/lib/python3.11/site-packages/e3nn/o3/_wigner.py
```

Find line 10:
```python
_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))
```

Change it to:
```python
_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'), weights_only=False)
```

**Security Note:** Only apply this fix if you trust the e3nn package source.

## Verification

### Test PyTorch and GPU Detection

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
PyTorch: 2.7.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
```

### Test DGL Import

```bash
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

Expected output:
```
DGL: 2.4.0+cu124
```

You may see a warning about version mismatch - this is expected and can be ignored.

### Test RFdiffusion

```bash
python scripts/run_inference.py \
  inference.output_prefix=test_output/test \
  inference.model_directory_path=models \
  inference.num_designs=1 \
  "contigmap.contigs=[10-40]" \
  inference.ckpt_override_path=models/Base_ckpt.pt
```

Expected: Design completes successfully and generates `test_output/test_0.pdb`

## System Configuration

This setup was tested with:
- **OS:** CachyOS Linux (Arch-based)
- **GPU:** NVIDIA GeForce RTX 5090
- **CUDA Driver:** 13.0 (580.95.05)
- **CUDA Toolkit:** 12.8
- **Python:** 3.11
- **PyTorch:** 2.7.0+cu128
- **DGL:** 2.4.0+cu124
- **Conda:** miniforge3

## Known Issues and Warnings

### 1. DGL Version Warning
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
dgl 2.4.0+cu124 requires torch<=2.4.0, but you have torch 2.7.0+cu128 which is incompatible.
```

**Status:** Warning can be ignored. DGL 2.4 works with PyTorch 2.7 despite the version constraint.

### 2. Deprecated torch.cross Warning
```
UserWarning: Using torch.cross without specifying the dim arg is deprecated.
```

**Status:** Cosmetic warning from RFdiffusion code. Does not affect functionality.

### 3. Deprecated torch.cuda.amp.autocast
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
```

**Status:** Cosmetic warning from RFdiffusion code. Does not affect functionality.

## Why This Works

While DGL 2.4 was compiled against PyTorch 2.4 libraries, the core DGL C++ extensions and Python API remain compatible with PyTorch 2.7. The version constraint in DGL's metadata is conservative and doesn't reflect actual runtime compatibility.

Key factors:
1. PyTorch maintains backwards compatibility in its C++ API across minor versions
2. DGL primarily uses stable PyTorch APIs (autograd, CUDA ops, tensor operations)
3. The CUDA version mismatch (12.4 vs 12.8) is handled by CUDA's forward compatibility

## Limitations

- **Unsupported Configuration:** This is a community workaround, not officially supported by DGL or RFdiffusion developers
- **Potential Bugs:** Some edge cases may exhibit unexpected behavior due to version mismatches
- **No Guarantees:** Future PyTorch or DGL updates may break this configuration
- **Use at Your Own Risk:** For production work, wait for official DGL support for PyTorch 2.7+

## Alternative: Wait for Official Support

Check for updates:
- **DGL Repository:** https://github.com/dmlc/dgl
- **DGL Releases:** https://github.com/dmlc/dgl/releases
- Monitor for PyTorch 2.7+ and Blackwell GPU support announcements

## Troubleshooting

### ImportError: Cannot load DGL C++ libraries
Ensure you're using the correct DGL wheel for your PyTorch version. The wheel URL must match your environment.

### CUDA out of memory errors
The RTX 5090 has 32GB VRAM, but RFdiffusion can still run out with large designs. Try:
- Reducing batch size
- Using smaller protein lengths
- Clearing CUDA cache: `torch.cuda.empty_cache()`

### Model checkpoint loading fails
Make sure you've downloaded the RFdiffusion model weights to the `models/` directory.

## Additional Notes

This same approach should work for **RFdiffusion2**, which has similar DGL dependencies.

To apply this setup to RFdiffusion2, follow the same steps but clone the RFdiffusion2 repository instead.

## Date

Successfully tested: October 7, 2025
Last updated: October 7, 2025
