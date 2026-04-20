# Async Video Generator

This directory contains a rebuilt Gradio app that keeps only two native inference backends:

- `anisora-v3.2` via `/Users/fangzhou/Workspace/Index-anisora/anisoraV3.2`
- `ltx-2.3` via `/Users/fangzhou/Workspace/LTX-2/packages/ltx-pipelines`

It preserves two core product capabilities from the old `wanvace` flow:

- asynchronous task queue execution in a dedicated worker subprocess
- browser-based preview and cleanup of generated videos

## What changed

- removed the `diffsynth-studio` dependency path entirely
- replaced the old pipeline dispatch with native model wrappers
- added optional SageAttention patch hooks for both Anisora and LTX when the local Python environment provides `sageattention`
- narrowed the UI and queue schema to the two target models only

## Run

```bash
cd /Users/fangzhou/Workspace/qufafa/video_gen
python main.py
```

The app starts on `http://127.0.0.1:7860` by default.

## Required model paths

### Anisora

Fill these in the UI or export them before launch:

```bash
export ANISORA_CKPT_DIR=/path/to/Index-anisora-V3.2
export ANISORA_LOW_SUBDIR=low_noise_model
export ANISORA_HIGH_SUBDIR=high_noise_model
```

### LTX 2.3

```bash
export LTX_CHECKPOINT_PATH=/path/to/ltx23_full_model.safetensors
export LTX_GEMMA_ROOT=/path/to/gemma_root
export LTX_DISTILLED_LORA_PATH=/path/to/ltx23_distilled_lora.safetensors
export LTX_SPATIAL_UPSAMPLER_PATH=/path/to/ltx23_spatial_upsampler.safetensors
```

## Notes on SageAttention

The code includes optional runtime patching, but activation still depends on your local environment:

- `sageattention` must be importable in the Python environment
- Anisora patching is based on the native `wan/modules/attention.py` flow
- LTX patching wraps its transformer attention callables and falls back to the stock implementation if masks or unsupported layouts appear

If `sageattention` is missing or incompatible, the app falls back to the original attention implementation automatically.
