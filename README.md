# vlm-speedrun — study notes

Personal learning material for [vlm-speedrun](../README.md), kept on the orphan
branch `notes` so it stays out of the project's `main` history. The notebooks are
written in Vietnamese. Each one explains a concept from its definition, then checks it
with code on the project's own recorded results or on live measurements.

Read them in order; each notebook only uses concepts introduced earlier.

| Notebook | Topic | GPU |
|---|---|---|
| `00_thong_ke_suy_luan` | inferential statistics: standard error, Wald vs Wilson intervals, p-values, exact McNemar, power, paired bootstrap, effect size | no |
| `01_bo_tri_thi_nghiem_cho_cong_bang` | fair experiment design, through the twelve measurement mistakes met in the project and its audit | no |
| `02_do_thoi_gian_tren_gpu` | timing GPU work: asynchronous execution, warm-up, noise, clocks, drift, profiler, Amdahl's law | yes |
| `03_ben_trong_bo_ma_hoa_thi_giac` | the vision encoder: tiling, patch embedding, ViT, pixel shuffle, FLOP counting | yes |
| `04_prefill_decode_kv_cache` | autoregressive decoding, KV cache, compute vs bandwidth limits, roofline | yes |
| `05_so_thuc_va_luong_tu_hoa` | floating-point formats, absmax and NF4 quantization, LLM.int8, why nf4 is not faster here | yes |
| `06_phuc_vu_mo_hinh` | serving: gRPC and protobuf, latency vs throughput, Little's law, M/G/1 queues, capacity under an SLO | no |
| `07_tong_hop_don_bay` | synthesis: where the time goes and which levers are worth pulling | yes |

GPU notebooks load SmolVLM-2.2B (about 5 GB of VRAM) and take one to two minutes each on
an RTX 1000 Ada 6 GB; run them one at a time.

## Setup

This branch is checked out as a worktree inside the main checkout, so `..` is the
project root and the notebooks import `bench/` and `serve/` and read `results/` from
there:

```bash
git clone git@github.com:geminitt/vlm-speedrun.git && cd vlm-speedrun
git worktree add notes notes      # ./notes is ignored on main
pixi run jupyter lab notes/       # pixi finds ../pixi.toml
```

## Editing a notebook

Each notebook has a plain-text source in `src/`, with `### MD` and `### CODE`
markers between cells; edit the source, not the `.ipynb`. `tools/nbtool.py`
converts between the two (run from this folder):

```bash
pixi run --manifest-path ../pixi.toml python tools/nbtool.py build src/05_so_thuc_va_luong_tu_hoa.txt 05_so_thuc_va_luong_tu_hoa.ipynb
pixi run --manifest-path ../pixi.toml python tools/nbtool.py dump 05_so_thuc_va_luong_tu_hoa.ipynb   # print outputs
```

`build` writes the notebook and executes it, so the stored outputs always match the
current code and results. `sync` copies only the prose into an existing notebook
and keeps its outputs (for text-only edits), and `extract` goes from notebook to text.
