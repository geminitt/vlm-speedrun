# vlm-speedrun — study notes

Personal learning material for [vlm-speedrun](../README.md), kept on the orphan
branch `notes` so it stays out of the project's `main` history. The notebooks are
written in Vietnamese.

| Notebook | Topic |
|---|---|
| `00_vi_sao_mot_anh_ton_nghin_token.ipynb` | why one image costs over a thousand tokens (needs the GPU) |
| `01_bo_tri_thi_nghiem_cho_cong_bang.ipynb` | designing fair experiments; the four measurement mistakes (reads `results/` only) |

## Setup

This branch is checked out as a worktree inside the main checkout, so `..` is the
project root and the notebooks import `bench/` and read `results/` from there:

```bash
git clone git@github.com:geminitt/vlm-speedrun.git && cd vlm-speedrun
git worktree add notes notes      # ./notes is ignored on main
pixi run jupyter lab notes/       # pixi finds ../pixi.toml
```
