"""Kiểm tra tỉnh táo: chạy trên DocVQA rồi đối chiếu với con số đã công bố.

Nhóm SmolVLM công bố DocVQA (test) = 81,6. Nếu pipeline của ta cho ra con số xa
hẳn mức đó thì có lỗi hệ thống ở đâu đó — prompt, cách chấm, hay tiền xử lý —
và mọi kết quả khác trong dự án đều đáng ngờ theo.

Lưu ý: ta chạy trên split validation (test không có nhãn công khai) và dùng đúng
thước đo ANLS, nhưng số mẫu ít hơn, nên chỉ kỳ vọng khớp ở mức bậc độ lớn.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples
from bench.metrics import anls

PUBLISHED = 81.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="vi", help="vi | en")
    ap.add_argument("--out", default="results/sanity_docvqa.json")
    a = ap.parse_args()

    runner = Runner(a.model)
    samples = load_samples(a.samples, seed=0, dataset="docvqa")
    cfg = Config("baseline", max_new_tokens=a.max_new_tokens, prompt=a.prompt)

    scores, rows = [], []
    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            pred, n_img, _ = runner.infer(s, cfg)
            sc = anls(pred, s["golds"])
            scores.append(sc)
            rows.append({"sample_id": s["sample_id"], "pred": pred,
                         "golds": s["golds"], "anls": sc})
            if i % 25 == 0:
                print(f"  {i}/{len(samples)} | ANLS tạm tính "
                      f"{100*statistics.fmean(scores):.1f}")

    mean = 100 * statistics.fmean(scores)
    res = {"model": a.model, "dataset": "docvqa/validation", "n": len(scores),
           "anls_mean": mean, "published_test_anls": PUBLISHED,
           "gap": mean - PUBLISHED, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nANLS của ta : {mean:.1f}  ({len(scores)} mẫu, split validation)")
    print(f"đã công bố  : {PUBLISHED:.1f}  (split test, toàn bộ)")
    print(f"chênh lệch  : {mean - PUBLISHED:+.1f} điểm")
    print("kết luận    : " + ("pipeline khớp với con số công bố, không có lỗi hệ thống"
                              if abs(mean - PUBLISHED) < 10 else
                              "LỆCH LỚN — cần soát lại prompt, cách chấm, tiền xử lý"))
    print(f"đã lưu {a.out}")


if __name__ == "__main__":
    main()
