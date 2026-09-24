"""Phân tích kết quả đã đo: so sánh theo cặp cả tốc độ lẫn độ chính xác.

Chạy lại được trên file JSON cũ, không cần chạy lại mô hình.
"""
import argparse, json

from bench.metrics import paired_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gate2_edge_sweep.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    r = json.load(open(a.results))
    recs = r["records"]
    keys = list(r["configs"])
    base = keys[0]
    sp = r.get("paired_speedup_vs_baseline", {})

    print(f"{r['model']} | {r['samples']} mẫu × {r['rounds']} vòng "
          f"| đối chiếu với: {base}\n")
    head = (f"{'cấu hình':<22}{'token':>7}{'đúng':>7}{'Δ điểm':>8}"
            f"{'chỉ gốc':>9}{'chỉ mới':>9}{'p':>9}  kết luận")
    print(head); print("-" * len(head))
    acc_base = 100 * r["configs"][base]["accuracy"]
    for k in keys:
        c = r["configs"][k]
        acc = 100 * c["accuracy"]
        tok = c["image_tokens_median"]
        if k == base:
            print(f"{k.split('(')[0]:<22}{tok:>7.0f}{acc:>6.1f}%{'—':>8}"
                  f"{'—':>9}{'—':>9}{'—':>9}  (đối chiếu)")
            continue
        pa = paired_accuracy(recs, base, k)
        p = pa["p_value"]
        speed = sp.get(k, {}).get("median_speedup", float("nan"))
        verdict = ("chất lượng GIẢM rõ rệt" if p < a.alpha and pa["only_a_correct"] > pa["only_b_correct"]
                   else "chất lượng TĂNG rõ rệt" if p < a.alpha
                   else "không phân biệt được")
        print(f"{k.split('(')[0]:<22}{tok:>7.0f}{acc:>6.1f}%{acc-acc_base:>+8.1f}"
              f"{pa['only_a_correct']:>9}{pa['only_b_correct']:>9}{p:>9.4f}"
              f"  {speed:.2f}× · {verdict}")

    print("\nGhi chú: p là kiểm định McNemar trên các cặp bất đồng. "
          f"Ngưỡng dùng ở đây là {a.alpha}.")


if __name__ == "__main__":
    main()
