"""So sánh hai lần chạy khác nhau trên CÙNG tập mẫu.

Dùng khi hai cấu hình không thể xen kẽ trong một lần chạy — ví dụ bf16 và nf4,
vì mỗi lúc chỉ nạp được một mô hình vào card 6 GB. Khi đó ta mất lợi thế xen kẽ
theo thời gian, nhưng vẫn giữ được so sánh theo cặp trên từng mẫu.
"""
import argparse, json, statistics

from bench.metrics import mcnemar


def records(path, key):
    r = json.load(open(path))
    recs = [x for x in r["records"] if x["config"] == key]
    return r, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="file kết quả A")
    ap.add_argument("--b", required=True, help="file kết quả B")
    ap.add_argument("--key", default="baseline", help="tên cấu hình cần so")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    x = ap.parse_args()

    ra, reca = records(x.a, x.key)
    rb, recb = records(x.b, x.key)

    def fold(recs):
        acc, lat = {}, {}
        for r in recs:
            acc.setdefault(r["sample_id"], []).append(r["correct"])
            lat.setdefault(r["sample_id"], []).append(r["generate_ms"])
        return ({k: sum(v) * 2 >= len(v) for k, v in acc.items()},
                {k: statistics.median(v) for k, v in lat.items()})

    acc_a, lat_a = fold(reca)
    acc_b, lat_b = fold(recb)
    common = sorted(set(acc_a) & set(acc_b))
    only_a = sum(acc_a[i] and not acc_b[i] for i in common)
    only_b = sum(acc_b[i] and not acc_a[i] for i in common)
    p = mcnemar(only_a, only_b)
    ratios = sorted(lat_a[i] / lat_b[i] for i in common)
    n = len(ratios)

    pa = 100 * sum(acc_a[i] for i in common) / len(common)
    pb = 100 * sum(acc_b[i] for i in common) / len(common)
    print(f"cấu hình: {x.key} | {len(common)} mẫu chung\n")
    print(f"{x.label_a:>12}: {pa:5.1f}% đúng | VRAM {ra.get('peak_vram_mb', 0):.0f} MB")
    print(f"{x.label_b:>12}: {pb:5.1f}% đúng | VRAM {rb.get('peak_vram_mb', 0):.0f} MB")
    print(f"\nchênh lệch    : {pb - pa:+.1f} điểm "
          f"(chỉ {x.label_a} đúng: {only_a} · chỉ {x.label_b} đúng: {only_b} · p = {p:.4f})")
    print(f"tốc độ        : {x.label_b} nhanh hơn {statistics.median(ratios):.2f}× "
          f"[{ratios[n//4]:.2f}–{ratios[(3*n)//4]:.2f}]")
    print(f"kết luận      : " + ("chất lượng khác biệt rõ rệt" if p < 0.05
                                 else "không phân biệt được về chất lượng"))


if __name__ == "__main__":
    main()
