import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import Config, build_plan


def test_moi_cau_hinh_chay_du_so_mau():
    cfgs = [Config("a"), Config("b")]
    plan = build_plan(cfgs, n_samples=20, rounds=4, seed=0)
    c = Counter(cfg.name for cfg, _, _ in plan)
    assert c["a"] == c["b"] == 20 * 4  # mỗi vòng chạy lại toàn bộ mẫu


def test_cac_cau_hinh_duoc_xen_ke_chu_khong_chay_theo_khoi():
    """Nếu chạy theo khối, 10 bước đầu sẽ toàn cấu hình 'a'."""
    cfgs = [Config("a"), Config("b")]
    plan = build_plan(cfgs, n_samples=20, rounds=4, seed=0)
    dau = {cfg.name for cfg, _, _ in plan[:10]}
    assert dau == {"a", "b"}


def test_seed_khac_nhau_cho_thu_tu_khac_nhau():
    cfgs = [Config("a"), Config("b")]
    p0 = [(c.name, i) for c, i, _ in build_plan(cfgs, 20, 4, seed=0)]
    p1 = [(c.name, i) for c, i, _ in build_plan(cfgs, 20, 4, seed=1)]
    assert p0 != p1


def test_cung_seed_thi_tai_lap_duoc():
    cfgs = [Config("a"), Config("b")]
    p0 = [(c.name, i, r) for c, i, r in build_plan(cfgs, 20, 4, seed=7)]
    p1 = [(c.name, i, r) for c, i, r in build_plan(cfgs, 20, 4, seed=7)]
    assert p0 == p1


def test_moi_vong_deu_chay_du_moi_mau():
    """Các vòng phải là bản lặp, để so sánh theo cặp được."""
    cfgs = [Config("a")]
    plan = build_plan(cfgs, n_samples=5, rounds=3, seed=0)
    for r in range(3):
        idx = sorted(i for _, i, rr in plan if rr == r)
        assert idx == [0, 1, 2, 3, 4]


def test_so_sanh_theo_cap_loai_bo_khac_biet_giua_cac_mau():
    from bench.harness import Record, paired_speedup
    recs = []
    for sid, base_ms in [(1, 100.0), (2, 1000.0)]:   # mẫu 2 nặng gấp 10
        recs.append(Record("A", sid, 0, True, "", "", 0, base_ms, 0, 0))
        recs.append(Record("B", sid, 0, True, "", "", 0, base_ms / 2, 0, 0))
    r = paired_speedup(recs, "A", "B")
    assert abs(r["median_speedup"] - 2.0) < 1e-9   # đúng 2 lần, bất kể mẫu nặng nhẹ


def test_phan_tich_cau_hinh_ghep_nhieu_don_bay():
    from bench.harness import Config
    c = Config.parse("nosplit+keep0.5:pool")
    assert c.split is False and c.keep_ratio == 0.5 and c.method == "pool"
    c2 = Config.parse("edge768")
    assert c2.max_edge == 768 and c2.split is True
    assert Config.parse("baseline").keep_ratio == 1.0


def test_cau_hinh_trung_cach_cat_o_thi_dung_chung_bo_xu_ly():
    from bench.harness import Config
    assert Config.parse("keep0.5").proc_key() == Config.parse("keep0.25").proc_key()
    assert Config.parse("nosplit").proc_key() != Config.parse("baseline").proc_key()
