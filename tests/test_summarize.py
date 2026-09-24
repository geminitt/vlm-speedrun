"""Test cho hàm tóm tắt thống kê — phần lõi của mọi con số trong dự án."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.latency_probe import summarize


def test_hang_so_thi_khong_co_dao_dong():
    s = summarize([100.0] * 20)
    assert s["median"] == 100.0
    assert s["cv_pct"] == 0.0
    assert s["iqr_pct"] == 0.0


def test_thu_tu_cac_phan_vi():
    s = summarize([float(x) for x in range(1, 101)])
    assert s["min"] <= s["p25"] <= s["median"] <= s["p75"] <= s["p95"] <= s["max"]


def test_trung_vi_khong_bi_keo_boi_mot_gia_tri_ngoai_lai():
    """Lý do dự án báo cáo trung vị thay vì trung bình."""
    xs = [100.0] * 19 + [10_000.0]
    s = summarize(xs)
    assert s["median"] == 100.0
    assert s["mean"] > 500.0


def test_cv_tang_khi_do_phan_tan_tang():
    hep = summarize([99.0, 100.0, 101.0] * 7)
    rong = summarize([50.0, 100.0, 150.0] * 7)
    assert rong["cv_pct"] > hep["cv_pct"]


def test_khong_vo_voi_mot_phan_tu():
    s = summarize([42.0])
    assert s["median"] == 42.0 and s["std"] == 0.0
