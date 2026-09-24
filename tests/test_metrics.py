import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.metrics import relaxed_match, relaxed_accuracy, to_number, wilson_interval


def test_so_khop_tuyet_doi():
    assert relaxed_match("42", "42")


def test_sai_lech_duoi_5_phan_tram_van_dung():
    assert relaxed_match("102", "100")      # lệch 2%
    assert not relaxed_match("106", "100")  # lệch 6%


def test_dau_phay_va_phan_tram():
    assert to_number("1,234") == 1234.0
    assert relaxed_match("45%", "45")


def test_dap_an_chu_so_khop_sau_chuan_hoa():
    assert relaxed_match("  Yes. ", "yes")
    assert not relaxed_match("no", "yes")


def test_dap_an_bang_khong():
    assert relaxed_match("0", "0")
    assert not relaxed_match("1", "0")


def test_do_chinh_xac_tong_the():
    assert relaxed_accuracy(["10", "20"], ["10", "99"]) == 0.5


def test_khoang_tin_cay_bao_quanh_ti_le():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo_n, hi_n = wilson_interval(5, 10)
    assert (hi_n - lo_n) > (hi - lo)  # n nhỏ thì khoảng rộng hơn


def test_mcnemar_khong_khac_biet_thi_p_lon():
    from bench.metrics import mcnemar
    assert mcnemar(10, 10) > 0.9


def test_mcnemar_lech_han_thi_p_nho():
    from bench.metrics import mcnemar
    assert mcnemar(20, 2) < 0.001


def test_mcnemar_khong_co_cap_bat_dong_thi_p_bang_1():
    from bench.metrics import mcnemar
    assert mcnemar(0, 0) == 1.0


def test_paired_accuracy_dem_dung_cac_o():
    from bench.metrics import paired_accuracy
    recs = [
        {"config": "A", "sample_id": 1, "correct": True},
        {"config": "B", "sample_id": 1, "correct": False},   # chỉ A đúng
        {"config": "A", "sample_id": 2, "correct": False},
        {"config": "B", "sample_id": 2, "correct": True},    # chỉ B đúng
        {"config": "A", "sample_id": 3, "correct": True},
        {"config": "B", "sample_id": 3, "correct": True},    # cả hai đúng
    ]
    r = paired_accuracy(recs, "A", "B")
    assert (r["only_a_correct"], r["only_b_correct"], r["both"]) == (1, 1, 1)
    assert r["n_pairs"] == 3


def test_levenshtein_co_ban():
    from bench.metrics import levenshtein
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3


def test_anls_khop_hoan_toan_duoc_mot_diem():
    from bench.metrics import anls
    assert anls("Hà Nội", ["hà nội"]) == 1.0


def test_anls_sai_mot_ky_tu_van_duoc_diem_cao():
    from bench.metrics import anls
    assert anls("hanoi", ["hanoj"]) == 0.8   # 1 − 1/5, sai đúng một ký tự trên năm


def test_anls_khac_han_thi_bang_khong():
    from bench.metrics import anls
    assert anls("hoàn toàn khác", ["xyz"]) == 0.0


def test_anls_lay_dap_an_tot_nhat_trong_nhieu_dap_an():
    from bench.metrics import anls
    assert anls("2022", ["nam 2019", "2022"]) == 1.0
