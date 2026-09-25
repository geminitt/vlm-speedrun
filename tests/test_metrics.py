import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.metrics import (accuracy_ci, holm, lenient_match, relaxed_accuracy,
                           relaxed_correctness, score_chartqa, to_number, wilson_interval)


# --- ChartQA relaxed accuracy, as defined by the benchmark ---------------------------

def test_exact_match():
    assert relaxed_correctness("42", "42")


def test_within_five_percent_is_correct():
    assert relaxed_correctness("102", "100")      # 2% off
    assert not relaxed_correctness("106", "100")  # 6% off


def test_percent_sign_divides_by_one_hundred():
    assert relaxed_correctness("12%", "0.12")
    assert not relaxed_correctness("45%", "45")


def test_the_whole_answer_must_be_a_number():
    # no number is fished out of a sentence, and thousands separators do not parse
    assert not relaxed_correctness("82.39 billion U.S. dollars", "82.39")
    assert not relaxed_correctness("1,234", "1234")
    assert not relaxed_correctness("1992, 2016", "2009")


def test_text_answers_compare_case_insensitively():
    assert relaxed_correctness("Yes", "yes")
    assert not relaxed_correctness("no", "yes")


def test_zero_as_the_answer():
    assert relaxed_correctness("0", "0")
    assert not relaxed_correctness("1", "0")


def test_years_get_the_numeric_tolerance_unless_asked_not_to():
    assert relaxed_correctness("2019", "2017")                     # the benchmark's known flaw
    assert not relaxed_correctness("2019", "2017", exact_years=True)
    assert relaxed_correctness("2017", "2017", exact_years=True)
    assert relaxed_correctness("2018.5", "2019", exact_years=False)


def test_score_chartqa_cleans_the_answer_first():
    assert score_chartqa(" Cameroon. ", "Cameroon")
    assert score_chartqa("2.5.", "2.5")


def test_lenient_variant_kept_for_the_sensitivity_check():
    assert to_number("1,234") == 1234.0
    assert lenient_match("45%", "45")
    assert lenient_match("82.39 billion U.S. dollars", "82.39")


def test_overall_accuracy():
    assert relaxed_accuracy(["10", "20"], ["10", "99"]) == 0.5


def test_holm_adjusts_in_order_and_never_decreases():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.5]))


def test_confidence_interval_brackets_the_proportion():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo_n, hi_n = wilson_interval(5, 10)
    assert (hi_n - lo_n) > (hi - lo)  # smaller n gives a wider interval


def test_mcnemar_no_difference_gives_large_p():
    from bench.metrics import mcnemar
    assert mcnemar(10, 10) > 0.9


def test_mcnemar_lopsided_gives_small_p():
    from bench.metrics import mcnemar
    assert mcnemar(20, 2) < 0.001


def test_mcnemar_without_discordant_pairs_gives_p_of_one():
    from bench.metrics import mcnemar
    assert mcnemar(0, 0) == 1.0


def test_paired_accuracy_counts_each_cell():
    from bench.metrics import paired_accuracy
    recs = [
        {"config": "A", "sample_id": 1, "correct": True},
        {"config": "B", "sample_id": 1, "correct": False},   # only A correct
        {"config": "A", "sample_id": 2, "correct": False},
        {"config": "B", "sample_id": 2, "correct": True},    # only B correct
        {"config": "A", "sample_id": 3, "correct": True},
        {"config": "B", "sample_id": 3, "correct": True},    # both correct
    ]
    r = paired_accuracy(recs, "A", "B")
    assert (r["only_a_correct"], r["only_b_correct"], r["both"]) == (1, 1, 1)
    assert r["n_pairs"] == 3


def test_levenshtein_basics():
    from bench.metrics import levenshtein
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3


def test_anls_exact_match_scores_one():
    from bench.metrics import anls
    # non-ASCII on purpose: case folding must work beyond ASCII
    assert anls("Hà Nội", ["hà nội"]) == 1.0


def test_anls_one_character_off_still_scores_high():
    from bench.metrics import anls
    assert anls("hanoi", ["hanoj"]) == 0.8   # 1 - 1/5: exactly one character in five differs


def test_anls_unrelated_answer_scores_zero():
    from bench.metrics import anls
    assert anls("something else entirely", ["xyz"]) == 0.0


def test_anls_takes_the_best_of_several_answers():
    from bench.metrics import anls
    assert anls("2022", ["year 2019", "2022"]) == 1.0


def test_accuracy_ci_counts_each_sample_once():
    # Greedy decoding gives the same verdict in every round, so three replicate
    # rounds carry no more information than one: the interval must not shrink.
    one_round = [{"sample_id": i, "correct": i < 12} for i in range(40)]
    three_rounds = one_round * 3
    (lo1, hi1), n1 = accuracy_ci(one_round)
    (lo3, hi3), n3 = accuracy_ci(three_rounds)
    assert n1 == n3 == 40
    assert (lo1, hi1) == (lo3, hi3)
    assert (lo3, hi3) == wilson_interval(12, 40)


def test_accuracy_ci_averages_rounds_that_disagree():
    # With random token selection the rounds can differ; each sample then
    # contributes its mean correctness, still as one observation.
    recs = [{"sample_id": 0, "correct": True}, {"sample_id": 0, "correct": False},
            {"sample_id": 1, "correct": True}, {"sample_id": 1, "correct": True}]
    (lo, hi), n = accuracy_ci(recs)
    assert n == 2
    assert (lo, hi) == wilson_interval(1.5, 2)


def test_anls_keeps_punctuation_like_the_official_metric():
    from bench.metrics import anls
    # official DocVQA ANLS only lowercases and collapses whitespace
    assert anls("u.s.", ["us"]) < 1.0
    assert anls("  New   York ", ["new york"]) == 1.0


def test_trailing_full_stop_is_removed_before_scoring():
    from bench.metrics import anls, clean_answer
    assert clean_answer(" Cameroon. ") == "Cameroon"
    assert clean_answer("2.5") == "2.5"
    assert anls(clean_answer("Cameroon."), ["Cameroon"]) == 1.0


def test_bootstrap_interval_brackets_the_mean():
    from bench.metrics import bootstrap_ci
    xs = [0.0] * 30 + [1.0] * 70
    lo, hi = bootstrap_ci(xs, n_boot=2000)
    assert lo < 0.7 < hi and hi - lo < 0.25


def test_rescore_changes_only_the_verdicts():
    from bench.rescore import rescore_run
    run = {"configs": {"a": {"accuracy": 1.0}},
           "records": [{"config": "a", "sample_id": 0, "pred": "82.39 billion.", "gold": "82.39",
                        "correct": True, "generate_ms": 900.0},
                       {"config": "a", "sample_id": 1, "pred": "12%", "gold": "0.12",
                        "correct": False, "generate_ms": 800.0}]}
    new = rescore_run(run, "relaxed", {0: "human", 1: "augmented"})
    assert [r["correct"] for r in new["records"]] == [False, True]
    assert [r["generate_ms"] for r in new["records"]] == [900.0, 800.0]
    assert new["configs"]["a"]["accuracy"] == 0.5 and new["metric"] == "relaxed"
    assert [r["subset"] for r in new["records"]] == ["human", "augmented"]
    assert run["records"][0]["correct"] is True                  # the input is not modified
