import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.metrics import accuracy_ci, relaxed_match, relaxed_accuracy, to_number, wilson_interval


def test_exact_match():
    assert relaxed_match("42", "42")


def test_within_five_percent_is_correct():
    assert relaxed_match("102", "100")      # 2% off
    assert not relaxed_match("106", "100")  # 6% off


def test_thousands_separator_and_percent():
    assert to_number("1,234") == 1234.0
    assert relaxed_match("45%", "45")


def test_text_answers_match_after_normalisation():
    assert relaxed_match("  Yes. ", "yes")
    assert not relaxed_match("no", "yes")


def test_zero_as_the_answer():
    assert relaxed_match("0", "0")
    assert not relaxed_match("1", "0")


def test_overall_accuracy():
    assert relaxed_accuracy(["10", "20"], ["10", "99"]) == 0.5


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
