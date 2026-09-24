import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import Config, Record, build_plan, integrity, timing_spikes


def test_every_config_runs_every_sample():
    cfgs = [Config("a"), Config("b")]
    plan = build_plan(cfgs, n_samples=20, rounds=4, seed=0)
    c = Counter(cfg.name for cfg, _, _ in plan)
    assert c["a"] == c["b"] == 20 * 4  # every round reruns the full sample set


def test_configs_are_interleaved_not_run_in_blocks():
    """Run in blocks, the first 10 steps would all be config 'a'."""
    cfgs = [Config("a"), Config("b")]
    plan = build_plan(cfgs, n_samples=20, rounds=4, seed=0)
    head = {cfg.name for cfg, _, _ in plan[:10]}
    assert head == {"a", "b"}


def test_different_seeds_give_different_orders():
    cfgs = [Config("a"), Config("b")]
    p0 = [(c.name, i) for c, i, _ in build_plan(cfgs, 20, 4, seed=0)]
    p1 = [(c.name, i) for c, i, _ in build_plan(cfgs, 20, 4, seed=1)]
    assert p0 != p1


def test_same_seed_is_reproducible():
    cfgs = [Config("a"), Config("b")]
    p0 = [(c.name, i, r) for c, i, r in build_plan(cfgs, 20, 4, seed=7)]
    p1 = [(c.name, i, r) for c, i, r in build_plan(cfgs, 20, 4, seed=7)]
    assert p0 == p1


def test_every_round_covers_every_sample():
    """Rounds must be replicates so that comparisons can be paired."""
    cfgs = [Config("a")]
    plan = build_plan(cfgs, n_samples=5, rounds=3, seed=0)
    for r in range(3):
        idx = sorted(i for _, i, rr in plan if rr == r)
        assert idx == [0, 1, 2, 3, 4]


def test_paired_comparison_removes_sample_to_sample_variation():
    from bench.harness import Record, paired_speedup
    recs = []
    for sid, base_ms in [(1, 100.0), (2, 1000.0)]:   # sample 2 is 10x heavier
        recs.append(Record("A", sid, 0, True, "", "", 0, base_ms, 0, 0))
        recs.append(Record("B", sid, 0, True, "", "", 0, base_ms / 2, 0, 0))
    r = paired_speedup(recs, "A", "B")
    assert abs(r["median_speedup"] - 2.0) < 1e-9   # exactly 2x, regardless of sample weight


def test_parse_combines_several_levers():
    from bench.harness import Config
    c = Config.parse("nosplit+keep0.5:pool")
    assert c.split is False and c.keep_ratio == 0.5 and c.method == "pool"
    c2 = Config.parse("edge768")
    assert c2.max_edge == 768 and c2.split is True
    assert Config.parse("baseline").keep_ratio == 1.0


def test_configs_with_same_tiling_share_a_processor():
    from bench.harness import Config
    assert Config.parse("keep0.5").proc_key() == Config.parse("keep0.25").proc_key()
    assert Config.parse("nosplit").proc_key() != Config.parse("baseline").proc_key()


# --- run integrity (rule 6) ------------------------------------------------------

def _rec(cfg, sid, ms, foreign=0.0):
    return Record(config=cfg, sample_id=sid, round_idx=0, correct=True, pred="", gold="",
                  prefill_ms=0.0, generate_ms=ms, image_tokens=0, input_tokens=0,
                  foreign_mb=foreign)


def test_replicates_that_agree_have_no_spikes():
    recs = [_rec("a", s, 100.0 + s) for s in range(10) for _ in range(3)]
    assert timing_spikes(recs) == 0.0


def test_a_slow_replicate_counts_as_a_spike():
    recs = [_rec("a", 0, 100.0), _rec("a", 0, 100.0), _rec("a", 0, 200.0), _rec("a", 0, 101.0)]
    assert timing_spikes(recs) == 0.25


def test_one_spike_in_a_short_run_is_not_a_failure():
    recs = [_rec("a", s, 100.0) for s in range(24) for _ in range(2)]
    recs[0] = _rec("a", 0, 200.0)                   # 1 of 48 timings: 2.1%, but a single event
    assert integrity(recs, 0.0)["ok"]


def test_many_spikes_fail_the_run():
    recs = [_rec("a", s, 100.0) for s in range(100) for _ in range(2)]
    recs += [_rec("a", s, 200.0) for s in range(10)]   # 10 of 210 timings
    check = integrity(recs, 0.0)
    assert not check["ok"] and "spikes" in check["problems"][0]


def test_integrity_fails_when_another_process_holds_gpu_memory():
    clean = [_rec("a", s, 100.0) for s in range(50)]
    assert integrity(clean, 0.0)["ok"]
    shared = clean + [_rec("a", 0, 100.0, foreign=900.0)]
    check = integrity(shared, 0.0)
    assert not check["ok"] and "GPU memory" in check["problems"][0]


def test_speedup_interval_excludes_one_for_a_clear_gain():
    from bench.harness import paired_speedup, speedup_verdict
    recs = []
    for s in range(60):
        recs.append(_rec("base", s, 200.0 + s))
        recs.append(_rec("fast", s, 160.0 + s * 0.8))      # 1.25x on every sample
    sp = paired_speedup(recs, "base", "fast")
    assert sp["ci95"][0] > 1.2 and speedup_verdict(sp) == "faster"


def test_speedup_of_one_is_not_distinguishable():
    from bench.harness import paired_speedup, speedup_verdict
    recs = []
    for s in range(60):
        jitter = 1 + (0.05 if s % 2 else -0.05)
        recs.append(_rec("base", s, 200.0))
        recs.append(_rec("same", s, 200.0 * jitter))
    assert speedup_verdict(paired_speedup(recs, "base", "same")) == "not distinguishable from 1.00x"
