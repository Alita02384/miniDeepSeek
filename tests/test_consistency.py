from __future__ import annotations

from minideepseek.consistency import (
    check_checkpoint_consistency,
    check_logprob_consistency,
    check_routing_consistency,
    run_all_checks,
)


def test_logprob_consistency(debug_model, stub_tokenizer):
    report = check_logprob_consistency(
        debug_model, stub_tokenizer, [[3, 4, 5], [6, 7, 8, 9]], max_new_tokens=4
    )
    assert report.passed
    assert report.checked_tokens == 8
    assert report.max_abs_diff < report.threshold


def test_routing_consistency(debug_model, stub_tokenizer):
    report = check_routing_consistency(
        debug_model, stub_tokenizer, [[3, 4, 5], [6, 7, 8]], max_new_tokens=4
    )
    assert report.passed
    assert report.match_rate == 1.0
    assert len(report.per_layer) == 4


def test_checkpoint_consistency(debug_model, stub_tokenizer):
    report = check_checkpoint_consistency(
        debug_model,
        stub_tokenizer,
        [[3, 4, 5]],
        flavor="debugmodel",
        seq_len=128,
        n_mtp_layers=0,
    )
    assert report.passed
    assert report.max_abs_diff == 0.0


def test_run_all_checks(debug_model, stub_tokenizer):
    report = run_all_checks(
        debug_model,
        stub_tokenizer,
        [[3, 4, 5]],
        flavor="debugmodel",
        seq_len=128,
        n_mtp_layers=0,
        max_new_tokens=3,
    )
    assert report["passed"]
    assert report["logprob"]["passed"]
    assert report["routing"]["passed"]
    assert report["checkpoint"]["passed"]
