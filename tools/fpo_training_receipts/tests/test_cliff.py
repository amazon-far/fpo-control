"""Unit tests for cliff detection (no GPU required)."""

from fpo_training_receipts.cliff import detect_reward_cliff


def test_healthy_monotonic_curve():
    rewards = [float(i) for i in range(100)]
    report = detect_reward_cliff(rewards, min_plateau_iters=10, window=10)
    assert report.status == "healthy"


def test_cliff_after_plateau():
    # Plateau at 35, then collapse to ~10 around iter 4000
    rewards = [30.0] * 2000 + [35.0] * 2000 + [12.0] * 1000
    report = detect_reward_cliff(
        rewards,
        min_peak_reward=10.0,
        min_plateau_iters=100,
        drop_threshold=0.30,
        window=50,
    )
    assert report.status == "cliff_detected"
    assert report.peak_reward == 35.0
    assert report.cliff_iteration is not None
    assert report.cliff_iteration >= 3900


def test_insufficient_data():
    report = detect_reward_cliff([1.0, 2.0, 3.0])
    assert report.status == "insufficient_data"
