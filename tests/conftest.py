"""Shared fixtures for pathway tests."""
import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_engine import SimEngine


@pytest.fixture(scope="module")
def engine():
    """Module-scoped SimEngine instance (expensive to create).

    Uses the real connectome if flywire_v783.bin exists, otherwise synthetic.
    """
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "flywire_v783.bin")
    if not os.path.exists(data_file):
        data_file = None
    eng = SimEngine(data_file=data_file, seed=42)
    return eng


@pytest.fixture(autouse=True)
def reset_stimulus(engine):
    """Clear all stimulus before each test."""
    engine.clear_stimulus()
    yield
    engine.clear_stimulus()


def run_steps(engine, n_batches=20, batch_size=50):
    """Run simulation and return the last metrics dict."""
    result = None
    for _ in range(n_batches):
        result = engine.step(n=batch_size)
    return result
