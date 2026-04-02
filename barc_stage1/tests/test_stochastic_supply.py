from __future__ import annotations

import unittest

import numpy as np

from src.simulator import simulate_trace
from src.stochastic_supply import run_stochastic_supply_trials, simulate_trace_stochastic_binomial


class StochasticSupplyTests(unittest.TestCase):
    def test_binomial_stochastic_reduces_to_deterministic_when_p_is_one(self) -> None:
        demand = [1, 0, 2, 1, 0, 1]
        deterministic = simulate_trace(demand, C=2, B=1)
        stochastic = simulate_trace_stochastic_binomial(
            demand,
            C=2,
            B=1,
            p_acc=1.0,
            rng=np.random.default_rng(123),
        )
        self.assertEqual(stochastic.T_exe, deterministic.T_exe)
        self.assertEqual(stochastic.stall_cycles, deterministic.stall_cycles)
        self.assertEqual(stochastic.served_history, deterministic.served_history)

    def test_run_trials_returns_requested_number_of_rows(self) -> None:
        dataframe = run_stochastic_supply_trials(
            demand=[1, 1, 1],
            C=1,
            B=0,
            p_acc=0.9,
            trials=5,
            seed=7,
        )
        self.assertEqual(len(dataframe), 5)
        self.assertIn("T_exe", dataframe.columns)


if __name__ == "__main__":
    unittest.main()
