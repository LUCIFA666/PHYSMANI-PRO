import json
from pathlib import Path
import tempfile
import unittest

from our_mani.local_mpc_control_20261103.native_task_evaluation import (
    evaluate_case, evaluate_states, write_case_evaluation,
)


def state(tick, *, held=False, lift=0., success=False, x=0.):
    return dict(time=tick * .05, counted_tick=tick, held=held, native_success=success,
                native_result=dict(success=success), object_position=[x, 0., .75 + lift],
                tcp_position=[0., 0., .8], tcp_quaternion=[0., 0., 0., 1.])


def write_fixture(case, ticks, *, no_proxy=False, test_read=False, complete=True):
    case.mkdir()
    values = {
        "initial_snapshot.json": dict(state=state(0)),
        "run_inputs.json": dict(mode="p1", scene="static", TEST_read=test_read,
                                config=dict(physics_dt=.05, control_dt=.1,
                                            counterfactual_no_proxy=no_proxy, formal_evaluation=True)),
        "result.json": dict(complete=complete, source_unchanged=True),
    }
    for name, value in values.items():
        (case / name).write_text(json.dumps(value))
    (case / "native_ticks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ticks))


class NativeTaskEvaluationTests(unittest.TestCase):
    def test_native_first_success_hold_and_final_drop_are_separate(self):
        ticks = [state(k, held=k < 13, lift=.05 if k < 13 else 0., success=k == 6)
                 for k in range(1, 14)]
        report, ledger = evaluate_states(state(0), ticks)
        self.assertAlmostEqual(report["native_success_first_time"], .3)
        self.assertTrue(report["native_success_any"])
        self.assertFalse(report["native_success_final"])
        self.assertAlmostEqual(report["requested_joint_condition"]["first_time"], .55)
        self.assertAlmostEqual(report["strict_current_joint_condition"]["first_time"], .55)
        self.assertFalse(report["requested_joint_condition"]["final"])
        self.assertFalse(report["strict_current_joint_condition"]["final"])
        self.assertEqual(report["p1_hold_metric"]["drops"], 1)
        self.assertAlmostEqual(report["p1_hold_metric"]["longest_held_seconds"], .55)
        self.assertEqual(len(ledger), len(ticks) + 1)

    def test_historical_pass_cannot_certify_a_later_low_grasp(self):
        ticks = [state(k, held=k != 13, lift=.05 if k < 13 else 0., success=k == 6)
                 for k in range(1, 28)]
        report, ledger = evaluate_states(state(0), ticks)
        self.assertTrue(report["p1_hold_metric"]["passed"])
        self.assertTrue(report["requested_joint_condition"]["final"])
        self.assertFalse(report["strict_current_joint_condition"]["final"])
        self.assertTrue(report["final"]["held"])
        self.assertEqual(report["final"]["current_lifted_hold_seconds"], 0.)
        self.assertEqual(ledger[-1]["current_lift_m"], 0.)

    def test_original_relative_drift_reset_is_preserved(self):
        ticks = [state(k, held=True, lift=.05, success=True, x=0. if k < 7 else .021)
                 for k in range(1, 14)]
        report, ledger = evaluate_states(state(0), ticks)
        self.assertFalse(report["p1_hold_metric"]["passed"])
        self.assertAlmostEqual(report["final"]["current_lifted_hold_seconds"], .3)
        self.assertEqual(report["p1_hold_metric"]["drops"], 0)
        self.assertAlmostEqual(ledger[7]["current_lifted_hold_seconds"], 0.)

    def test_gaps_duplicates_and_missing_native_labels_are_rejected(self):
        for ticks in ([state(1), state(3)], [state(1), state(1)]):
            with self.assertRaises(ValueError):
                evaluate_states(state(0), ticks)
        missing = state(1)
        del missing["held"]
        with self.assertRaises(ValueError):
            evaluate_states(state(0), [missing])
        contradictory = state(1, success=True)
        contradictory["native_result"]["success"] = False
        with self.assertRaises(ValueError):
            evaluate_states(state(0), [contradictory])

    def test_counterfactual_and_public_logged_receipt_never_become_benchmark_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "case"
            ticks = [state(k, held=True, lift=.05, success=True) for k in range(1, 14)]
            write_fixture(case, ticks, no_proxy=True)
            public = [dict(public_lift_hold=dict(latest=dict(timestamp=t,
                public_hold_seconds=d, current_visual_target=True,
                current_public_attachment_confirmed=True, public_lift_hold_observed=False)))
                for t, d in ((.5, .25), (.6, 0.))]
            (case / "public_grasp_v4_decisions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in public))
            before = {p.name: p.read_bytes() for p in case.iterdir()}
            report = write_case_evaluation(case, root / "artifact")
            self.assertTrue(report["metrics"]["strict_current_joint_condition"]["final"])
            self.assertTrue(report["benchmark"]["counterfactual_no_proxy"])
            self.assertFalse(report["benchmark"]["standard_benchmark_positive_credited"])
            self.assertEqual(report["public_logged_metrics"]["longest_logged_public_hold_seconds"], .25)
            self.assertEqual(report["public_logged_metrics"]["final_logged"]["public_hold_seconds"], 0.)
            self.assertEqual(before, {p.name: p.read_bytes() for p in case.iterdir()})
            with self.assertRaises(FileExistsError):
                write_case_evaluation(case, root / "artifact")
            with self.assertRaises(ValueError):
                write_case_evaluation(case, case / "new_metrics")

    def test_active_or_declared_test_source_is_rejected(self):
        for label, options in (("active", dict(complete=False)), ("declared", dict(test_read=True))):
            with tempfile.TemporaryDirectory() as directory:
                case = Path(directory) / label
                write_fixture(case, [state(1)], **options)
                with self.assertRaises(ValueError):
                    evaluate_case(case)


if __name__ == "__main__":
    unittest.main()
