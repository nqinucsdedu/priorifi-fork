import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path("/home/runner/work/priorifi-fork/priorifi-fork")
MODULE_PATH = REPO_ROOT / "examples" / "smart-pixel" / "sensitivity_pointer_experiment.py"

spec = importlib.util.spec_from_file_location("sensitivity_pointer_experiment", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SensitivityPointerExperimentTests(unittest.TestCase):
    def test_mean_uses_available_history_when_shorter_than_last_k(self):
        wbi_lists = [[10], [11], [12]]
        histories = [[1.0, 5.0], [3.0], [4.0, 4.0, 4.0]]

        pointer = module.compute_sensitivity_pointer(
            wbi_lists=wbi_lists,
            wbi_list_delta_metrics=histories,
            last_k=3,
            aggregation_mode="mean",
        )

        self.assertEqual(pointer, 2)

    def test_median_uses_last_k_window(self):
        wbi_lists = [[10], [11], [12]]
        histories = [[100.0, 0.0, 0.0], [1.0, 9.0, 9.0], [5.0, 5.0, 5.0]]

        pointer = module.compute_sensitivity_pointer(
            wbi_lists=wbi_lists,
            wbi_list_delta_metrics=histories,
            last_k=2,
            aggregation_mode="median",
        )

        self.assertEqual(pointer, 1)

    def test_exhausted_lists_are_ignored(self):
        wbi_lists = [[], [11], [12]]
        histories = [[1000.0], [1.0], [0.5]]

        pointer = module.compute_sensitivity_pointer(
            wbi_lists=wbi_lists,
            wbi_list_delta_metrics=histories,
            last_k=1,
            aggregation_mode="mean",
        )

        self.assertEqual(pointer, 1)

    def test_ties_are_deterministic_lowest_index_wins(self):
        wbi_lists = [[10], [11], [12]]
        histories = [[3.0], [3.0], [1.0]]

        pointer = module.compute_sensitivity_pointer(
            wbi_lists=wbi_lists,
            wbi_list_delta_metrics=histories,
            last_k=1,
            aggregation_mode="mean",
        )

        self.assertEqual(pointer, 0)

    def test_raises_if_all_lists_exhausted(self):
        with self.assertRaises(ValueError):
            module.compute_sensitivity_pointer(
                wbi_lists=[[], []],
                wbi_list_delta_metrics=[[1.0], [2.0]],
                last_k=1,
                aggregation_mode="mean",
            )


if __name__ == "__main__":
    unittest.main()
