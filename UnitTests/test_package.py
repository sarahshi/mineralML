import importlib
import inspect
import unittest

import mineralML as mm


# Submodules whose public names are star-imported into the top-level mineralML namespace.
EXPORTED_MODULES = ["core", "hybrid", "mapping", "microanalysis", "stoichiometry",
                    "synthetic_minerals", "confusion_matrix"]


class TestPublicAPI(unittest.TestCase):
    def test_public_functions_and_classes_are_in_all(self):
        # A new public function or class missing from __all__ would silently drop off mm.*
        # and out of the API docs.
        for name in EXPORTED_MODULES:
            module = importlib.import_module(f"mineralML.{name}")
            defined = {n for n, obj in vars(module).items()
                       if not n.startswith("_")
                       and (inspect.isfunction(obj) or inspect.isclass(obj))
                       and obj.__module__ == module.__name__}
            with self.subTest(module=name):
                self.assertEqual(sorted(defined - set(module.__all__)), [])

    def test_all_names_exist_on_package(self):
        for name in EXPORTED_MODULES + ["plotting"]:
            module = importlib.import_module(f"mineralML.{name}")
            target = mm.plotting if name == "plotting" else mm
            with self.subTest(module=name):
                self.assertEqual([n for n in module.__all__ if not hasattr(target, n)], [])

    def test_third_party_modules_do_not_leak(self):
        for leaked in ["np", "pd", "plt", "torch", "nn", "F", "sns"]:
            with self.subTest(name=leaked):
                self.assertFalse(hasattr(mm, leaked))

    def test_confusion_matrix_is_the_submodule(self):
        self.assertTrue(inspect.ismodule(mm.confusion_matrix))


if __name__ == "__main__":
    unittest.main()
