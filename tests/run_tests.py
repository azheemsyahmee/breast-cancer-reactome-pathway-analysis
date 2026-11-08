"""
Standalone unit test runner for Breast Cancer Reactome Pathway Analysis.
Executes all test modules discovered in the tests/ directory.
"""

import os
import sys
import unittest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    print("================================================================================")
    print("RUNNING UNIT TESTS: BREAST CANCER REACTOME PATHWAY ANALYSIS")
    print(f"Working Directory: {REPO_ROOT}")
    print("================================================================================")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("================================================================================")
    if result.wasSuccessful():
        print(f"ALL TESTS PASSED ({result.testsRun} tests executed successfully)")
        print("================================================================================")
        return 0
    else:
        print(f"TESTS FAILED (Failures: {len(result.failures)}, Errors: {len(result.errors)})")
        print("================================================================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
