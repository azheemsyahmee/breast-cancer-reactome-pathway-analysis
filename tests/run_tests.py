"""
Standalone unit test runner for Breast Cancer Reactome Pathway Analysis.
Executes all test modules discovered in the tests/ directory.
"""

import sys
import unittest

def main():
    print("================================================================================")
    print("RUNNING UNIT TESTS: BREAST CANCER REACTOME PATHWAY ANALYSIS")
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
