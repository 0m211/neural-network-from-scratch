import subprocess
import sys


TEST_MODULES = [
    "tests.test_gradient",
    "tests.test_operations",
    "tests.test_tensor_gradient",
    "examples.test_tensor_operations",
    "examples.test_operation_gradients",
    "examples.test_losses",
    "examples.test_bce",
    "examples.test_cross_entropy",
    "examples.test_optimizers",
    "examples.test_dropout",
    "examples.test_batch_norm",
    "examples.test_batch_norm_integration",
    "examples.test_batch_norm_save_load",
    "examples.test_save_load",
    "examples.gradient_check",
    "examples.gradient_check_batch_norm",
    "examples.gradient_check_batch_norm_training",
    "tests.test_tensor_edge_cases",
    "tests.test_advanced_gradient_check",
    "tests.test_numerical_stability",
    "tests.test_binary_classification",
    "tests.test_metrics",
    "tests.test_data_split",
]


def main():
    passed = []
    failed = []

    print("=" * 65)
    print("NEURAL NETWORK FROM SCRATCH - REGRESSION SUITE")
    print("=" * 65)

    for module in TEST_MODULES:
        print(f"\nRunning: {module}")
        print("-" * 65)

        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
            errors="replace"
        )

        if result.stdout:
            print(result.stdout, end="")

        if result.stderr:
            print(result.stderr, end="")

        if result.returncode == 0:
            passed.append(module)
            print(f"\nRESULT: PASS - {module}")
        else:
            failed.append(module)
            print(f"\nRESULT: FAIL - {module}")

    print("\n" + "=" * 65)
    print("REGRESSION SUITE SUMMARY")
    print("=" * 65)

    print(f"Total tests: {len(TEST_MODULES)}")
    print(f"Passed:      {len(passed)}")
    print(f"Failed:      {len(failed)}")

    if failed:
        print("\nFailed modules:")

        for module in failed:
            print(f"  - {module}")

        print("\nRegression suite FAILED.")
        return 1

    print("\nALL REGRESSION TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())