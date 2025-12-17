"""Command-line validation wrapper for the nlse_ssfm package."""

from nlse_ssfm.validation import run_all_validation_checks


if __name__ == "__main__":
    rows = run_all_validation_checks()
    for row in rows:
        status = "[PASS]" if row["passed"] else "[FAIL]"
        print(f"{status} {row['test']}: expected {row['expected']}, "
              f"measured {row['measured']}")
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)
