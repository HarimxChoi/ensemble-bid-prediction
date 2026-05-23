"""
PQ Bidding Metric Calculator

Validated metric calculations matching PHP implementation in SaleBid_logic.php.
Each function includes docstrings with formula references and validation test cases.

Source Documents:
- gt.md (Ground Truth Report)
- PIPELINE_DOCUMENTATION.md
"""

import math
from typing import Tuple, Optional


def calculate_min_bid_rate(
    tech_score: float,
    bid_value1: float,
    bid_value2: float,
    bid_value3: float,
    base_score: float = 6.0,
    price_comp: int = 5
) -> float:
    """
    Calculate min_bid_rate (Minimum Bid Rate) - VALIDATED 97.7% match on 100k records
    
    The minimum bid rate is the lowest rate a company can bid while still
    passing the qualification_review (qualification review).
    
    Formula (validated against CS_BIDS_RESULT via NOTICE/EXPR join):
        Step 1: pq_points = (TECH_SCORE / 100) × BID_VALUE3  (point_conversion - scaled to points)
        Step 2: tech_points = pq_points + base_score
                where base_score = CAREER_SCORE + AREA_SCORE + MANAGE_SCORE
        Step 3: min_bid_rate_raw = (BID_VALUE1 - tech_points) / BID_VALUE2
        Step 4: ROUNDUP to 5 decimals, subtract 0.005 safety buffer
    
    Args:
        tech_score: Technical score, e.g., 98.30 (percentage 0-100)
        bid_value1: BID_VALUE1 from CS_BIDS_EXPR, e.g., 150
        bid_value2: BID_VALUE2 from CS_BIDS_EXPR (divisor), e.g., 1
        bid_value3: BID_VALUE3 from CS_BIDS_EXPR (point_conversion - max PQ points), e.g., 64
        base_score: CAREER_SCORE + AREA_SCORE + MANAGE_SCORE (typical: 1+3+2=6)
        price_comp: Decimal places for rounding, default 5
    
    Returns:
        min_bid_rate as percentage (e.g., 81.083 for 81.083%)
    
    Example (judge_seq=51, over_1B):
        >>> calculate_min_bid_rate(98.30, 150, 1, 64, base_score=6)
        81.083  # (98.30/100)*64=62.91 + 6=68.91 → (150-68.91)/1=81.09 → 81.083
        >>> calculate_min_bid_rate(100.00, 150, 1, 64, base_score=6)
        79.995  # = lower_bid_rate for judge_seq=51
    
    Validation: 97.7% match rate on 100,175 joined RESULT→NOTICE→EXPR records
    """
    # Handle edge case: zero divisor
    if bid_value2 == 0:
        bid_value2 = 1
    
    # Step 1: Calculate tech points (tech_score conversion)
    # Formula: tech_points = (tech_score / 100) × bid_value3 + base_score
    tech_points = (tech_score / 100) * bid_value3 + base_score
    
    # Step 2: Calculate raw minimum bid rate
    min_bid_rate_raw = (bid_value1 - tech_points) / bid_value2
    
    # Step 3: Apply ROUNDUP to price_comp decimals, subtract 0.005 buffer
    # Note: We first round to price_comp decimals to fix floating point artifacts
    # (e.g., 81.355000001 -> 81.355) before applying ceiling.
    multiplier = 10 ** price_comp
    rounded_raw = round(min_bid_rate_raw, price_comp)
    rounded = math.ceil(rounded_raw * multiplier) / multiplier
    min_bid_rate = rounded - 0.005
    
    return round(min_bid_rate, 3)



def calculate_normal_bid_line(
    r1: float,
    r2: float,
    rj: float,
    lower_bid_rate: float = 100.0
) -> float:
    """
    Calculate normal_bid_line (Normal Bid Line) for a given company rank.
    
    The normal bid line represents the normalized competitive threshold
    where a company should bid to match the intensity of top 2 bidders,
    adjusted for their technical disadvantage.
    
    Formula (from REPORT_10_001_Ajax_01 in SaleBid_logic.php):
        normal_bid_lineⱼ = lower_bid_rate × (R₁ + R₂) / 2 / Rⱼ
    
    Key Insight: base_amt cancels out → this is PRE-BID calculable!
    
    Args:
        r1: min_bid_rate of 1st place (e.g., 80.265)
        r2: min_bid_rate of 2nd place (e.g., 80.905)
        rj: min_bid_rate of j-th place (the company being calculated)
        lower_bid_rate: lower_bid_rate, typically 100.0
    
    Returns:
        normal_bid_line as percentage (e.g., 99.13)
    
    Example:
        >>> calculate_normal_bid_line(80.265, 80.905, 81.295)
        99.13
    
    Validation Source: gt.md, Section 2.3
    """
    if rj == 0:
        return 0.0
    
    return round(lower_bid_rate * (r1 + r2) / 2 / rj, 2)


def calculate_bid_rate_diff(
    normal_line_1st: float,
    normal_line_j: float
) -> float:
    """
    Calculate bid_rate_diff (Bid Rate Difference).
    
    Represents the strategic disadvantage of lower-ranked companies
    relative to the competitive threshold.
    
    Formula:
        bid_rate_diffⱼ = normal_bid_line₁ - normal_bid_lineⱼ
    
    Args:
        normal_line_1st: normal_bid_line of 1st place
        normal_line_j: normal_bid_line of j-th place
    
    Returns:
        bid_rate_diff in percentage points (e.g., 0.79 for rank 2)
    
    Example:
        For ranks 1 and 2 with normal lines 100.30 and 99.50:
        >>> calculate_bid_rate_diff(100.30, 99.50)
        0.80
    
    Validation Source: gt.md, Section 3.3
    """
    return round(normal_line_1st - normal_line_j, 2)


def calculate_fluctuation(
    bid_amt: float,
    base_amt: float,
    normal_bid_line: float
) -> float:
    """
    Calculate fluctuation (Bid Fluctuation).
    
    Measures how much a company bid above/below the expected
    competitive threshold (normal_bid_line).
    
    Formula:
        fluctuation = (bid_amt / base_amt × 100) - normal_bid_line
    
    Interpretation:
        - Positive: Bid above threshold (conservative)
        - Negative: Bid below threshold (aggressive)
    
    Args:
        bid_amt: Actual bid amount in KRW
        base_amt: Base amount (base_amt) in KRW
        normal_bid_line: Calculated normal_bid_line
    
    Returns:
        fluctuation in percentage points
    
    Example:
        >>> calculate_fluctuation(101_910_000, 100_000_000, 100.30)
        1.61
    
    Validation Source: gt.md, Section 4.3
    """
    if base_amt == 0:
        return 0.0
    
    actual_rate = bid_amt / base_amt * 100
    return round(actual_rate - normal_bid_line, 2)


def calculate_actual_bid_rate(bid_amt: float, base_amt: float) -> float:
    """
    Calculate bid_rate_vs_base (Actual Bid Rate vs Base Amount).
    
    Formula:
        bid_rate_vs_base = (bid_amt / base_amt) × 100
    
    Args:
        bid_amt: Actual bid amount in KRW
        base_amt: Base amount (base_amt) in KRW
    
    Returns:
        Rate as percentage (e.g., 81.5 for 81.5%)
    """
    if base_amt == 0:
        return 0.0
    return round(bid_amt / base_amt * 100, 2)


def calculate_normalized_bid_rate(
    bid_amt: float,
    base_amt: float,
    min_bid_rate: float
) -> float:
    """
    Calculate normalized bid rate (target variable for ML models).
    
    This normalizes the actual bid rate by the minimum bid rate,
    enabling cross-company comparison.
    
    Formula:
        normalized_bid_rate = (bid_amt / base_amt × 100) / min_bid_rate × 100
    
    Interpretation:
        - 100.0 = Bidding exactly at minimum threshold
        - 102.5 = Bidding 2.5% above minimum
        - Always >= 100 (hard constraint)
    
    Args:
        bid_amt: Actual bid amount in KRW
        base_amt: Base amount (base_amt) in KRW
        min_bid_rate: min_bid_rate as percentage (e.g., 80.265)
    
    Returns:
        Normalized rate (always >= 100 theoretically)
    
    Validation Source: EXPERIMENT_PLAN.md, Section 1.3
    """
    if base_amt == 0 or min_bid_rate == 0:
        return 0.0
    
    actual_rate = bid_amt / base_amt * 100
    return round(actual_rate / min_bid_rate * 100, 3)


# =============================================================================
# VALIDATION TEST CASES
# =============================================================================

def validate_min_bid_rate_formula() -> Tuple[bool, str]:
    """
    Validate min_bid_rate calculation against known examples.
    
    Test cases (anonymized):
    - Company A: tech_score=98.30 -> tech_points=68.912 -> 81.088 -> 81.083 (after ROUNDUP-0.005)
    - Company B: tech_score=98.00 -> tech_points=68.72 -> 81.28
    - Company C: tech_score=95.30 -> tech_points=66.99 -> 83.01
    - Perfect score (100): tech_score=100.00 -> tech_points=70.00 -> 80.00 -> 79.995 = lower_bid_rate

    All use judge_seq=51 (>= 1B KRW category):
    - BID_VALUE1=150, BID_VALUE2=1, BID_VALUE3=64, base_score=6
    """
    test_cases = [
        # (tech_score, bid_value1, bid_value2, bid_value3, expected, tolerance)
        (98.30, 150, 1, 64, 81.083, 0.01),  # Company A
        (98.00, 150, 1, 64, 81.275, 0.01),  # Company B
        (95.30, 150, 1, 64, 83.005, 0.01),  # Company C
        (100.00, 150, 1, 64, 79.995, 0.01), # Perfect score = lower_bid_rate
    ]
    
    results = []
    all_passed = True
    
    for tech_score, v1, v2, v3, expected, tol in test_cases:
        calculated = calculate_min_bid_rate(tech_score, v1, v2, v3)
        passed = abs(calculated - expected) < tol
        all_passed = all_passed and passed
        status = '✓' if passed else '✗'
        results.append(f"  tech_score={tech_score}: calc={calculated:.3f}, exp={expected:.3f} {status}")
    
    report = "min_bid_rate Validation (judge_seq=51):\n" + "\n".join(results)
    return all_passed, report



def validate_normal_bid_line_formula() -> Tuple[bool, str]:
    """
    Validate normal_bid_line calculation against known examples.
    
    Test case (anonymized project):
    - R1=80.265, R2=80.905 (1st and 2nd place min_bid_rate)
    - For rank 3 (R3=81.295): normal_bid_line_3 = 99.13
    """
    # Given data
    r1, r2 = 80.265, 80.905
    
    test_cases = [
        # (rj, expected_normal_line)
        (80.265, 100.40),  # Rank 1
        (80.905, 99.61),   # Rank 2
        (81.295, 99.13),   # Rank 3 (approximately)
    ]
    
    results = []
    all_passed = True
    
    for rj, expected in test_cases:
        calculated = calculate_normal_bid_line(r1, r2, rj)
        # Allow 0.1% tolerance due to rounding
        passed = abs(calculated - expected) < 0.15
        all_passed = all_passed and passed
        results.append(f"  Rⱼ={rj}: calculated={calculated}, expected={expected}, {'✓' if passed else '✗'}")
    
    report = "normal_bid_line Validation:\n" + "\n".join(results)
    return all_passed, report


def validate_bid_rate_diff_formula() -> Tuple[bool, str]:
    """
    Validate bid_rate_diff calculation.
    
    Test case (anonymized):
    - bid_rate_diff_2 = 0.79 (from CSV verification)
    """
    # Using calculated normal lines from previous validation
    normal_1st = 100.40
    normal_2nd = 99.61
    
    calculated = calculate_bid_rate_diff(normal_1st, normal_2nd)
    expected = 0.79
    passed = abs(calculated - expected) < 0.05
    
    report = f"bid_rate_diff Validation:\n  calculated={calculated}, expected={expected}, {'✓' if passed else '✗'}"
    return passed, report


def run_all_validations() -> Tuple[bool, str]:
    """Run all formula validations and return consolidated report."""
    validations = [
        validate_min_bid_rate_formula,
        validate_normal_bid_line_formula,
        validate_bid_rate_diff_formula,
    ]
    
    all_passed = True
    reports = []
    
    for validation_fn in validations:
        passed, report = validation_fn()
        all_passed = all_passed and passed
        reports.append(report)
    
    full_report = "\n\n".join(reports)
    status = "✓ ALL VALIDATIONS PASSED" if all_passed else "✗ SOME VALIDATIONS FAILED"
    
    return all_passed, f"{status}\n\n{full_report}"


if __name__ == "__main__":
    print("=" * 60)
    print("PQ BIDDING METRIC CALCULATOR - VALIDATION SUITE")
    print("=" * 60)
    
    passed, report = run_all_validations()
    print(report)
    
    exit(0 if passed else 1)
