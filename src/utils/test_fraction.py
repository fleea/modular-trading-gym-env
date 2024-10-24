import pytest
from src.utils.fraction import calculate_fraction

def test_calculate_fraction_positive_increase():
    # Test when current value is greater than previous value
    assert calculate_fraction(15, 10) == 0.5

def test_calculate_fraction_positive_decrease():
    # Test when current value is less than previous value
    assert calculate_fraction(8, 10) == -0.2

def test_calculate_fraction_no_change():
    # Test when current value equals previous value
    assert calculate_fraction(10, 10) == 0.0

def test_calculate_fraction_zero_previous_value():
    # Test division by zero handling
    with pytest.raises(ZeroDivisionError):
        calculate_fraction(10, 0)

def test_calculate_fraction_zero_both_values():
    # Test when both values are zero
    with pytest.raises(ZeroDivisionError):
        calculate_fraction(0, 0)

def test_calculate_fraction_zero_current_value():
    # -1 is the minimum value
    assert calculate_fraction(0, 10) == -1

def test_calculate_fraction_negative_previous_value():
    # Test that ValueError is raised when previous_value is negative
    with pytest.raises(ValueError, match="previous_value must be positive."):
        calculate_fraction(10, -10)

def test_calculate_fraction_negative_current_value():
    # Test that ValueError is raised when current_value is negative
    with pytest.raises(ValueError, match="current_value must be positive."):
        calculate_fraction(-10, 10)

def test_calculate_fraction_float_values():
    # Test with floating point numbers
    assert calculate_fraction(7.5, 5.0) == 0.5