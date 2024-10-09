def calculate_fraction(current_value, previous_value):
    """
    Calculate the fraction (current_value - previous_value) / previous_value.
    
    :param current_value: The current numerical value.
    :param previous_value: The previous numerical value.
    :return: The calculated fraction.
    :raises ZeroDivisionError: If previous_value is zero.
    """
    if previous_value == 0:
        raise ZeroDivisionError("previous_value cannot be zero.")
    if previous_value < 0:
        raise ValueError("previous_value must be positive.")
    if current_value < 0:
        raise ValueError("current_value must be positive.")
    return (current_value - previous_value) / previous_value