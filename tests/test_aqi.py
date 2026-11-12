import math

import pytest

from src.aqi import (
    BREAKPOINTS,
    calculate_aqi,
    get_aqi_category_info,
    get_health_advice,
)


# One row per EPA breakpoint boundary: (pm25, expected category, expected AQI)
BREAKPOINT_CASES = [
    (-5.0, "Good", 0),
    (0.0, "Good", 0),
    (9.0, "Good", 50),
    (9.1, "Moderate", 51),
    (35.4, "Moderate", 100),
    (35.5, "Unhealthy for Sensitive Groups", 101),
    (55.4, "Unhealthy for Sensitive Groups", 150),
    (55.5, "Unhealthy", 151),
    (125.4, "Unhealthy", 200),
    (125.5, "Very Unhealthy", 201),
    (225.4, "Very Unhealthy", 300),
    (225.5, "Hazardous", 301),
    (500.4, "Hazardous", 500),
]


@pytest.mark.parametrize("pm25, category, expected_aqi", BREAKPOINT_CASES)
def test_each_breakpoint_boundary(pm25, category, expected_aqi):
    assert get_aqi_category_info(pm25)["category"] == category
    assert calculate_aqi(pm25) == expected_aqi


def test_every_aqi_breakpoint_is_covered():
    assert len(BREAKPOINTS) == 6


def test_boundary_gap_values_do_not_fall_through():
    # Values between the printed table rows are truncated to 0.1 µg/m³,
    # so they must still map to a category instead of the 500 fallback.
    assert get_aqi_category_info(9.05)["category"] == "Good"
    assert get_aqi_category_info(35.45)["category"] == "Moderate"
    assert calculate_aqi(9.05) == 50
    assert calculate_aqi(35.45) == 100


def test_extreme_and_invalid_values():
    assert calculate_aqi(600.0) > 500
    assert calculate_aqi(10000.0) == 999
    with pytest.raises(ValueError):
        calculate_aqi(float("nan"))
    with pytest.raises(ValueError):
        get_aqi_category_info(None)


def test_aqi_is_monotonic():
    values = list(range(0, 400, 5))
    aqis = [calculate_aqi(v) for v in values]
    assert all(a <= b for a, b in zip(aqis, aqis[1:]))


def test_get_aqi_category_info_colors():
    assert get_aqi_category_info(5.0)["hex_color"] == "#00E400"
    assert get_aqi_category_info(25.0)["hex_color"] == "#FFFF00"
    assert get_aqi_category_info(80.0)["hex_color"] == "#FF0000"
    assert get_aqi_category_info(350.0)["hex_color"] == "#7E0023"


def test_get_health_advice_structure():
    advice = get_health_advice(150.0)
    assert "pm25" in advice
    assert "aqi" in advice
    assert "category" in advice
    assert "hex_color" in advice
    assert "summary" in advice
    assert "general_public" in advice
    assert "sensitive_groups" in advice
    assert "actions" in advice
    assert isinstance(advice["actions"], list)
    assert len(advice["actions"]) > 0
    assert advice["category"] == "Very Unhealthy"

    for sample in [0, 20, 45, 80, 160, 300]:
        result = get_health_advice(sample)
        assert math.isfinite(result["pm25"])
        assert result["aqi"] >= 0
        assert result["category"] in {b[4] for b in BREAKPOINTS}
