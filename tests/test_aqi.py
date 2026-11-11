import pytest
from src.aqi import calculate_aqi, get_aqi_category_info, get_health_advice


def test_calculate_aqi_breakpoints():
    # Good
    assert 0 <= calculate_aqi(5.0) <= 50
    assert calculate_aqi(0.0) == 0
    assert calculate_aqi(9.0) == 50

    # Moderate
    assert 51 <= calculate_aqi(20.0) <= 100
    assert calculate_aqi(35.4) == 100

    # Unhealthy for Sensitive Groups
    assert 101 <= calculate_aqi(45.0) <= 150
    assert calculate_aqi(55.4) == 150

    # Unhealthy
    assert 151 <= calculate_aqi(90.0) <= 200
    assert calculate_aqi(125.4) == 200

    # Very Unhealthy
    assert 201 <= calculate_aqi(175.0) <= 300
    assert calculate_aqi(225.4) == 300

    # Hazardous
    assert calculate_aqi(300.0) > 300


def test_get_aqi_category_info():
    info_good = get_aqi_category_info(5.0)
    assert info_good["category"] == "Good"
    assert info_good["hex_color"] == "#00E400"

    info_mod = get_aqi_category_info(25.0)
    assert info_mod["category"] == "Moderate"
    assert info_mod["hex_color"] == "#FFFF00"

    info_unhealthy = get_aqi_category_info(80.0)
    assert info_unhealthy["category"] == "Unhealthy"
    assert info_unhealthy["hex_color"] == "#FF0000"

    info_haz = get_aqi_category_info(350.0)
    assert info_haz["category"] == "Hazardous"
    assert info_haz["hex_color"] == "#7E0023"


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

