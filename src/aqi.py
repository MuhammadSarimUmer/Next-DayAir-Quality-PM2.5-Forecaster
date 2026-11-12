"""
AQI Calculation and Health Advice Module (Task 14).

Translates PM2.5 concentrations (µg/m³) into US EPA AQI index values,
health categories, color codes, and practical actionable guidance for
the general public and sensitive groups.
"""

from typing import Dict, Any
import numpy as np


# EPA Breakpoints for PM2.5 (2024 update: Good is 0.0 - 9.0 µg/m³)
BREAKPOINTS = [
    # (C_low, C_high, I_low, I_high, Category, Color, Hex)
    (0.0, 9.0, 0, 50, "Good", "Green", "#00E400"),
    (9.1, 35.4, 51, 100, "Moderate", "Yellow", "#FFFF00"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "Orange", "#FF7E00"),
    (55.5, 125.4, 151, 200, "Unhealthy", "Red", "#FF0000"),
    (125.5, 225.4, 201, 300, "Very Unhealthy", "Purple", "#8F3F97"),
    (225.5, 500.4, 301, 500, "Hazardous", "Maroon", "#7E0023"),
]


def _truncate_to_breakpoint_resolution(pm25: float) -> float:
    """EPA AQI tables use 0.1 µg/m³ resolution; truncate instead of rounding.

    Truncation closes the nominal gaps between table rows (for example 9.05)
    and prevents such values from falling through every range.
    """
    if pm25 is None or not np.isfinite(pm25):
        raise ValueError("pm25 must be a finite number")
    return float(np.floor(max(0.0, float(pm25)) * 10.0) / 10.0)


def calculate_aqi(pm25: float) -> int:
    """
    Calculate US EPA AQI index value from PM2.5 concentration (µg/m³).

    Formula:
        I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    """
    pm25 = _truncate_to_breakpoint_resolution(pm25)

    # Cap / extrapolate for extreme values above 500 µg/m³
    if pm25 > 500.4:
        return min(999, int(500 + (pm25 - 500.4) * 0.5))

    for c_low, c_high, i_low, i_high, _, _, _ in BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi))

    return 500


def get_aqi_category_info(pm25: float) -> Dict[str, Any]:
    """Get category metadata (name, color, hex) for a given PM2.5 level."""
    pm25 = _truncate_to_breakpoint_resolution(pm25)

    for c_low, c_high, _, _, category, color, hex_code in BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            return {
                "category": category,
                "color_name": color,
                "hex_color": hex_code,
            }

    # Greater than 500.4
    return {
        "category": "Hazardous",
        "color_name": "Maroon",
        "hex_color": "#7E0023",
    }


def get_health_advice(pm25: float) -> Dict[str, Any]:
    """
    Provide structured, actionable public health advice based on predicted PM2.5.
    
    Returns
    -------
    dict with fields:
        - pm25: float concentration
        - aqi: int EPA AQI value
        - category: str
        - hex_color: str
        - summary: str
        - general_public: str
        - sensitive_groups: str
        - actions: list of str
    """
    pm25 = max(0.0, float(pm25))
    aqi_val = calculate_aqi(pm25)
    cat_info = get_aqi_category_info(pm25)
    category = cat_info["category"]
    hex_color = cat_info["hex_color"]

    if category == "Good":
        summary = "Air quality is satisfactory and poses little or no risk."
        general = "Ideal for all outdoor sports, workouts, and outdoor activities."
        sensitive = "No special precautions needed."
        actions = [
            "Enjoy outdoor recreation.",
            "Open windows for fresh air ventilation."
        ]
    elif category == "Moderate":
        summary = "Air quality is acceptable; however, a slight risk may exist for sensitive individuals."
        general = "Normal outdoor activities are safe."
        sensitive = "Unusually sensitive people with asthma or lung conditions should monitor symptoms."
        actions = [
            "Outdoor exercise is generally fine.",
            "Keep inhalers nearby if you have asthma."
        ]
    elif category == "Unhealthy for Sensitive Groups":
        summary = "Members of sensitive groups may experience health effects; general public is less affected."
        general = "OK to be outside, but reduce prolonged high-exertion outdoor activities."
        sensitive = "Children, elderly, and people with respiratory or heart disease should limit prolonged outdoor exertion."
        actions = [
            "Sensitive individuals should wear an N95 mask outdoors.",
            "Close windows during morning traffic and rush hours.",
            "Consider indoor exercise alternatives."
        ]
    elif category == "Unhealthy":
        summary = "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."
        general = "Reduce strenuous outdoor activities. Take frequent breaks."
        sensitive = "Avoid outdoor exertion; keep activities indoors and at low intensity."
        actions = [
            "Wear a well-fitted N95 or KN95 mask outside.",
            "Keep home windows closed; run an air purifier if available.",
            "Avoid strenuous running or cycling outdoors."
        ]
    elif category == "Very Unhealthy":
        summary = "Health alert: The risk of adverse health effects is increased for everyone."
        general = "Avoid all prolonged outdoor exertion. Move workouts indoors."
        sensitive = "Stay indoors and keep activity levels low. Avoid outdoor exposure."
        actions = [
            "Do not exercise outdoors.",
            "Wear an N95 mask if you must go outdoors.",
            "Seal window gaps and use HEPA air purifiers.",
            "Keep emergency rescue inhalers readily accessible."
        ]
    else:  # Hazardous
        summary = "Health warning of emergency conditions: Everyone is likely to be seriously affected."
        general = "Avoid all physical activity outdoors. Remain indoors in clean air spaces."
        sensitive = "Remain strictly indoors and avoid any physical exertion."
        actions = [
            "Stay indoors with doors and windows tightly closed.",
            "Run air purifiers on high setting.",
            "Wear an N95 mask even for short trips outdoors.",
            "Seek medical attention if experiencing breathing distress or chest tightness."
        ]

    return {
        "pm25": round(pm25, 2),
        "aqi": aqi_val,
        "category": category,
        "hex_color": hex_color,
        "summary": summary,
        "general_public": general,
        "sensitive_groups": sensitive,
        "actions": actions,
    }


if __name__ == "__main__":
    sample_values = [5.2, 22.0, 48.0, 85.0, 160.0, 310.0]
    print("=" * 70)
    print("AQI HEALTH ADVICE MODULE DEMO")
    print("=" * 70)
    for val in sample_values:
        adv = get_health_advice(val)
        print(f"\nPM2.5: {adv['pm25']} µg/m³ -> AQI: {adv['aqi']} ({adv['category']})")
        print(f"  Summary: {adv['summary']}")
        print(f"  Advice:  {adv['general_public']}")
        print(f"  Actions: {', '.join(adv['actions'])}")

