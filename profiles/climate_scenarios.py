### this is to allow for different climate scenarios to be used in the model, and to allow for different climate scenarios to be assessed to understand future trends due to global warming
"""
Key assumptions:
- The climate deltas are applied to the dry bulb temperature only.
- The dT are based on UKCP18 projections for the UK
- No changes to humidity or solar weather variables
- The climate deltas are applied to the TMY weather data, which is a historical dataset, and therefore the results are not representative of future weather conditions.
"""
import pandas as pd

def apply_climate_scenario(weather_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Apply a climate delta to a TMY weather DataFrame.
    Returns a modified copy — never mutates the original.

    Scenarios
    ---------
    'baseline'     : no change (TMY as-is)
    '2050_central' : UKCP18 RCP4.5 central estimate
                     +2.7°C summer (Jun-Aug), +1.0°C winter (Dec-Feb),
                     +1.8°C shoulder seasons — linear interpolation
    '2050_high'    : UKCP18 RCP8.5 high emissions + urban heat island
                     +4.0°C summer, +2.0°C winter, +3.0°C shoulder
                     plus +2.5°C urban heat island offset year-round
    """
    df = weather_df.copy()
    month = df.index.month

    DELTAS = {
        'baseline':     {m: 0.0 for m in range(1, 13)},
        '2050_central': {
            12: 1.0, 1: 1.0, 2: 1.0,   # winter
            3: 1.4,  4: 1.8, 5: 2.2,   # spring shoulder
            6: 2.7,  7: 2.7, 8: 2.7,   # summer
            9: 2.2,  10:1.8, 11:1.4,   # autumn shoulder
        },
        '2050_high': {
            12: 2.0, 1: 2.0, 2: 2.0,
            3: 2.5,  4: 3.0, 5: 3.5,
            6: 4.0,  7: 4.0, 8: 4.0,
            9: 3.5,  10:3.0, 11:2.5,
        },
    }

    SCENARIOS = {
        "2050_central": {
            "period": "2041-2060",
            "pathway": "RCP4.5",
            "percentile": "50th",
            "location": "Site-specific UKCP18 grid cell",
            "uhi_C": 0.0,
        },
        "2050_high": {
            "period": "2041-2060",
            "pathway": "RCP8.5",
            "percentile": "50th",
            "location": "Site-specific UKCP18 grid cell",
            "uhi_C": 2.5,
        },
    }

    UHI_OFFSET = {'baseline': 0.0, '2050_central': 0.0, '2050_high': 2.5}

    delta = month.map(DELTAS[scenario])
    df['temp_drybulb_C'] = df['temp_drybulb_C'] + delta + UHI_OFFSET[scenario]

    return df