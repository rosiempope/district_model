"""
grant.py
=========
Green Heat Network Fund (GHNF) grant modelling.

The GHNF (Green Heat Network Fund) is the UK government's main capital
subsidy for new and expanding heat networks. It replaced the HNIP
(Heat Networks Investment Project) in April 2022 and runs through to
March 2028.

Key public parameters (from DESNZ published guidance, last updated
September 2025 — GOV.UK "Green Heat Network Fund: guidance"):

  - Available to new district heat networks in England
  - Covers CAPEX gap funding (the difference between project CAPEX and
    what would be commercially viable without the grant)
  - Maximum grant intensity: UP TO 50% of eligible CAPEX for schemes
    meeting decarbonisation thresholds
  - Minimum scheme size: 2+ buildings, excluding the energy centre
  - Carbon intensity threshold: must demonstrate lower carbon than
    the counterfactual (individual gas)
  - The grant is a ONE-TIME capital contribution (not ongoing revenue
    support), reducing the effective CAPEX in year 0

This module models the grant as a simple percentage reduction of
eligible CAPEX. For a real application, the actual grant percentage
depends on a DESNZ assessment of the "commercialisation gap" — the
model treats it as a user-editable parameter with a sensible default.

Usage
-----
    from economics.grant import apply_ghnf_grant

    result = apply_ghnf_grant(
        total_capex_GBP=8_500_000,
        network_capex_GBP=3_200_000,
        source_capex_GBP=5_300_000,
        grant_rate=0.40,
    )
    print(result["grant_GBP"], result["net_capex_GBP"])
"""

# Default grant rate — 40% is a reasonable mid-case for a well-scoring
# scheme. GHNF allows up to 50%, but most awards are 30-50% depending
# on the commercialisation gap assessment. 40% is a sensible screening
# assumption.
DEFAULT_GHNF_RATE = 0.40
MAX_GHNF_RATE = 0.50

# Eligible CAPEX: network pipework + energy centre sources are eligible.
# Land acquisition, internal building distribution, and planning/legal
# fees are NOT eligible. This module treats all source + network CAPEX
# as eligible (a simplification — some items may be excluded in a real
# application, but for screening purposes this is reasonable).


def apply_ghnf_grant(
    total_capex_GBP,
    network_capex_GBP=0.0,
    source_capex_GBP=0.0,
    grant_rate=DEFAULT_GHNF_RATE,
    cap_GBP=None,
):
    """
    Apply a GHNF-style capital grant to the project CAPEX.

    Parameters
    ----------
    total_capex_GBP   : total project CAPEX (£)
    network_capex_GBP : network (pipework) CAPEX component (£)
    source_capex_GBP  : energy centre sources CAPEX component (£)
    grant_rate        : grant as fraction of eligible CAPEX (0-0.5)
    cap_GBP           : optional absolute cap on the grant (£)

    Returns
    -------
    dict: {
        "eligible_capex_GBP": the CAPEX base the grant is calculated on,
        "grant_rate": the rate applied,
        "grant_GBP": the grant amount (£),
        "net_capex_GBP": total CAPEX minus grant (£),
    }
    """
    grant_rate = min(float(grant_rate), MAX_GHNF_RATE)
    grant_rate = max(grant_rate, 0.0)

    eligible = network_capex_GBP + source_capex_GBP
    if eligible <= 0:
        eligible = total_capex_GBP   # fallback if breakdown not available

    grant_GBP = eligible * grant_rate
    if cap_GBP is not None and grant_GBP > cap_GBP:
        grant_GBP = float(cap_GBP)

    return {
        "eligible_capex_GBP": round(eligible, 0),
        "grant_rate": grant_rate,
        "grant_GBP": round(grant_GBP, 0),
        "net_capex_GBP": round(total_capex_GBP - grant_GBP, 0),
    }
