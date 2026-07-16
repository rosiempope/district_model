"""Birmingham heat network zoning — scenarios built from the real DESNZ report.

Source
------
DESNZ, "Heat Network Zoning: Zone Opportunity Report — Birmingham", February 2025.
assets.publishing.service.gov.uk/media/679b598d15f01fdf8e05e79a/
  heat-network-zone-opportunity-report-birmingham.pdf

Produced under the Heat Network Zoning Pilot programme, which applied a
standard set of technical and economic assumptions across the 28 participating
areas. Every figure in this module is quoted from that report with its table or
figure number. Nothing here is a model preset, and nothing is invented — where
the report does not give a number, this module says so rather than filling the
gap from the model's own defaults.

What the report gives, and what it does not
--------------------------------------------
GIVES:  network length and network cost per IZO (Tables 7, 12, 17, 22)
        total CapEx, annual heat, linear heat density, CO2e savings (Tables 3, 8, ...)
        the top-10 named heat demands with real annual MWh (Table 4)
        the full demand split by building category in MWh (Figure 7)
        the heat sources with capacity in kWp and SOURCE temperature (Table 5)
        the existing BDEC network's three growth scenarios (Table 2)
        the design target: linear heat density of 4 MWh/m/yr for existing built stock

DOES NOT GIVE:
        network design temperatures (flow/return) — NOT PRESENT anywhere in the
          report. The "Temperature (C)" column of Table 5 is the SOURCE
          temperature (8C air, 10C river, 12C ground), not a network design point.
        the £/m cost basis — Tables 7/12/17/22 point to "Appendix 5 for the
          assumptions used", and Appendix 5 is NOT in this PDF. This document
          carries Appendices 1 (maps) and 2 (data room) only; 3, 4 and 5 live in
          a separate main report. So the report's £/m is an OUTPUT we can compare
          against, not a method we can reproduce.
        tariffs, discount rate, NPV, IRR — NOT PRESENT. The programme used "a
          proxy for economic viability", not a cash-flow model.
        per-building floor areas, peak demands, or connection counts beyond
          "1 connection" for each named building.

Costs are report values, not model values
------------------------------------------
Every scenario here sets network.capex_GBP_override to the report's own network
cost. That is the point of the exercise: the model's SEAI-fitted pipe curve
(£1,158/m at DN100) is NOT used for these runs. The comparison between the two
is the study's main finding — see reports/birmingham_study.py.

    IZO                     Length   Network cost   Implied £/m
    Birmingham Central      40 km    £150m          3,750
    Tyseley Central         25 km    £75m           3,000
    North East              40 km    £100m          2,500
    QE Hospital / Uni        4 km    £7m            1,750   <- reuses existing pipe

The spread is not noise. Birmingham Central crosses New Street Station, an
underground rail line, a tram line, three multi-lane highway junctions and two
canals (report constraints C1-C4), and the report notes "significant congestion
of existing services beneath many of the roadways and footpaths". The QE
Hospital IZO, at less than half the cost per metre, "utilises the existing heat
network pipes serving the Queen Elizabeth Hospital Heritage heat network and the
University of Birmingham's low temperature hot water network branch".

A known modelling gap: WSHP and GSHP
-------------------------------------
Birmingham Central's identified supply is "ASHPs and WSHPs" (Table 3). Of the
~23 MWth identified in Table 5, 14.8 MWth is ASHP — which this model can
represent — while 5 MWth of river WSHP and 1.2+ MWth of GSHP CANNOT be: the
model's HEAT_SOURCE_TYPES has no water- or ground-source heat pump. They are not
substitutable for an ASHP, because a 10C river or 12C ground source is a far
smaller and far more stable lift than UK winter air. Modelling them as ASHPs
would understate their COP and overstate their carbon.

So these scenarios carry the report's real 14.8 MW of ASHP and leave the
6.2+ MW of WSHP/GSHP out, with gas peak covering the balance. That makes the
carbon and OPEX figures CONSERVATIVE (worse than the report's intent), and it is
the single biggest reason not to read these NPVs as a verdict on Birmingham's
actual plan. Adding a water/ground-source type is the obvious next step.
"""
from __future__ import annotations

# ── Report constants ──────────────────────────────────────────────────────────

REPORT_CITATION = (
    "DESNZ, Heat Network Zoning: Zone Opportunity Report — Birmingham, February 2025"
)

# Design target for the IZOs, quoted: "Initial zone opportunity design targeted a
# linear heat density (LHD) of 4MWh/m/yr, for the existing built environment.
# This is considered a relatively low proxy for economic viability with the heat
# network sector in England."
REPORT_DESIGN_LHD_TARGET_MWh_per_m = 4.0

# Tables 7, 12, 17, 22 — indicative heat network statistics per IZO.
# network_cost_GBP is the report's own figure and is used directly.
IZO_NETWORK = {
    "Birmingham Central": {
        "length_km": 40.0, "network_cost_GBP": 150e6,
        "total_capex_GBP": 325e6,          # Table 3
        "annual_heat_GWh": 260.0,          # Section 3.1.4 text (Table 3 says ">250")
        "linear_heat_density_MWh_per_m": 6.7,   # Table 3
        "co2e_saving_ktCO2e_yr": 40.0,     # Table 3
        "buildings": 439,                  # Section 3.1.4 text
        "sources_described": "ASHPs and WSHPs",
        "table": "Tables 3 and 7",
    },
    "Tyseley Central": {
        "length_km": 25.0, "network_cost_GBP": 75e6,
        "total_capex_GBP": 150e6,          # Table 8
        "annual_heat_GWh": 73.0,           # Section 3.2.1 text (Table 8 says ">50")
        "linear_heat_density_MWh_per_m": 2.9,
        "co2e_saving_ktCO2e_yr": 5.0,
        "buildings": 157,
        "sources_described": "ERF and Biomass",
        "table": "Tables 8 and 12",
    },
    "North East": {
        "length_km": 40.0, "network_cost_GBP": 100e6,
        "total_capex_GBP": 200e6,
        "annual_heat_GWh": 150.0,
        "linear_heat_density_MWh_per_m": 3.3,
        "co2e_saving_ktCO2e_yr": 20.0,
        "buildings": None,                 # NOT PRESENT for this zone
        "sources_described": "Minworth STW",
        "table": "Tables 13 and 17",
    },
    "Queen Elizabeth Hospital / University of Birmingham": {
        "length_km": 4.0, "network_cost_GBP": 7e6,
        "total_capex_GBP": 50e6,
        "annual_heat_GWh": 100.0,
        "linear_heat_density_MWh_per_m": 21.5,
        "co2e_saving_ktCO2e_yr": 15.0,
        "buildings": None,                 # NOT PRESENT for this zone
        "sources_described": "ASHPs",
        "table": "Tables 18 and 22",
        "reuses_existing_pipework": True,
        "existing_pipework_note": (
            "The proposed IZO distribution network utilises the existing heat network pipes "
            "serving the Queen Elizabeth Hospital Heritage heat network and the University of "
            "Birmingham's low temperature hot water network branch."
        ),
    },
}

# Table 2 — Birmingham District Energy Company decarbonisation and expansion.
# This is the EXISTING network in the city centre, with a real costed growth path.
BDEC_GROWTH_SCENARIOS = {
    "Low": {"annual_demand_GWh": 74.0, "capex_GBP": 16e6,
            "sources": "Data Centre and ASHPs", "construction_start": 2026},
    "Medium": {"annual_demand_GWh": 176.0, "capex_GBP": 62e6,
               "sources": "WSHP (Edgbaston Reservoir)", "construction_start": 2026},
    "High": {"annual_demand_GWh": 661.0, "capex_GBP": 545e6,
             "sources": "WSHPs (reservoir, river, STW)", "construction_start": 2026},
}

# Table 5 — Birmingham Central key heat source opportunities.
# NOTE the temperature column is the SOURCE temperature, not a network design point.
CENTRAL_HEAT_SOURCES = [
    {"name": "WSHP River Rea", "type": "WSHP", "capacity_kWp": 5000, "source_temp_C": 10, "ec_ref": "E6"},
    {"name": "GSHP Birmingham Children's Hospital", "type": "GSHP", "capacity_kWp": None, "source_temp_C": 12, "ec_ref": "E1"},
    {"name": "ASHP Aston energy centre", "type": "ASHP", "capacity_kWp": 2200, "source_temp_C": 8, "ec_ref": "E2"},
    {"name": "GSHP Aston University", "type": "GSHP", "capacity_kWp": 1200, "source_temp_C": 12, "ec_ref": "E3"},
    {"name": "ASHP Bullring", "type": "ASHP", "capacity_kWp": 2800, "source_temp_C": 8, "ec_ref": "E4"},
    {"name": "ASHP Birmingham New Street Station", "type": "ASHP", "capacity_kWp": 2800, "source_temp_C": 8, "ec_ref": "E5"},
    {"name": "ASHP Smithfield energy centre", "type": "ASHP", "capacity_kWp": 7000, "source_temp_C": 8, "ec_ref": "E13"},
]
CENTRAL_TOTAL_IDENTIFIED_SUPPLY_MW = 23.0   # Section 3.1.5: "approximately 23MWth"


def _total_MW(source_type: str) -> float:
    return sum(
        s["capacity_kWp"] for s in CENTRAL_HEAT_SOURCES
        if s["type"] == source_type and s["capacity_kWp"] is not None
    ) / 1000.0


CENTRAL_ASHP_TOTAL_MW = _total_MW("ASHP")   # 14.8 MW
CENTRAL_WSHP_TOTAL_MW = _total_MW("WSHP")   # 5.0 MW — River Rea
CENTRAL_GSHP_TOTAL_MW = _total_MW("GSHP")   # 1.2 MW — Aston University only; the
                                            # Children's Hospital GSHP has capacity
                                            # "Unknown" in Table 5 and is excluded
                                            # rather than guessed.
CENTRAL_REPRESENTABLE_MW = CENTRAL_ASHP_TOTAL_MW + CENTRAL_WSHP_TOTAL_MW + CENTRAL_GSHP_TOTAL_MW  # 21.0

# Table 4 — the ten named heat demands, with the report's own annual MWh.
# building_type only shapes the HOURLY PROFILE here: annual_heat_kWh is supplied
# directly from the report, so the archetype benchmark never sets the magnitude.
CENTRAL_NAMED_DEMANDS = [
    {"report_name": "Birmingham New Street Station", "category": "Non-Domestic",
     "annual_MWh": 14650, "type": "mixed_use",
     "note": "Largest single heat user identified. Contains an existing gas-fired CHP heating system."},
    {"report_name": "Bullring Shopping Centre West", "category": "Non-Domestic",
     "annual_MWh": 7100, "type": "retail"},
    {"report_name": "Smithfield Development", "category": "New Developments",
     "annual_MWh": 6550, "type": "mixed_use"},
    {"report_name": "Millennium Point", "category": "Non-Domestic",
     "annual_MWh": 5850, "type": "office"},
    {"report_name": "Bullring Shopping Centre East", "category": "Non-Domestic",
     "annual_MWh": 5800, "type": "retail"},
    {"report_name": "New Monaco Development", "category": "New Developments",
     "annual_MWh": 5500, "type": "mixed_use"},
    {"report_name": "House of Fraser", "category": "Non-Domestic",
     "annual_MWh": 4300, "type": "retail"},
    {"report_name": "Eastside Locks Development", "category": "New Developments",
     "annual_MWh": 4150, "type": "mixed_use"},
    {"report_name": "Cannon House", "category": "Non-Domestic",
     "annual_MWh": 3200, "type": "office"},
    {"report_name": "Colmore Plaza", "category": "Non-Domestic",
     "annual_MWh": 3100, "type": "office"},
]

# Figure 7 — the full Birmingham Central demand split, in MWh/yr. These are the
# report's own plotted values, summing to 260,394 MWh (the "260GWh/yr" headline).
CENTRAL_DEMAND_BY_CATEGORY_MWh = {
    "Non-Domestic": 199_767,
    "Residential": 27_569,
    "New Developments": 20_246,
    "Council owned": 12_461,
    "Public Sector": 351,
}

# Model archetype used for each report category's UNNAMED residual. The report
# gives no floor areas or per-building splits below the top ten, so the residual
# is carried as one block per category at the report's own category total minus
# the named buildings in it. That block's archetype sets its hourly SHAPE only —
# its annual energy is the report's number.
_CATEGORY_ARCHETYPE = {
    "Non-Domestic": "office",
    "Residential": "residential_existing",
    "New Developments": "mixed_use",
    "Council owned": "office",
    "Public Sector": "office",
}


def central_anchor_buildings() -> list[dict]:
    """The ten NAMED Birmingham Central demands only (Table 4), 60.2 GWh/yr.

    100% report data: real building names, real annual MWh, real connection counts.
    No lumping, no residual, no derivation.

    This is the subset the model can actually run. The full 439-building zone
    cannot be, for two separate reasons — see central_buildings() and the note in
    reports/birmingham_study.py.
    """
    return [
        {
            "name": d["report_name"],
            "type": d["type"],
            "annual_heat_kWh": float(d["annual_MWh"]) * 1000.0,
            "connections": 1,
            "connection_year": 1,
            "connection_probability": 1.0,
        }
        for d in CENTRAL_NAMED_DEMANDS
    ]


def central_buildings() -> list[dict]:
    """Birmingham Central IZO demand: the ten named buildings plus a residual
    block per category, reconciling exactly to Figure 7's totals.

    Every annual_heat_kWh is the report's own measured/benchmarked figure. The
    archetype on each entry shapes the 8,760-hour profile; it never sets the
    annual total.
    """
    buildings = []
    for d in CENTRAL_NAMED_DEMANDS:
        buildings.append({
            "name": d["report_name"],
            "type": d["type"],
            "annual_heat_kWh": float(d["annual_MWh"]) * 1000.0,
            "connections": 1,                 # Table 4, "Number of connections"
            "connection_year": 1,
            "connection_probability": 1.0,
        })

    named_by_category: dict[str, float] = {}
    for d in CENTRAL_NAMED_DEMANDS:
        named_by_category[d["category"]] = named_by_category.get(d["category"], 0.0) + d["annual_MWh"]

    named_count = len(CENTRAL_NAMED_DEMANDS)
    residual_buildings = int(IZO_NETWORK["Birmingham Central"]["buildings"]) - named_count
    residual_total_MWh = sum(
        CENTRAL_DEMAND_BY_CATEGORY_MWh[c] - named_by_category.get(c, 0.0)
        for c in CENTRAL_DEMAND_BY_CATEGORY_MWh
    )

    for category, total_MWh in CENTRAL_DEMAND_BY_CATEGORY_MWh.items():
        residual_MWh = total_MWh - named_by_category.get(category, 0.0)
        if residual_MWh <= 0:
            continue
        # Connections are apportioned across the residual by that category's share
        # of residual energy — the report gives a building count for the zone but
        # not per category, so this is a stated derivation, not a report figure.
        share = residual_MWh / residual_total_MWh if residual_total_MWh else 0.0
        buildings.append({
            "name": f"{category} — balance of zone",
            "type": _CATEGORY_ARCHETYPE[category],
            "annual_heat_kWh": float(residual_MWh) * 1000.0,
            "connections": max(1, int(round(residual_buildings * share))),
            "connection_year": 1,
            "connection_probability": 1.0,
        })
    return buildings


# ── Scenario builders ─────────────────────────────────────────────────────────

def _economics_from_report(izo: str, peak_total_MW: float) -> tuple[dict, dict]:
    """Economics for an IZO, with the report's own costs substituted in.

    The report gives a TOTAL CapEx and a NETWORK cost per IZO, and nothing else.
    So:
      - network cost      -> network.capex_GBP_override (the report's figure, used directly)
      - the remainder     -> energy_centre_building_GBP, as ONE lumped line

    The remainder is real money from the report (total CapEx minus network cost)
    but the report never breaks it down, so lumping it is honest: it is not
    "energy centre building" in any literal sense, it is everything the report
    counted that is not pipe. It is parked on that key because the engine needs
    somewhere to put it. Every other fixed CAPEX line is therefore ZEROED — the
    Ealing-calibrated defaults would otherwise double-count against the report's
    own total.

    The percentage adders are also zeroed: the report's CapEx is a delivered
    scheme cost, so adding a design fee and a contingency on top would inflate a
    number that already includes them (on whatever basis Appendix 5 used, which
    is not in this PDF).

    Tariffs, discount rate and overheads are NOT in the report. Those keep the
    model's own assumptions and are flagged in the study output. This is the
    honest split: report costs where the report has costs, model assumptions
    where it does not, and a clear line between them.
    """
    from scenarios.fixed_cost_scaling import scaled_economics

    meta = IZO_NETWORK[izo]
    econ, scale = scaled_economics(peak_total_MW)

    non_network_capex = float(meta["total_capex_GBP"]) - float(meta["network_cost_GBP"])
    econ["capex_items"] = {
        # Report: total CapEx minus network cost. One lumped line — see docstring.
        "energy_centre_building_GBP": non_network_capex,
        "land_and_enabling_GBP": 0.0,
        "electricity_connection_GBP": 0.0,
        "gas_connection_GBP": 0.0,
        "controls_and_scada_GBP": 0.0,
        # The report gives ONE total CapEx which already includes connections.
        # Pricing them again from DECC components would double-count.
        "connection_cost_mode": "flat_per_connection",
        "customer_connection_GBP_per_connection": 0.0,
        "metering_GBP_per_connection": 0.0,
        # Zeroed: the report's CapEx is already a delivered scheme cost.
        "development_and_design_pct": 0.0,
        "commissioning_pct": 0.0,
        "contingency_pct": 0.0,
    }
    provenance = {
        "network_capex_GBP": float(meta["network_cost_GBP"]),
        "network_capex_source": f"{REPORT_CITATION}, {meta['table']}",
        "non_network_capex_GBP": non_network_capex,
        "non_network_capex_source": "Report total CapEx minus report network cost; not broken down in the report",
        "fixed_opex_scale_factor": round(scale, 4),
        "fixed_opex_source": "MODEL ASSUMPTION — the report gives no operating costs",
        "tariff_source": "MODEL ASSUMPTION — the report gives no tariffs",
        "discount_rate_source": "MODEL ASSUMPTION — the report gives no discount rate",
    }
    return econ, provenance


def central_izo_scenario(name: str = "Birmingham Central IZO (report costs)",
                         heat_flow_temp_C: float = 70.0,
                         heat_return_temp_C: float = 40.0,
                         dhw_system: str = "instantaneous_hiu",
                         use_report_network_cost: bool = True) -> tuple[dict, dict]:
    """The Birmingham Central IZO, built from the report's real figures.

    Design temperatures are NOT in the report — 70/40 is this model's default and
    is passed explicitly here so the assumption is visible rather than inherited.

    use_report_network_cost=False drops the override and lets the model's own
    SEAI-fitted pipe curve size and cost the route instead. That is the
    comparison the study turns on.
    """
    meta = IZO_NETWORK["Birmingham Central"]
    buildings = central_anchor_buildings()

    # The ANCHOR-LOAD core only: the ten named buildings, 60.2 of the zone's
    # 260 GWh/yr, across 60.2/260 = 23% of the route pro-rata. The full
    # 439-building zone is not modellable here, for two independent reasons:
    #
    #  1. The report names only the top ten. Carrying the other 429 as lumped
    #     per-category blocks (central_buildings()) destroys inter-building
    #     diversity — every block then peaks in the same hour, producing a 331 MW
    #     peak at a load factor of 0.09, which is not a real network.
    #  2. Even with realistic diversity, a 260 GWh zone peaks well beyond what a
    #     single DN600 trunk can carry (~80 MW at 70/40). generic_length mode
    #     models one equivalent trunk, so it cannot represent this zone at all.
    #     That is a real constraint, not a modelling nuisance: a zone-scale route
    #     needs parallel mains or a genuine branched tree, and the report's own
    #     Figure 6 shows a branched route.
    #
    # Route is pro-rated by the anchor loads' share of zone heat, holding the
    # report's own £/m. That keeps the £/m real and the linear heat density
    # honest, and is stated rather than hidden.
    anchor_MWh = sum(d["annual_MWh"] for d in CENTRAL_NAMED_DEMANDS)
    anchor_share = anchor_MWh / (meta["annual_heat_GWh"] * 1000.0)
    route_m = meta["length_km"] * 1000.0 * anchor_share
    report_per_m = meta["network_cost_GBP"] / (meta["length_km"] * 1000.0)

    approx_peak_MW = anchor_MWh / 8760.0 / 0.30 / 1000.0

    econ, provenance = _economics_from_report("Birmingham Central", approx_peak_MW)
    # Non-network CapEx is pro-rated on the same basis.
    econ["capex_items"]["energy_centre_building_GBP"] *= anchor_share
    provenance["anchor_load_share_of_zone_heat"] = round(anchor_share, 4)
    provenance["route_m_prorated"] = round(route_m, 0)
    provenance["report_network_cost_per_m"] = round(report_per_m, 0)

    network = {
        "mode": "generic_length",
        "length_m": route_m,
        "include_cooling": False,
        "heat_flow_temp_C": heat_flow_temp_C,
        "heat_return_temp_C": heat_return_temp_C,
        "dhw_system": dhw_system,
    }
    if use_report_network_cost:
        network["capex_GBP_override"] = report_per_m * route_m

    scenario = {
        "name": name,
        "description": (
            f"Built from {REPORT_CITATION}. Network cost, total CapEx, demands and heat-source "
            f"capacities are the report's own figures. Design temperatures, tariffs, discount rate "
            f"and operating costs are model assumptions — the report contains none."
        ),
        "climate_scenario": "baseline",
        "demand": {"buildings": buildings},
        "network": network,
        "sources": [
            # The report's real 14.8 MW of ASHP (Table 5). The 5 MW river WSHP and
            # 1.2+ MW of GSHP have no model source type and are omitted — see the
            # module docstring. This makes carbon and OPEX conservative.
            # The report's real ASHP capacity (Table 5), pro-rated to the anchor
            # core's share of zone heat on the same basis as the route and CapEx.
            {"type": "ashp", "preset": "large_energy_centre", "name": "ASHP (report Table 5)",
             "capacity_MW": round(CENTRAL_ASHP_TOTAL_MW * anchor_share, 2), "n_units": 4,
             "flow_temp_C": heat_flow_temp_C},
            # The report's real river WSHP and Aston GSHP (Table 5). These lift
            # from a 10C river and a 12C borehole, not from winter air — which is
            # why they cannot be approximated as ASHPs. See
            # components/water_ground_source_hp.py.
            {"type": "wshp", "preset": "birmingham_river_rea", "name": "WSHP River Rea (report Table 5)",
             "capacity_MW": round(CENTRAL_WSHP_TOTAL_MW * anchor_share, 2), "n_units": 2},
            {"type": "gshp", "preset": "birmingham_aston_university",
             "name": "GSHP Aston University (report Table 5)",
             "capacity_MW": round(CENTRAL_GSHP_TOTAL_MW * anchor_share, 2), "n_units": 2},
            # Gas peak covers the balance. The report itself identifies a supply
            # deficit in this zone: ~23 MWth of source against 260 GWh/yr of demand,
            # which averages ~30 MW before any peak diversity.
            {"type": "gas_boiler", "preset": "ealing_phase2", "name": "Gas peak/backup",
             "capacity_MW": 70.0},
        ],
        "economics": econ,
        "screening": {
            "maximum_unmet_energy_fraction": 0.001,
            "maximum_carbon_gCO2e_per_kWh": 100.0,
            "require_n_minus_one": False,
        },
    }
    return scenario, provenance
