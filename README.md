# district_model
techno-economic model of district heating and cooling for UK building context - using Ealing town centre project as case study and data collection: https://zerocarbonaccelerator.london/wp-content/uploads/2025/07/Ealing-Town-Centre-Heat-Network-Feasibility-Report.pdf

Assumptions:

demand_synthesis:
BENCHMARK_REFERENCES = {
    "heating_non_residential": {
        "source": "CIBSE TM46:2008 Energy Benchmarks, Table 1",
        "url": "https://www.cibse.org/knowledge-research/knowledge-portal/tm46-energy-benchmarks",
        "note": "Fossil-thermal typical benchmark. DHW split from total is approximate.",
        "confidence": "HIGH",
    },
    "heating_residential_existing": {
        "source": "Passivhaus Trust / Part L analysis",
        "url": "https://briaryenergy.co.uk/knowledge-bank/parts-l-and-f-of-building-regulations-planned-changes/",
        "note": "130-140 kWh/m2/yr for typical existing UK stock",
        "confidence": "HIGH",
    },
    "heating_residential_new": {
        "source": "Engineering estimate — Part L 2021 SAP modelling typical outcome",
        "url": "https://www.gov.uk/government/publications/building-for-2050",
        "note": "No single published EUI target — varies by dwelling type and SAP route",
        "confidence": "MEDIUM",
    },
    "hospital_total_eui": {
        "source": "NHS ERIC 2024/25 data analysis",
        "url": "https://doi.org/10.3390/buildings16091782",
        "note": "Total EUI 400-460 kWh/m2. Thermal split is approximate.",
        "confidence": "HIGH",
    },
    "cooling_euis": {
        "source": "Ealing Town Centre Feasibility Study (SEL, 2025) + CIBSE Guide F",
        "url": "Local report — see Ealing-Town-Centre-Heat-Network-Feasibility-Report.pdf",
        "note": "100% benchmark-based. Least certain values in the dataset.",
        "confidence": "LOW-MEDIUM",
    },
    "base_load_fractions": {
        "source": "Engineering judgement",
        "url": None,
        "note": "Sensitivity test these — they significantly affect winter baseload sizing.",
        "confidence": "LOW",
    },
}