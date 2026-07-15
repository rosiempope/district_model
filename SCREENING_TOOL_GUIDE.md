# Internal screening tool guide

## What the tool can decide

Version 2.4 supports a consistent early-stage decision across heating-only and
four-pipe heating/cooling options. A scenario is not labelled feasible merely
because its NPV is positive. It is tested against the criteria recorded in the
scenario JSON:

1. annual unmet heat/cooling tolerance;
2. operational carbon ceiling;
3. minimum investor NPV;
4. investor IRR hurdle;
5. optional N-1 peak-capacity requirement; and
6. optional maximum break-even heat tariff.

The UI, result export and comparison table use the same decision object. A
`CONDITIONAL PASS` means the selected gates pass but evidence gaps remain, such
as an equivalent-trunk route, archetype demand or N-1 not being mandatory.

## Recommended internal workflow

1. Build one common customer and route case. Use measured annual demand and
   peak data where available; record connection year and probability.
2. Prefer tree network mode. Use generic length only for a first conversation
   and expect a conditional result.
3. Compare source options without changing the demand, route, commercial
   hurdle or price basis. Add a separate scenario only when an assumption must
   genuinely change.
   Keep customer heat bills on gas-counterfactual parity unless the purpose is
   an explicitly labelled tariff sensitivity. Use the equivalent heat tariff
   reported by the model rather than comparing the district tariff directly
   with the gas input unit rate.
4. Set the internal hurdle, unmet-energy tolerance, carbon limit and (if known)
   maximum acceptable heat tariff before running the cases.
5. Review the gate table first, then the evidence actions, then the detailed
   CAPEX/OPEX and annual cash-flow audits.
6. Use sensitivities to find the switching variables: route length/heat
   density, source temperature and availability, electricity-to-heat price
   spread, customer contribution/grant, connection phasing and tariff.

## New technical and economic drivers

- linear heat and cooling density (MWh per route metre per year);
- network heat-loss share and network CAPEX per metre;
- unmet heat/cooling energy percentages;
- peak available capacity and N-1 firm-capacity margin;
- CAPEX per kW of peak service;
- gross and grant-adjusted CAPEX;
- investor NPV, IRR, hurdle rate, discounted payback and peak funding;
- current heat tariff and break-even heat tariff;
- whole-system NPV versus the selected individual-system counterfactual; and
- one combined decision with explicit failed gates and evidence actions.
- full-buildout and year-1 OPEX reconciliation, including hourly pumping cost;
- customer heat/cooling bill ratios against individual gas/AC alternatives.

## Graphs to use internally

The application now includes the first three priority views:

1. **Investor NPV versus operational carbon** — bubble size is CAPEX and colour
   is the screening decision. This quickly separates low-carbon value from
   expensive or high-carbon options.
2. **Cumulative discounted investor cash position over project life** — shows
   whether a positive endpoint depends on late cash flows and makes REPEX dips
   visible.
3. **Selectable scenario metric bar chart** — useful for route length, heat
   density, CAPEX, OPEX, break-even tariff, unmet demand or N-1 margin.

For the next assurance phase, the most useful additions would be:

- tornado chart of NPV sensitivity and switching values;
- heat/cooling load-duration curve with installed and N-1 firm capacity;
- annual source-mix stacked area chart with electricity/gas/third-party heat;
- CAPEX and OPEX stacked bars by line item;
- route heat-density versus break-even tariff scatter; and
- demand/connection ramp with downside, central and upside cases.

## What the decision does not prove

The N-1 calculation is a peak-hour capacity screen, not a dynamic outage or
network-resilience simulation. The physical year is repeated through the
financial life. GIS routing, utility conflicts, detailed hydraulics, network
outages, construction phasing, debt, tax, VAT and formal customer-detriment or
grant eligibility tests remain outside the model. Do not present a pass as an
investment approval.
