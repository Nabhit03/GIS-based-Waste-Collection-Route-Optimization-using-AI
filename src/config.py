"""
Project-wide configuration.
Edit these values to point at a real city / change fleet parameters.
"""

# --- Location ---
CITY_NAME = "Chandigarh, India"   # used if OSMnx/internet is available
NETWORK_TYPE = "drive"

# --- Fleet ---
NUM_TRUCKS = 6
TRUCK_CAPACITY = 60          # max demand units a truck can service per trip
                              # (kept comfortably above avg total demand / num_trucks)
DEPOT_INDEX = 0              # index of depot in the points list (added automatically)

# --- Tier 3 solver tuning (workload balance across trucks) ---
# OR-Tools' CVRP solver only uses as many vehicles as it needs to minimize total
# cost -- if fewer trucks can handle all demand, it will leave others empty and/or
# produce very uneven stop counts per truck (both purely to shave a few minutes of
# depot-to-first-stop travel). These two settings counteract that:

# If True, the effective balance weight is auto-scaled to this problem's real
# travel-time magnitude (average minutes between stops * the multiplier
# below), instead of using the fixed WORKLOAD_BALANCE_WEIGHT value. A fixed
# number that works for a small synthetic grid (short travel times) becomes
# meaningless on a real city with much larger travel times -- the solver will
# just ignore a balance penalty that's small relative to genuine time
# savings, which is exactly what was observed on real Chandigarh data
# (workload_std_stops stayed high despite the dimension being active).
# Auto-scaling keeps the balance penalty meaningful regardless of city size.
WORKLOAD_BALANCE_AUTO_SCALE = True

# "Worth", in average stop-to-stop travel times, of imbalance the solver will
# trade off against. E.g. 6 means the solver treats a 1-stop increase in the
# gap between busiest/least-busy truck as roughly as costly as 6 average
# inter-stop trips -- strong enough to actually move stops around for balance,
# not just a token penalty.
WORKLOAD_BALANCE_AUTO_MULTIPLIER = 6

# Only used if WORKLOAD_BALANCE_AUTO_SCALE is False, or when a caller passes
# workload_balance_weight=None to solve_cvrptw() with auto-scale disabled.
WORKLOAD_BALANCE_WEIGHT = 50

# If True, forces every vehicle to be assigned at least a minimum number of stops
# (see MIN_STOPS_FRACTION_OF_FAIR_SHARE below), so trucks are never left idle just
# because a smaller subset could technically cover all demand. If this makes the
# problem infeasible (e.g. capacity or time-window constraints don't allow it),
# the solver automatically retries with this relaxed and prints a warning --
# it will never fail silently or crash because of this setting.
ENFORCE_MIN_STOPS_PER_VEHICLE = True

# Minimum stops per vehicle = this fraction * (total stops / NUM_TRUCKS), rounded
# down, minimum 1. e.g. 0.5 with 85 stops / 6 trucks (~14 fair share) requires each
# truck get at least 7 stops -- enough to guarantee all trucks are used, but loose
# enough to leave the solver room to still rebalance around real constraints.
MIN_STOPS_FRACTION_OF_FAIR_SHARE = 0.5

# Cost (in "minutes" of arc-cost-equivalent) charged per minute a stop is
# serviced after its time-window deadline. High enough that the solver will
# always prefer an on-time route when one exists, but finite -- so a few
# genuinely unreachable deadlines (common with real road-network travel
# times) produce a late arrival + a counted violation instead of making the
# entire solve infeasible. Set to 0 to restore old hard-deadline behavior
# (not recommended on real OSM data -- see vrp_solver.py for why).
TIME_WINDOW_LATE_PENALTY_PER_MINUTE = 500

# --- Demand simulation (used only if you don't supply real data) ---
NUM_COLLECTION_POINTS = 80
RANDOM_SEED = 42

# --- Time windows ---
# Simulation minutes are relative to a single shift starting at 0 = 6:00 AM
SHIFT_START_MINUTE = 0        # 6:00 AM -> minute 0
COMMERCIAL_DEADLINE_MINUTE = 180   # commercial stops must be visited by 9:00 AM (180 min after 6 AM)
SHIFT_END_MINUTE = 360         # depot closes route acceptance at 12:00 PM (6 hrs shift)
SERVICE_TIME_MINUTES = 3       # time spent servicing each stop (loading waste)

# Fraction of stops that are "commercial" (hard morning deadline) vs "residential" (flexible)
COMMERCIAL_FRACTION = 0.3

# --- Vehicle speed assumption (used when real edge speed data is missing) ---
DEFAULT_SPEED_KMPH = 25

# --- Fuel / emissions assumptions (for reporting only) ---
TRUCK_FUEL_ECONOMY_KM_PER_L = 3.5     # garbage trucks are heavy, low mileage
DIESEL_CO2_KG_PER_L = 2.68

# --- Output paths ---
OUTPUT_DIR = "outputs"
