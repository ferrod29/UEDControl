"""Physical constants (CODATA 2018, exact where defined) used across the package."""

SPEED_OF_LIGHT = 299_792_458.0  # m/s
SPEED_OF_LIGHT_MM_PER_PS = SPEED_OF_LIGHT * 1e3 / 1e12  # mm/ps
ELEMENTARY_CHARGE = 1.602_176_634e-19  # C
ELECTRON_MASS = 9.109_383_7015e-31  # kg
ELECTRON_REST_ENERGY_EV = 510_998.950_00  # eV
PLANCK = 6.626_070_15e-34  # J s
HBAR = PLANCK / (2 * 3.141_592_653_589_793)  # J s

# Handy derived value: one elementary charge expressed in femtocoulomb.
ELEMENTARY_CHARGE_FC = ELEMENTARY_CHARGE * 1e15
