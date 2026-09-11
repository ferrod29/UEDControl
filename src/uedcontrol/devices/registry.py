"""Driver registry: maps the ``driver`` keys of the configuration to classes.

Classes are imported lazily, so a missing vendor SDK (pypylon, pipython...)
only matters for the instrument that needs it.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Device

_PKG = "uedcontrol.devices"

DRIVERS: dict[str, str] = {
    # Simulated instruments (development, demos, tests)
    "simulated_camera": f"{_PKG}.simulated:SimulatedCamera",
    "simulated_stage": f"{_PKG}.simulated:SimulatedStage",
    "simulated_power_supply": f"{_PKG}.simulated:SimulatedPowerSupply",
    "simulated_hv_supply": f"{_PKG}.simulated:SimulatedHVSupply",
    "simulated_mirrors": f"{_PKG}.simulated:SimulatedMirrors",
    "simulated_vacuum_gauge": f"{_PKG}.simulated:SimulatedVacuumGauge",
    "simulated_rf_amplifier": f"{_PKG}.simulated:SimulatedRFAmplifier",
    "simulated_rf_attenuator": f"{_PKG}.simulated:SimulatedRFAttenuator",
    "simulated_rf_sensor": f"{_PKG}.simulated:SimulatedRFPowerSensor",
    # Cameras
    "dectris_quadro": f"{_PKG}.cameras.dectris:DectrisCamera",
    "basler": f"{_PKG}.cameras.basler:BaslerCamera",
    "pco": f"{_PKG}.cameras.pco:PCOCamera",
    "micromanager": f"{_PKG}.cameras.micromanager:MicroManagerCamera",
    # Motion
    "newport_agilis_uc8": f"{_PKG}.motion.newport_agilis:NewportAgilisUC8",
    "newport_smc100": f"{_PKG}.motion.newport_smc100:NewportSMC100",
    "pi_mercury_c663": f"{_PKG}.motion.pi_mercury:PIMercuryC663",
    "thorlabs_kdc101": f"{_PKG}.motion.thorlabs_kdc101:ThorlabsKDC101",
    "attocube_ecc100": f"{_PKG}.motion.attocube_ecc100:AttocubeECC100Axis",
    # Power supplies
    "matsusada_cohvu32": f"{_PKG}.power.matsusada:MatsusadaCOHVU32",
    "heinzinger_pnchp100": f"{_PKG}.power.heinzinger:HeinzingerPNChp100",
    "rs_hmp4040": f"{_PKG}.power.rohde_schwarz:RohdeSchwarzHMP4040",
    "rs_nge100": f"{_PKG}.power.rohde_schwarz:RohdeSchwarzNGE100",
    "delta_sm70ar24": f"{_PKG}.power.delta_elektronika:DeltaSM70AR24",
    # RF chain
    "minicircuits_power_sensor": f"{_PKG}.rf.minicircuits:MiniCircuitsPowerSensor",
    "minicircuits_attenuator": f"{_PKG}.rf.minicircuits:MiniCircuitsAttenuator",
    "drxworks_statera100": f"{_PKG}.rf.statera:DrXWorksStatera100",
    # Monitoring
    "newport_1935c": f"{_PKG}.sensors.newport_1935c:Newport1935C",
    "edwards_tic": f"{_PKG}.sensors.edwards_tic:EdwardsTIC",
    "agilent_twistorr": f"{_PKG}.sensors.agilent_twistorr:AgilentTwisTorr",
    # Thermal
    "oasis_three": f"{_PKG}.thermal.oasis:OasisThreeChiller",
    "smc_hecr": f"{_PKG}.thermal.smc_hecr:SMCThermoChillerHECR",
    # Timing
    "mpsd_delay_generator": f"{_PKG}.timing.delay_generator:MPSDDelayGenerator",
}


def driver_class(key: str) -> type[Device]:
    """Import and return the class registered under ``key``."""
    try:
        target = DRIVERS[key]
    except KeyError:
        known = ", ".join(sorted(DRIVERS))
        raise KeyError(f"unknown driver {key!r}; known drivers: {known}") from None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def create_device(name: str, driver: str, **options: Any) -> Device:
    """Instantiate a device (without connecting it)."""
    return driver_class(driver)(name=name, **options)
