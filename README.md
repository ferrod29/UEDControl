# uedcontrol

Control software for the ultrafast electron diffraction (UED) instrument: detectors, pump-probe delay
line, electron gun high voltage, magnetic lenses, laser steering, RF compression cavity, vacuum and
auxiliary equipment, with live beam diagnostics, pump-probe scans and image analysis.

It replaces and merges the two earlier code bases, **InstrumentControl** and **UEDControlSystem**, into
one installable Python package with a single application, one configuration file per set-up, and
tests that run without hardware.

## Quick start

```bash
# Python 3.10 or newer. Pick one Qt binding (PySide6 or PyQt5) and the hardware extras you need.
pip install -e ".[pyside6,hardware,dev]"

uedcontrol --simulate                     # the complete application with simulated instruments
uedcontrol --config config/local.yaml     # the real instrument
python -m uedcontrol --help               # same as the uedcontrol command
```

Without `--config`, the file named by the `UEDCONTROL_CONFIG` environment variable is used, so a lab
PC can start the right set-up from a plain desktop shortcut.

Optional extras, each needed only by the drivers that use it (a missing SDK only matters once you
connect that instrument):

| Extra          | Installs    | Needed for                                                          |
|----------------|-------------|---------------------------------------------------------------------|
| `serial`       | pyserial    | every serial / USB-COM instrument                                   |
| `usb`          | libusb1     | Mini-Circuits power sensors and attenuator, DrXWorks RF amplifier   |
| `dectris`      | requests    | DECTRIS QUADRO                                                      |
| `pi`           | PIPython    | PI Mercury delay stage                                              |
| `basler`       | pypylon     | Basler cameras                                                      |
| `pco`          | pco         | PCO cameras                                                         |
| `micromanager` | pymmcore    | any Micro-Manager camera (e.g. Andor Zyla)                          |
| `hardware`     | the four above | everything except the camera SDKs                  |

The attocube ECC100 uses the vendor DLL; give its path with `dll_path`.

## Configuration

A set-up is described by one YAML file. Copy [config/example_instrument.yaml](config/example_instrument.yaml)
(which lists every supported instrument with the ports and calibrations of the original code) to
`config/local.yaml` (ignored by git), delete what your set-up does not have and adjust the ports:

```yaml
name: UED
data_dir: D:/UEDData            # every scan gets its own sub-folder here

devices:
  detector:
    driver: dectris_quadro      # key from uedcontrol/devices/registry.py
    label: DECTRIS QUADRO       # label, group, autoconnect, panel: GUI options
    group: Cameras
    host: 10.6.101.55           # everything else goes to the driver
  delay_stage:
    driver: pi_mercury_c663
    group: Delay line
    serial_number: "0115500008"
    autoconnect: true

delay_line:
  stage: delay_stage            # the device used for pump-probe scans
  t0_position: 153.0            # mm; "Set T0 here" in the GUI changes it for the session
  passes: 2                     # optical passes over the stage (retro-reflector: 2)
```

The file is validated when the program starts: unknown drivers, misspelled keys, wrong driver options
and devices used in the wrong role (e.g. a camera as the delay stage) are reported with a clear
message instead of failing later.

## The application

* **Live view** - detector image with histogram, projections, circular or annular ROI, crosshair and
  ROI tracking; frame averaging, background/reference capture and subtraction, software beam block
  and (DECTRIS) detector pixel masking; beam size (RMS, FWHM, 1/e²), counts, electrons per pulse,
  bunch charge and beam current.
* **Scan** - pump-probe delay scans (range plus extra segments such as `-1:1:0.1; 20:100:5`, or a list
  loaded from a file), frames per point, several or unlimited runs with alternating direction,
  settling time, and a time-series mode without stage motion (the former "continuous" mode). Delays
  outside the stage range are refused before anything moves.
* **Pump-probe** - ROI profile map versus delay (optionally relative to negative delays), ROI counts
  and RMS contrast, updated after every point.
* **Instruments** - one tab per group of the configuration: HV supply (ramps, interlock, ripple,
  breakdown detection), multi-channel lens supplies (with coil temperature), delay stage (delay and
  T0), piezo mirrors, RF power interlock, and a generic panel for everything else (vacuum, chillers,
  power meters, RF parts...) with time plots and CSV export.
* **Image analysis** (Tools menu, or "Send to analysis") - Gaussian/Lorentzian beam-profile fits,
  n-fold symmetrisation, azimuthal averages versus radius or q, ring detection and export.
* **Log** - everything is logged to the log dock, the console and `<log_dir>/uedcontrol.log`
  (rotating).

Every instrument panel talks to its instrument from its own thread, and acquisition runs in another,
so a slow or unresponsive instrument never freezes the window.

## Data files

Each scan is saved in its own folder, `<data_dir>/<date>_<time>_<name>/` (the date prefix can be
switched off); an existing folder is never overwritten, a suffix `_001`, `_002`... is added instead.

* `<name>.h5` - metadata (sample, notes, camera, exposure, T0, configuration, software version...),
  delays, background and reference, every averaged image with its delay and ROI signals, and the
  averaged results (`results/roi_total`, `results/roi_rms`, `results/roi_profile`, `results/points`).
* `tsteps.txt`, `pp_avgi_data.txt`, `pp_rmsc_data.txt`, `pump_probe_image.txt` - the text files of
  the original software, so existing analysis scripts keep working.
* optionally one TIFF (float32, values preserved) and/or PNG preview per point.

Writing happens on a background thread, so the acquisition never waits for the disk.

## Scripting

Everything except the `gui` package works without Qt, e.g. in a notebook:

```python
import threading
from uedcontrol.acquisition.delay import delays_from_range
from uedcontrol.acquisition.scan import PumpProbeScan, ScanSettings
from uedcontrol.acquisition.storage import ScanWriter
from uedcontrol.config import load_config
from uedcontrol.instrument import Instrument

with Instrument.from_config(load_config("config/local.yaml")) as ued:
    ued.connect("detector", "delay_stage")
    camera = ued.cameras()["detector"]
    settings = ScanSettings(delays_from_range(-5, 20, 0.5), frames_per_point=10, runs=3)
    writer = ScanWriter("D:/UEDData", "gold_film", {"sample": "Au 20 nm"})
    camera.start_acquisition()          # required by streaming cameras (Basler, PCO, Micro-Manager)
    try:
        result = PumpProbeScan(camera, ued.delay_line, settings, writer=writer).run(threading.Event())
    finally:
        camera.stop_acquisition()
    print(result.mean_total)
```

Leaving the `with` block disconnects every instrument, even after an error.

## Supported instruments

| Driver key                  | Instrument                                | Interface            | Connection         |
|-----------------------------|-------------------------------------------|----------------------|--------------------|
| `dectris_quadro`            | DECTRIS QUADRO (EIGER family)             | Camera               | SIMPLON REST API   |
| `basler`                    | Basler cameras                            | Camera               | pypylon            |
| `pco`                       | PCO cameras                               | Camera               | pco                |
| `micromanager`              | any Micro-Manager camera                  | Camera               | pymmcore           |
| `pi_mercury_c663`           | PI Mercury C-663 + M-531 (delay line)     | Positioner           | PIPython           |
| `newport_smc100`            | Newport SMC100CC                          | Positioner           | serial             |
| `thorlabs_kdc101`           | Thorlabs KDC101 K-Cube (wave plate)       | Positioner           | serial (APT)       |
| `attocube_ecc100`           | attocube ECC100, one axis                 | Positioner           | vendor DLL         |
| `newport_agilis_uc8`        | Newport Agilis AG-UC8 mirror mounts       | PiezoMirrorController| serial             |
| `matsusada_cohvu32`         | Matsusada HV supply via CO-HVU32          | HighVoltageSupply    | serial (RS-485)    |
| `heinzinger_pnchp100`       | Heinzinger PNChp 100-1 (MPSD control box) | HighVoltageSupply    | serial             |
| `rs_hmp4040`, `rs_nge100`   | Rohde & Schwarz HMP4040, NGE100           | PowerSupply          | serial (SCPI)      |
| `delta_sm70ar24`            | Delta Elektronika SM 70-AR-24             | PowerSupply          | serial             |
| `minicircuits_power_sensor` | Mini-Circuits PWR-4GHS                    | RFPowerSensor        | USB (libusb1)      |
| `minicircuits_attenuator`   | Mini-Circuits RCDAT-6G-120H               | RFAttenuator         | USB (libusb1)      |
| `drxworks_statera100`       | DrXWorks Statera 100 RF amplifier         | RFAmplifier          | USB (libusb1)      |
| `newport_1935c`             | Newport 1935-C optical power meter        | generic              | serial             |
| `edwards_tic`               | Edwards TIC gauges and turbo pump         | generic              | serial             |
| `agilent_twistorr`          | Agilent TwisTorr 304/305 FS turbo pump    | generic              | serial             |
| `oasis_three`               | Solid State Cooling Oasis Three chiller   | generic              | serial             |
| `smc_hecr`                  | SMC Thermo-con HECR chiller ⚠             | generic              | serial             |
| `mpsd_delay_generator`      | MPSD delay/trigger generator ⚠            | generic              | serial             |
| `simulated_*`               | simulated versions of the main categories | all                  | none               |

⚠ The original code of these two drivers was unfinished, so their protocol could not be confirmed.
Check them against the instrument manual before relying on them; see the notes at the top of
[smc_hecr.py](src/uedcontrol/devices/thermal/smc_hecr.py) and
[delay_generator.py](src/uedcontrol/devices/timing/delay_generator.py).

## Architecture

```
src/uedcontrol/
├── config.py          YAML configuration -> dataclasses, with validation
├── instrument.py      all devices + delay line; entry point for scripts
├── constants.py       physical constants (CODATA 2018)
├── devices/           drivers
│   ├── base.py        Device lifecycle, locking, Reading / Setting / Action
│   ├── interfaces.py  Camera, Positioner, PowerSupply, HighVoltageSupply, RF..., PiezoMirrorController
│   ├── transport.py   serial transport and SerialInstrument (fake transport for tests)
│   ├── usb_hid.py     interrupt-packet USB devices
│   ├── registry.py    driver keys -> classes, imported lazily
│   ├── simulated.py   simulated instruments
│   └── cameras/ motion/ power/ rf/ sensors/ thermal/ timing/
├── analysis/          pure numpy/scipy: ROI statistics, fits, diffraction, pump-probe averaging
├── acquisition/       delay line, scan engine, HDF5/text storage, safety logic (interlock, ramps)
└── gui/               Qt application (PySide6, PyQt5 or PyQt6 through pyqtgraph)
    ├── app.py         command line
    ├── main_window.py
    ├── acquisition_panel.py  scan_panel.py  pumpprobe_view.py  analysis_window.py
    ├── workers.py     device threads and the acquisition thread
    ├── panels/        instrument panels, chosen by interface
    └── widgets/       image view, time plots, small helpers
```

Dependencies only point downwards: `gui` uses everything, `acquisition` uses `analysis` and
`devices`, and `analysis` depends on nothing but numpy and scipy. The GUI and the acquisition code
program against the interfaces in `devices/interfaces.py`, never against a concrete driver.

Every driver method holds the device's re-entrant lock for a complete command sequence, so a panel
can poll an instrument while a scan commands it from the acquisition thread.

## Adding an instrument

1. Write a driver that subclasses the matching interface (or `Device` for a monitoring-only
   instrument) and implements `_connect`, `_disconnect` and the abstract methods. Serial instruments
   subclass `SerialInstrument` and get framing, timeouts and locking for free.
2. Declare what the generic panel should show: `read_status()` (readings), `settings()` (editable
   parameters) and `actions()` (buttons).
3. Register it in `devices/registry.py`, add an example to `config/example_instrument.yaml`, and a
   test using `FakeTransport` that checks the bytes it sends and parses.

The GUI then picks the right panel automatically; no GUI code needs to change.

## What changed compared with the original software

* One package, one application and one configuration file instead of per-instrument scripts with
  hard-coded ports, paths and calibrations.
* Hardware is never accessed from the GUI thread, and all access to an instrument is serialised.
* **Physics fixes:** the scattering vector now uses q = 4π sin(θ)/λ with 2θ = arctan(r/L) (the old code
  used sin(2θ), about twice too large); "Set T0" now really moves time zero (the old code kept using
  the hard-coded 153 mm); the delay/position conversion uses the speed of light and the number of
  passes instead of rounded constants.
* ROI statistics are vectorised (the old code looped over every pixel in Python for every frame).
* Scans are stored in HDF5 with full metadata, never overwrite earlier data, and keep the old text
  files for compatibility.
* Out-of-range delays are refused before the scan starts, and the RF interlock, HV ramps and
  breakdown detection are covered by tests.

## Development

```bash
pytest                     # ~100 tests, no hardware needed; the GUI test runs offscreen
ruff check src tests
```

## Licence

MIT, see [LICENSE](LICENSE).
