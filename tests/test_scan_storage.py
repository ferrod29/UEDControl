import threading

import h5py
import numpy as np
import pytest

from uedcontrol.acquisition.delay import DelayLine, delays_from_range
from uedcontrol.acquisition.scan import PumpProbeScan, ScanSettings, average_frames
from uedcontrol.acquisition.storage import ScanWriter, load_image, save_image, to_uint16
from uedcontrol.analysis.beam import CircularROI
from uedcontrol.devices.base import DeviceError
from uedcontrol.devices.simulated import SimulatedCamera, SimulatedStage


@pytest.fixture
def setup():
    camera = SimulatedCamera(shape=(128, 128), realtime=False, seed=1)
    camera.connect()
    stage = SimulatedStage(realtime=False, position=150.0, limits=(0.0, 300.0))
    stage.connect()
    delay_line = DelayLine(stage, t0_position=150.0)
    camera.delay_source = delay_line.current_delay
    return camera, delay_line


def test_scan_runs_bidirectionally_and_stores_everything(tmp_path, setup):
    camera, delay_line = setup
    delays = delays_from_range(-2, 10, 2)
    settings = ScanSettings(delays, frames_per_point=2, runs=2, roi=CircularROI(64, 64, 50))
    writer = ScanWriter(tmp_path, "gold", {"sample": "Au", "voltage_kv": 100.0, "notes": None}, tiff=True, png=True)
    points = []
    scan = PumpProbeScan(camera, delay_line, settings, writer=writer, on_point=lambda p, acc: points.append(p))
    accumulator = scan.run(threading.Event())

    assert len(points) == 14
    assert [p.index for p in points] == [0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0]
    assert accumulator.counts.tolist() == [2] * 7
    assert accumulator.mean_total[-1] < accumulator.mean_total[0]  # simulated photo-induced drop

    scan_dir = writer.scan_dir
    with h5py.File(scan_dir / "gold.h5") as handle:
        assert handle.attrs["sample"] == "Au"
        assert handle.attrs["notes"] == ""
        assert len(handle["images"]) == 14
        assert handle["images/00001"].attrs["delay_ps"] == -2.0
        assert handle["results/roi_total"].shape == (7,)
        assert handle["results/roi_profile"].shape[0] == 7
        assert len(handle["results/points"]) == 14
        assert "end_time" in handle.attrs
    for name in ("tsteps.txt", "pp_avgi_data.txt", "pp_rmsc_data.txt", "pump_probe_image.txt"):
        assert (scan_dir / name).exists()
    table = np.loadtxt(scan_dir / "pp_avgi_data.txt")
    assert table.shape == (7, 3) and np.allclose(table[:, 1], delays)
    assert (scan_dir / "001_delay_-2ps.tiff").exists()
    assert (scan_dir / "014_delay_-2ps.png").exists()


def test_scan_can_be_stopped(tmp_path, setup):
    camera, delay_line = setup
    stop = threading.Event()
    points = []

    def on_point(point, accumulator):
        points.append(point)
        if len(points) == 3:
            stop.set()

    writer = ScanWriter(tmp_path, "stopped")
    PumpProbeScan(camera, delay_line, ScanSettings(np.arange(10.0), runs=0), writer=writer,
                  on_point=on_point).run(stop)
    assert len(points) == 3
    with h5py.File(writer.scan_dir / "stopped.h5") as handle:
        assert len(handle["images"]) == 3


def test_out_of_range_delays_fail_before_anything_is_written(tmp_path, setup):
    camera, delay_line = setup  # stage 0..300 mm, T0 at 150 mm -> about ±1000 ps
    writer = ScanWriter(tmp_path, "too_far")
    scan = PumpProbeScan(camera, delay_line, ScanSettings(np.array([0.0, 5000.0])), writer=writer)
    with pytest.raises(DeviceError, match="outside the stage range"):
        scan.run(threading.Event())
    assert delay_line.stage.position() == pytest.approx(150.0)  # the stage never moved
    assert not writer.scan_dir.exists()
    writer._executor.shutdown()


def test_time_series_without_delay_line(setup):
    camera, _ = setup
    settings = ScanSettings(np.arange(4.0), background=np.zeros((128, 128)))
    accumulator = PumpProbeScan(camera, None, settings).run(threading.Event())
    assert accumulator.completed_points == 4


def test_scan_directories_are_never_overwritten(tmp_path):
    first = ScanWriter(tmp_path, "run")
    second = ScanWriter(tmp_path, "run")
    first._executor.shutdown()
    first.scan_dir.mkdir()
    third = ScanWriter(tmp_path, "run")
    assert first.scan_dir.name == "run" and third.scan_dir.name == "run_001"
    second._executor.shutdown()
    third._executor.shutdown()


def test_average_frames(setup):
    camera, _ = setup
    assert average_frames(camera, 3).shape == (128, 128)


def test_scan_settings_validation():
    with pytest.raises(ValueError):
        ScanSettings(np.array([]))
    with pytest.raises(ValueError):
        ScanSettings(np.arange(3.0), frames_per_point=0)
    assert ScanSettings(np.arange(3.0), runs=0).total_points is None


def test_image_round_trip(tmp_path):
    image = np.random.default_rng(0).random((20, 30)) * 1000
    save_image(tmp_path / "a.tiff", image)
    assert np.allclose(load_image(tmp_path / "a.tiff"), image, rtol=1e-6)
    save_image(tmp_path / "a.png", image)
    assert load_image(tmp_path / "a.png").shape == (20, 30)
    save_image(tmp_path / "a.npy", image)
    assert np.array_equal(load_image(tmp_path / "a.npy"), image)
    with pytest.raises(ValueError):
        save_image(tmp_path / "a.bmp", image)
    assert to_uint16(np.ones((2, 2))).max() == 0
