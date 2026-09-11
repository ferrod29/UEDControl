from itertools import islice

import numpy as np
import pytest

from uedcontrol.acquisition.delay import (
    DelayLine,
    delays_from_file,
    delays_from_range,
    delays_from_segments,
    parse_segments,
)
from uedcontrol.analysis.pumpprobe import PumpProbeAccumulator, scan_order
from uedcontrol.devices.base import DeviceError
from uedcontrol.devices.simulated import SimulatedStage


def test_scan_order_bidirectional_and_forward():
    assert list(scan_order(3, 2)) == [(1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert list(scan_order(3, 2, bidirectional=False)) == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert list(islice(scan_order(2, None), 5)) == [(1, 0), (1, 1), (2, 1), (2, 0), (3, 0)]
    with pytest.raises(ValueError):
        list(scan_order(0, 1))


def test_accumulator_means_and_resampling():
    acc = PumpProbeAccumulator([0.0, 1.0, 2.0])
    acc.add(0, 1.0, 10.0, np.ones(4))
    acc.add(0, 3.0, 20.0, 3 * np.ones(4))
    acc.add(2, 5.0, 30.0, np.ones(8))  # different ROI size -> resampled to 4 samples
    assert acc.mean_total[0] == 2.0 and np.isnan(acc.mean_total[1]) and acc.mean_total[2] == 5.0
    assert acc.mean_rms[0] == 15.0
    assert acc.mean_profile.shape == (3, 4)
    assert np.allclose(acc.mean_profile[0], 2.0)
    assert acc.completed_points == 3


def make_delay_line(**kwargs):
    stage = SimulatedStage(realtime=False, position=153.0, limits=(0.0, 306.0))
    stage.connect()
    return DelayLine(stage, t0_position=153.0, **kwargs)


def test_delay_line_conversions():
    line = make_delay_line()
    assert line.mm_per_ps == pytest.approx(0.1499, abs=1e-4)
    assert line.position_for(10) == pytest.approx(153 + 1.49896, abs=1e-4)
    assert line.delay_for(line.position_for(-3.7)) == pytest.approx(-3.7)
    low, high = line.delay_limits()
    assert low == pytest.approx(-1020.7, abs=0.5) and high == pytest.approx(1020.7, abs=0.5)
    reverse = make_delay_line(direction=-1)
    assert reverse.position_for(10) < 153


def test_delay_line_moves_and_t0():
    line = make_delay_line()
    line.move_to_delay(5.0)
    assert line.current_delay() == pytest.approx(5.0)
    assert line.set_t0_here() == pytest.approx(line.stage.position())
    assert line.current_delay() == pytest.approx(0.0)
    with pytest.raises(DeviceError):
        line.move_to_delay(5000)


def test_delays_from_range():
    assert np.allclose(delays_from_range(-1, 1, 0.5), [-1, -0.5, 0, 0.5, 1])
    assert np.allclose(delays_from_range(0, 1, 0.3), [0, 0.3, 0.6, 0.9])
    assert np.allclose(delays_from_range(2, 0, -1), [2, 1, 0])
    with pytest.raises(ValueError):
        delays_from_range(0, 1, 0)
    with pytest.raises(ValueError):
        delays_from_range(0, 1, -0.1)


def test_delays_from_segments_and_file(tmp_path):
    assert np.allclose(delays_from_segments([(-1, 0, 0.5), (0, 2, 1)]), [-1, -0.5, 0, 1, 2])
    single = tmp_path / "single.txt"
    single.write_text("-1\n0\n2.5\n")
    assert np.allclose(delays_from_file(single), [-1, 0, 2.5])
    double = tmp_path / "double.txt"
    double.write_text("0 10\n1 20\n")
    assert np.allclose(delays_from_file(double), [0, 1])


def test_parse_segments():
    assert parse_segments("-1:1:0.1; 20:100:5\n 200:300:50 ;") == [(-1, 1, 0.1), (20, 100, 5), (200, 300, 50)]
    assert parse_segments("  ") == []
    with pytest.raises(ValueError, match="start:stop:step"):
        parse_segments("0:1")
    with pytest.raises(ValueError, match="not a number"):
        parse_segments("0:1:0,5")  # decimal comma is rejected, never split

