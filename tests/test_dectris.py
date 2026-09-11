import io
import json

import numpy as np
import pytest
import tifffile

from uedcontrol.devices.base import DeviceError
from uedcontrol.devices.cameras.dectris import DectrisCamera, decode_darray, encode_darray


class FakeResponse:
    def __init__(self, value=None, content=b"", status_code=200):
        self._value = value
        self.content = content
        self.status_code = status_code
        self.reason = "OK" if status_code == 200 else "Error"

    def json(self):
        return {"value": self._value}


class FakeSession:
    def __init__(self, image):
        self.image = image
        self.config = {
            "x_pixel_size": 7.5e-5,
            "count_time": 0.1,
            "frame_time": 0.1,
            "counting_mode": "normal",
            "trigger_mode": "ints",
            "pixel_mask": encode_darray(np.zeros((4, 6), dtype=np.uint32)),
        }
        self.commands = []

    def get(self, url, headers=None, timeout=None):
        if url.endswith("/images/monitor"):
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, self.image)
            return FakeResponse(content=buffer.getvalue())
        return FakeResponse(self.config[url.rsplit("/", 1)[1]])

    def put(self, url, data=None, headers=None, timeout=None):
        parameter = url.rsplit("/", 1)[1]
        if "/command/" in url:
            self.commands.append(parameter)
        else:
            self.config[parameter] = json.loads(data)["value"]
        return FakeResponse()


def test_darray_round_trip():
    array = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assert np.array_equal(decode_darray(encode_darray(array)), array)


def test_connect_exposure_and_grab_orientation():
    raw = np.arange(12, dtype=np.uint32).reshape(3, 4)
    session = FakeSession(raw)
    camera = DectrisCamera(session=session)
    camera.connect()
    assert session.config["mode"] == "enabled"  # monitor interface
    assert camera.pixel_size_um == pytest.approx(75.0)

    camera.set_exposure(250)
    assert session.config["count_time"] == pytest.approx(0.25)
    assert session.config["frame_time"] == pytest.approx(0.25)
    assert camera.exposure() == pytest.approx(250)

    image = camera.grab()
    assert session.commands[-3:] == ["arm", "trigger", "disarm"]
    # the orientation the original software displayed: fliplr(raw).T
    assert np.array_equal(image, np.fliplr(raw).T)


def test_mask_disk_maps_display_to_detector_pixels():
    session = FakeSession(np.zeros((4, 6), dtype=np.uint32))
    camera = DectrisCamera(session=session)
    camera.connect()
    camera.mask_disk(0, 0, 0.5)  # display pixel (row 0, column 0)
    mask = decode_darray(session.config["pixel_mask"])
    assert mask[0, 5] == 1 and mask.sum() == 1
    assert session.config["pixel_mask_applied"] is True
    camera.mask_disk(0, 0, 0.5, enabled=False)
    assert decode_darray(session.config["pixel_mask"]).sum() == 0


def test_http_errors_become_device_errors():
    session = FakeSession(np.zeros((2, 2)))
    session.put = lambda *args, **kwargs: FakeResponse(status_code=500)
    camera = DectrisCamera(session=session)
    with pytest.raises(DeviceError):
        camera.connect()
