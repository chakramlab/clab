from __future__ import annotations

import importlib

import pytest

pytest.importorskip("laboneq")

try:
    AcquisitionType = importlib.import_module("laboneq.dsl.enums").AcquisitionType
    create_default_map_and_calibration = importlib.import_module(
        "zurich_exp_rework2.calib_settings"
    ).create_default_map_and_calibration
except ImportError:
    pytest.skip("LabOne Q calibration helpers unavailable", allow_module_level=True)


class _StubAcquireLoop:
    def __init__(self, acquisition_type):
        self.acquisition_type = acquisition_type


class _StubExperiment:
    def __init__(self, signals, acquisition_type=AcquisitionType.SPECTROSCOPY):
        self.signals = set(signals)
        self._loop = _StubAcquireLoop(acquisition_type)

    def get_rt_acquire_loop(self):
        return self._loop


@pytest.fixture
def serial_num():
    return "DEV1234"


@pytest.fixture
def lo_settings(serial_num):
    return {
        "q0": {
            serial_num: {
                "QA0_LO": 7.6e9,
                "SG0_LO": 4.0e9,
                "SG2_LO": 4.5e9,
                "SG4_LO": 1.2e9,
            }
        }
    }


def _base_qubit_params():
    return {
        "qb_drive_dBm_range": 0,
        "qb_drive_resolved_dBm_range": 0,
        "qb_freq": 4.2e9,
        "qb_resolved_freq": 4.19e9,
        "qb_ef_freq": 4.1e9,
        "sb_alice_dBm_range": -10,
        "sb_bob_dBm_range": -8,
        "cav_alice_freq": 7.1e9,
        "cav_alice_dBm_range": -20,
        "cav_bob_freq": 7.2e9,
        "cav_bob_dBm_range": -18,
        "ro_freq": 7.7e9,
        "ro_drive_dBm_range": 5,
        "ro_acq_dBm_range": 0,
    }


def test_arrays_drive_correct_sideband_and_bs_entries(serial_num, lo_settings):
    qparams = _base_qubit_params()
    qparams.update(
        {
            "sb_alice_freqs": [4.05e9, 4.06e9, 4.07e9],
            "sb_alice_flat_lens": [2.0e-6, 2.1e-6, 2.2e-6],
            "sb_alice_ramp_lens": [20e-9, 20e-9, 20e-9],
            "bs_alice_freqs": [1.5e9],
            "bs_alice_flat_lens": [21e-6],
            "bs_alice_ramp_lens": [200e-9],
            "bs_alice_dBm_ranges": [3],
        }
    )
    qubit_parameters = {"q0": qparams}

    exp = _StubExperiment(
        signals={
            "sb_drive_alice_f1g2",
            "sb_drive_alice_bs0",
        }
    )

    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
    )

    los = lo_settings["q0"][serial_num]

    sideband = sig_freq_map[serial_num]["SG1"]["sb_drive_alice_f1g2"]
    assert pytest.approx(sideband["frequency"]) == qparams["sb_alice_freqs"][1] - los["SG0_LO"]
    assert sideband["length"] == pytest.approx(qparams["sb_alice_flat_lens"][1])

    bs_entry = sig_freq_map[serial_num]["SG4"]["sb_drive_alice_bs0"]
    assert pytest.approx(bs_entry["frequency"]) == qparams["bs_alice_freqs"][0] - los["SG4_LO"]
    assert bs_entry["range"] == qparams["bs_alice_dBm_ranges"][0]
    assert bs_entry["length"] == pytest.approx(qparams["bs_alice_flat_lens"][0])


def test_legacy_keys_still_supported(serial_num, lo_settings):
    qparams = _base_qubit_params()
    qparams.update(
        {
            "sb_f0g1_alice_freq": 4.04e9,
            "sb_f0g1_alice_flat_len": 2.5e-6,
            "sb_f0g1_alice_ramp_len": 200e-9,
            "bs0_alice_freq": 1.45e9,
            "bs0_alice_flat_len": 25e-6,
            "bs0_alice_ramp_len": 300e-9,
            "bs0_alice_dBm_range": 6,
        }
    )
    qubit_parameters = {"q0": qparams}

    exp = _StubExperiment(
        signals={
            "sb_drive_alice_f0g1",
            "sb_drive_alice_bs0",
        },
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    )

    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
    )

    los = lo_settings["q0"][serial_num]

    sideband = sig_freq_map[serial_num]["SG1"]["sb_drive_alice_f0g1"]
    assert pytest.approx(sideband["frequency"]) == qparams["sb_f0g1_alice_freq"] - los["SG0_LO"]
    assert sideband["length"] == pytest.approx(qparams["sb_f0g1_alice_flat_len"])

    bs_entry = sig_freq_map[serial_num]["SG4"]["sb_drive_alice_bs0"]
    assert pytest.approx(bs_entry["frequency"]) == qparams["bs0_alice_freq"] - los["SG4_LO"]
    assert bs_entry["range"] == qparams["bs0_alice_dBm_range"]
    assert bs_entry["length"] == pytest.approx(qparams["bs0_alice_flat_len"])
