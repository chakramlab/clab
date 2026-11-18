import numpy as np
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def snail_mode_cross_kerr(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="snail_mode_cross_kerr",
    average_exponent=5,
    time_swp=LinearSweepParameter(uid="time_swp", start=0, stop=20e-6, count=101),
    acquisition_type=AcquisitionType.INTEGRATION,
    sw_detuning_freq=0,
    n_echoes=0,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    averaging_mode=AveragingMode.CYCLIC,
    alice_or_bob="alice",
    load_buffer=None,
    load_storage=None,
    storage_swap_channel_name=None,
    lo_settings=None,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    if lo_settings is None:
        lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]

    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel

    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob = qubit_params_module.sb_pulses["bob"]["f0g1"]
    snail_X90 = qubit_params_module.snail_01_X90
    snail_transmon_swap = qubit_params_module.bs_transmon_snail_eg_ge
    bs = (
        qubit_params_module.sb_pulses[alice_or_bob][f"bs{load_storage}"]
        if load_storage is not None
        else None
    )

    if n_echoes > 0:
        snail_X180 = qubit_params_module.snail_01_X180

    transitions = [f"f{i}g{i+1}" for i in range(1)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    snail_01_freq = qubit_parameters["q0"]["snail_01_freq"]
    snail_transmon_swap_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]
    if (load_buffer is not None) and (load_storage is not None):
        storage_swap_freq = qubit_parameters["q0"][f"bs_{load_buffer}_freqs"][
            load_storage
        ]

    if storage_swap_channel_name == "SG3":
        storage_swap_lo_name = "SG2_LO"
    elif storage_swap_channel_name == "SG4":
        storage_swap_lo_name = "SG4_LO"
    else:
        storage_swap_lo_name = None

    for freq, lo_name in zip(
        [snail_01_freq, snail_transmon_swap_freq], ["SG2_LO", "SG4_LO"]
    ):
        lo_settings["q0"][serial_num][lo_name] = max(1e9, round(freq / 200e6) * 200e6)
    lo_snail = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_snail_transmon_swap = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_storage_swap = (
        lo_settings["q0"][serial_num][storage_swap_lo_name]
        if storage_swap_lo_name is not None
        else None
    )

    signals = [
        ExperimentSignal("qb_drive"),
        ExperimentSignal("qb_ef_drive"),
        *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
        ExperimentSignal("snail_drive"),
        ExperimentSignal("snail_transmon_swap"),
        ExperimentSignal("measure"),
        ExperimentSignal("acquire"),
    ]
    if load_storage is not None:
        signals.append(ExperimentSignal("bs"))
    exp = Experiment(
        uid=exp_id,
        signals=signals,
    )

    phase_swp = 2.0 * np.pi * time_swp * sw_detuning_freq
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
        averaging_mode=AveragingMode(averaging_mode),
    ):
        with exp.sweep(
            uid="time_sweep",
            parameter=time_swp,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):

            if load_buffer is not None:

                with exp.section(uid="ge_excitation"):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(
                    uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
                ):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                with exp.section(uid="sb_transition_f0g1", play_after="ef_excitation"):
                    exp.play(
                        signal=sb_drive_lines["f0g1"],
                        pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                    )
                play_after = "sb_transition_f0g1"
            else:
                play_after = None

            if load_storage is not None:
                with exp.section(uid="storage_swap", play_after=play_after):
                    exp.play(
                        signal="bs",
                        pulse=bs,
                    )
                play_after = "storage_swap"

            with exp.section(uid="snail_01", play_after=play_after):
                exp.play(signal="snail_drive", pulse=snail_X90, phase=0)
                exp.delay(signal="snail_drive", time=time_swp)
                exp.play(signal="snail_drive", pulse=snail_X90, phase=phase_swp)

            with exp.section(uid="snail_transmon_swap", play_after="snail_01"):
                exp.play(
                    signal="snail_transmon_swap",
                    pulse=snail_transmon_swap,
                )
            play_after = "snail_transmon_swap"

            with exp.section(uid="readout", play_after=play_after):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=(
                        5 * qubit_parameters["q0"]["snail_01_t1"]
                        if load_buffer is None and load_storage is None
                        else qubit_parameters["q0"]["cavity_reset_delay"]
                    ),
                    acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                )

    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
        rotate_ro=rotate_ro,
        thresholds=thresholds,
    )

    sig_freq_map[serial_num]["SG3"]["snail_drive"] = {
        "frequency": snail_01_freq - lo_snail,
        "range": qubit_parameters["q0"]["snail_01_dBm_range"],
    }
    if (load_buffer is not None) and (load_storage is not None):
        sig_freq_map[serial_num][storage_swap_channel_name]["bs"] = {
            "frequency": storage_swap_freq - lo_storage_swap,
            "range": qubit_parameters["q0"][f"bs_{load_buffer}_dBm_ranges"][
                load_storage
            ],
        }
    sig_freq_map[serial_num]["SG4"]["snail_transmon_swap"] = {
        "frequency": snail_transmon_swap_freq - lo_snail_transmon_swap,
        "range": qubit_parameters["q0"]["bs_transmon_snail_eg_ge_dBm_range"],
    }

    exp_calibration, map_q0 = default_signal_map_and_calibration(
        sig_freq_map,
        {
            "device_setup": device_setup,
            "lo_settings": lo_settings,
            "qubit_parameters": qubit_parameters,
        },
    )
    exp.set_calibration(exp_calibration)
    exp.set_signal_map(map_q0)

    return exp, lo_snail
