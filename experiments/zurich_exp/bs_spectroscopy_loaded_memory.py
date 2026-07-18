from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def bs_spectroscopy_loaded_memory(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="loaded_memory_bs_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=-700e6, stop=700e6, count=6
    ),
    bs_length=None,
    bs_range=None,
    bs_ramp=None,
    bs_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    buffer_spectator="alice",
    storage_spectator=1,
    buffer_target="bob",
    storage_target=3,
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    sb_delay=0,
    load_spectator=True,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180

    sb_f0g1_spectator = qubit_params_module.sb_pulses[buffer_spectator]["f0g1"]
    sb_f0g1_target = qubit_params_module.sb_pulses[buffer_target]["f0g1"]

    # bs pulses
    bs_spectator = qubit_params_module.sb_pulses[buffer_spectator][
        f"bs{storage_spectator}"
    ]
    bs_target = qubit_params_module.sb_pulses[buffer_target][f"bs{storage_target}"]

    if bs_length is None:
        bs_length = bs_target.length
    if bs_amplitude is not None:
        bs_target.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{buffer_target}_dBm_ranges"][
            storage_target
        ]

    bs_range_spectator = qubit_parameters["q0"][f"bs_{buffer_spectator}_dBm_ranges"][
        storage_spectator
    ]
    bs_freq_spectator = qubit_parameters["q0"][f"bs_{buffer_spectator}_freqs"][
        storage_spectator
    ]

    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9

    requires_shift = False
    if freq_swp.start < lo - lo_range or freq_swp.stop > lo + lo_range:
        requires_shift = True
    if bs_freq_spectator < lo - lo_range or bs_freq_spectator > lo + lo_range:
        requires_shift = True

    if requires_shift:
        new_lo = ((freq_swp.start + freq_swp.stop) / 2 + bs_freq_spectator) / 2
        step = 200e6
        new_lo = round(new_lo / step) * step
        if new_lo < 1e9:
            new_lo = 0
        lo_settings["q0"][serial_num]["SG4_LO"] = new_lo
        lo = new_lo
        print(f"Warning: LO frequency changed to {new_lo/1e9} GHz")

    freq_swp.start -= lo
    freq_swp.stop -= lo

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines_spectator = {}
    sb_drive_lines_target = {}
    for transition in transitions:
        sb_drive_lines_spectator[transition] = (
            f"sb_drive_{buffer_spectator}_{transition}"
        )
        sb_drive_lines_target[transition] = f"sb_drive_{buffer_target}_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs_spectator"),
            ExperimentSignal("bs_target"),
            *[ExperimentSignal(sb_drive_lines_spectator[_]) for _ in transitions],
            *[ExperimentSignal(sb_drive_lines_target[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):
            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            if load_spectator:
                with exp.section(
                    uid="sb_load_spectator",
                    play_after="ef_excitation",
                    on_system_grid=True,
                ):
                    exp.play(
                        signal=sb_drive_lines_spectator["f0g1"],
                        pulse=sb_f0g1_spectator,
                    )
                    exp.delay(signal=sb_drive_lines_spectator["f0g1"], time=sb_delay)

                with exp.section(
                    uid="bs_park_spectator",
                    play_after="sb_load_spectator",
                    on_system_grid=True,
                ):
                    exp.play(signal="bs_spectator", pulse=bs_spectator)

                with exp.section(
                    uid="ge_excitation_reprep",
                    play_after="bs_park_spectator",
                    on_system_grid=True,
                ):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(
                    uid="ef_excitation_reprep",
                    play_after="ge_excitation_reprep",
                    on_system_grid=True,
                ):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(
                uid="sb_load_target",
                play_after=(
                    "ef_excitation_reprep" if load_spectator else "ef_excitation"
                ),
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines_target["f0g1"],
                    pulse=sb_f0g1_target,
                )
                exp.delay(signal=sb_drive_lines_target["f0g1"], time=sb_delay)

            with exp.section(
                uid="bs_target", play_after="sb_load_target", on_system_grid=True
            ):
                exp.play(
                    signal="bs_target",
                    pulse=bs_target,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            with exp.section(
                uid="sb_unload_target", play_after="bs_target", on_system_grid=True
            ):
                exp.delay(signal=sb_drive_lines_target["f0g1"], time=sb_delay)
                exp.play(
                    signal=sb_drive_lines_target["f0g1"],
                    pulse=sb_f0g1_target,
                )
                exp.delay(signal=sb_drive_lines_target["f0g1"], time=sb_delay)

            with exp.section(
                uid="ef_excitation_2",
                play_after="sb_unload_target",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(
                uid="readout", play_after="ef_excitation_2", on_system_grid=True
            ):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
                    acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
        rotate_ro=rotate_ro,
        thresholds=thresholds,
    )

    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs_spectator"] = {}
    sig_freq_map[serial_num][ch]["bs_spectator"]["frequency"] = bs_freq_spectator - lo
    sig_freq_map[serial_num][ch]["bs_spectator"]["range"] = bs_range_spectator

    sig_freq_map[serial_num][ch]["bs_target"] = {}
    sig_freq_map[serial_num][ch]["bs_target"]["frequency"] = freq_swp
    sig_freq_map[serial_num][ch]["bs_target"]["range"] = bs_range
    sig_freq_map[serial_num][ch]["bs_target"]["length"] = bs_length
    sig_freq_map[serial_num][ch]["bs_target"]["amplitude"] = bs_amplitude

    print("bs_spectator map:", sig_freq_map[serial_num][ch]["bs_spectator"])
    print("bs_target map:", sig_freq_map[serial_num][ch]["bs_target"])

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

    return exp, lo
