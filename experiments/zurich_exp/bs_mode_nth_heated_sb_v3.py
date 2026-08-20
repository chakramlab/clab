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


def bs_mode_nth_heated_sb_v3(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_mode_nth_heated_sb_v3",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    bs_heating_swp_param=LinearSweepParameter(
        uid="swp_param_off_resonant", start=1e-9, stop=10e-6, count=6
    ),
    swp_param=LinearSweepParameter(uid="swp_param", start=1e-9, stop=10e-6, count=6),
    prepare_f=True,
    bs_heating_freq=None,
    bs_heating_amplitude=None,
    bs_heating_range=None,
    bs_swap_freq=None,
    bs_swap_range=None,
    bs_swap_length=None,
    bs_swap_amplitude=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob_heating="alice",
    alice_or_bob_swap="bob",
    max_fock_state=1,
    rotate_ro=False,
    thresholds=None,
    swap_storage_mode=0,
    heated_storage_mode=0,
    chunk_count=None,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob = qubit_params_module.sb_pulses["bob"]["f0g1"]

    # bs pulse (actual resonant one)
    bs_swap = qubit_params_module.sb_pulses[alice_or_bob_swap][f"bs{swap_storage_mode}"]
    if bs_swap_length is None:
        bs_swap_length = bs_swap.length
    if bs_swap_amplitude is not None:
        bs_swap.amplitude = 1
    if bs_swap_range is None:
        bs_swap_range = qubit_parameters["q0"][f"bs_{alice_or_bob_swap}_dBm_ranges"][
            swap_storage_mode
        ]
    if bs_swap_freq is None:
        bs_swap_freq = qubit_parameters["q0"][f"bs_{alice_or_bob_swap}_freqs"][
            swap_storage_mode
        ]

    # heating pulse
    bs_heating = qubit_params_module.sb_pulses[alice_or_bob_heating][
        f"bs{heated_storage_mode}"
    ]
    if bs_heating_amplitude is not None:
        bs_heating.amplitude = 1
    if bs_heating_freq is None:
        print(
            f"bs_heating_freq is None, using bs_{alice_or_bob_heating}_freqs[{heated_storage_mode}]"
        )
        bs_heating_freq = qubit_parameters["q0"][f"bs_{alice_or_bob_heating}_freqs"][
            heated_storage_mode
        ]
    if bs_heating_range is None:
        bs_heating_range = qubit_parameters["q0"][
            f"bs_{alice_or_bob_heating}_dBm_ranges"
        ][heated_storage_mode]
        if bs_heating_range != bs_swap_range:
            print(
                f"Warning: bs_heating_range ({bs_heating_range}) != bs_swap_range ({bs_swap_range})"
            )

    lo_heating = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_swap = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    lo_min = 1e9

    for lo, bs_type, bs_freq, lo_name in zip(
        [lo_heating, lo_swap],
        ["heating", "swap"],
        [bs_heating_freq, bs_swap_freq],
        ["SG2_LO", "SG4_LO"],
    ):
        new_lo = bs_freq
        step = 200e6
        new_lo = round(new_lo / step) * step
        if new_lo < lo_min:
            new_lo = lo_min

        lo_settings["q0"][serial_num][lo_name] = new_lo
        print(
            f"Warning: {bs_type} LO ({lo_name}) frequency changed to {new_lo/1e9} GHz"
        )
        lo_change = True

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    for transition in transitions:
        if alice_or_bob_swap == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    # Create Experiment - separate "bs_off_resonant" (heating) and "bs" (swap) signals
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs_swap"),
            ExperimentSignal("bs_heating"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="time_or_amp_sweep_heating",
            parameter=bs_heating_swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):

            with exp.sweep(
                uid="time_or_amp_sweep",
                parameter=swp_param,
                reset_oscillator_phase=True,
            ):

                with exp.section(
                    uid="bs_heating", play_after=None, on_system_grid=True
                ):
                    exp.play(
                        signal="bs_heating",
                        pulse=bs_heating,
                        length=bs_heating_swp_param,
                        amplitude=(
                            bs_heating_amplitude
                            if bs_heating_amplitude is not None
                            else None
                        ),
                    )

                with exp.section(
                    uid="bs_swap", play_after="bs_heating", on_system_grid=True
                ):
                    exp.play(
                        signal="bs_swap",
                        pulse=bs_swap,
                        length=bs_swap_length,
                        amplitude=(
                            bs_swap_amplitude if bs_swap_amplitude is not None else None
                        ),
                    )
                play_after = "bs_swap"

                if prepare_f:
                    with exp.section(
                        uid="ge_excitation", play_after="bs_swap", on_system_grid=True
                    ):
                        exp.play(
                            signal="qb_drive",
                            pulse=ge_X180,
                        )
                    with exp.section(
                        uid="ef_excitation",
                        play_after="ge_excitation",
                        on_system_grid=True,
                    ):
                        exp.play(
                            signal="qb_ef_drive",
                            pulse=ef_X180,
                        )
                    play_after = "ef_excitation"

                with exp.section(
                    uid="f0g1", play_after=play_after, on_system_grid=True
                ):
                    exp.play(
                        signal=sb_drive_lines["f0g1"],
                        pulse=(
                            sb_f0g1_alice
                            if alice_or_bob_swap == "alice"
                            else sb_f0g1_bob
                        ),
                        length=swp_param,
                    )

                with exp.section(
                    uid="ef_excitation_2", play_after="f0g1", on_system_grid=True
                ):
                    exp.play(
                        signal="qb_ef_drive",
                        pulse=ef_X180,
                    )

                with exp.section(
                    uid="readout", play_after="ef_excitation_2", on_system_grid=True
                ):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        handle="ac_0",
                        reset_delay=(
                            reset_delay
                            if reset_delay is not None
                            else qubit_parameters["q0"]["cavity_reset_delay"]
                        ),
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

    lo_swap = lo_settings["q0"][serial_num]["SG4_LO"]
    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs_swap"] = {}
    sig_freq_map[serial_num][ch]["bs_swap"]["frequency"] = bs_swap_freq - lo_swap
    sig_freq_map[serial_num][ch]["bs_swap"]["range"] = bs_swap_range

    lo_heating = lo_settings["q0"][serial_num]["SG2_LO"]
    ch = "SG3"
    sig_freq_map[serial_num][ch]["bs_heating"] = {}
    sig_freq_map[serial_num][ch]["bs_heating"]["frequency"] = (
        bs_heating_freq - lo_heating
    )
    sig_freq_map[serial_num][ch]["bs_heating"]["range"] = bs_heating_range

    print(f"lo_swap: {lo_swap}, lo_heating: {lo_heating}")

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

    return exp, None
