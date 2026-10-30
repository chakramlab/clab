from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
    parsing_single_alice_bob_cav_and_sb_transitions,
)
from .load_qubit_params import load_qubit_params


def bs_rabi_post_selected_snail_ground(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_rabi_post_selected_snail_ground",
    average_exponent=5,
    swp_param=LinearSweepParameter(uid="swp_param", start=1e-9, stop=10e-6, count=6),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    rabi_buffer="alice",
    max_fock_state=1,
    rotate_ro=False,
    thresholds=None,
    kernels=None,
    swp_amp=False,
    storage_mode=1,
    sb_delay=0,
    averaging_mode=AveragingMode.SINGLE_SHOT,
    measure_transmon=True,
    snail_transmon_bs=True,
    delay_after_first_measure=1e-6,
    temp_dump=True,
    temp_dump_storage_mode=2,
    snail_drive=False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel if kernels is None else kernels
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180

    sb_f0g1 = qubit_params_module.sb_pulses[rabi_buffer]["f0g1"]

    # bs rabi
    bs_rabi_pulse = qubit_params_module.sb_pulses[rabi_buffer][f"bs{storage_mode}"]

    # temp dump bs
    bs_temp_dump_pulse = qubit_params_module.sb_pulses[rabi_buffer][
        f"bs{temp_dump_storage_mode}"
    ]
    bs_temp_dump_freq = qubit_parameters["q0"][f"bs_{rabi_buffer}_freqs"][
        temp_dump_storage_mode
    ]
    bs_temp_dump_range = qubit_parameters["q0"][f"bs_{rabi_buffer}_dBm_ranges"][
        temp_dump_storage_mode
    ]

    # snail transmon bs
    bs_ST_pulse = qubit_params_module.bs_transmon_snail_eg_ge
    bs_ST_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]
    bs_ST_range = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_dBm_range"]

    # snail 12 by 2
    snail_drive = qubit_params_module.snail_12_by2
    snail_drive_freq = qubit_parameters["q0"]["snail_12_by2_freq"]
    snail_drive_range = qubit_parameters["q0"]["snail_12_by2_dBm_range"]

    if bs_length is None:
        bs_length = bs_rabi_pulse.length
    if bs_amplitude is not None:
        bs_rabi_pulse.amplitude = 1
    if bs_range is None:
        bs_range_rabi = qubit_parameters["q0"][f"bs_{rabi_buffer}_dBm_ranges"][
            storage_mode
        ]
    else:
        bs_range_rabi = bs_range
    if bs_freq is None:
        bs_freq_rabi = qubit_parameters["q0"][f"bs_{rabi_buffer}_freqs"][storage_mode]
    else:
        bs_freq_rabi = bs_freq

    lo_bs_rabi = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_bs_ST = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_snail = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_temp_dump = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    lo_min = 1e9

    # snail - transmon lo
    lo_name = "SG2_LO"
    new_lo = bs_ST_freq
    step = 200e6
    new_lo = round(new_lo / step) * step
    if new_lo < lo_min:
        new_lo = lo_min
    lo_settings["q0"][serial_num][lo_name] = new_lo
    print(
        f"Warning: bs transmon snail LO ({lo_name}) frequency changed to {new_lo/1e9} GHz"
    )
    lo_bs_ST = new_lo

    # bs rabi and snail drive and temp dump lo
    lo_name = "SG4_LO"
    new_lo = (snail_drive_freq + bs_freq_rabi) / 2
    step = 200e6
    new_lo = round(new_lo / step) * step
    if (
        snail_drive_freq > new_lo - lo_range or snail_drive_freq < new_lo + lo_range
    ) and (bs_freq_rabi > new_lo - lo_range or bs_freq_rabi < new_lo + lo_range):
        print(f"{new_lo} GHz for {lo_name} is good for snail drive and bs rabi")
    else:
        raise ValueError(
            f"LO frequency {new_lo} GHz for {lo_name} is not good for snail drive and bs rabi"
        )
    lo_settings["q0"][serial_num][lo_name] = new_lo
    print(
        f"Warning: bs transmon snail and snail drive LO and temp dump ({lo_name}) frequency changed to {new_lo/1e9} GHz"
    )
    lo_bs_rabi = new_lo
    lo_snail = new_lo
    lo_temp_dump = new_lo

    if swp_amp:
        bs_amplitude_rabi = swp_param
        bs_length_rabi = bs_length
    else:
        bs_length_rabi = swp_param
        bs_amplitude_rabi = bs_amplitude if bs_amplitude is not None else None

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines_rabi = {}
    for transition in transitions:
        sb_drive_lines_rabi[transition] = f"sb_drive_{rabi_buffer}_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs_rabi"),
            ExperimentSignal("bs_temp_dump"),
            ExperimentSignal("bs_ST"),
            ExperimentSignal("snail_drive"),
            *[ExperimentSignal(sb_drive_lines_rabi[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
        averaging_mode=averaging_mode,
    ):
        with exp.sweep(
            uid="time_or_amp_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
        ):
            # Initial excitation to transmon e, then f, then to rabi buffer
            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(
                uid="sb_load_rabi_buffer",
                play_after="ef_excitation",
                on_system_grid=True,
            ):
                exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1)
                exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                play_after = "sb_load_rabi_buffer"

            # Rabi sweep
            with exp.section(uid="bs_rabi", play_after=play_after, on_system_grid=True):
                exp.play(
                    signal="bs_rabi",
                    pulse=bs_rabi_pulse,
                    length=bs_length_rabi,
                    amplitude=bs_amplitude_rabi,
                )
            play_after = "bs_rabi"

            # move photon in buffer to a different storage mode
            if temp_dump:
                with exp.section(
                    uid="bs_temp_dump", play_after=play_after, on_system_grid=True
                ):
                    exp.play(
                        signal="bs_temp_dump",
                        pulse=bs_temp_dump_pulse,
                    )
                play_after = "bs_temp_dump"

            # Post-selection snail ground state Check
            if snail_transmon_bs:
                # 1. Move snail 1 to transmon e
                with exp.section(
                    uid="bs_ST", play_after=play_after, on_system_grid=True
                ):
                    exp.play(signal="bs_ST", pulse=bs_ST_pulse)

                play_after = "bs_ST"

            if measure_transmon:
                # 2. Measure transmon
                with exp.section(
                    uid="measure_transmon", play_after=play_after, on_system_grid=True
                ):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                        handle="transmon_state",
                    )
                    exp.delay(signal="measure", time=delay_after_first_measure)
                play_after = "measure_transmon"

            # undump back to the buffer
            if temp_dump:
                with exp.section(
                    uid="bs_temp_undump", play_after=play_after, on_system_grid=True
                ):
                    exp.play(
                        signal="bs_temp_dump",
                        pulse=bs_temp_dump_pulse,
                    )
                play_after = "bs_temp_undump"

            # standard buffer readout
            # 1. Sideband (rabi_buffer) -> Transmon f
            with exp.section(
                uid="sb_unload_rabi_buffer",
                play_after=play_after,
                on_system_grid=True,
            ):
                exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1)
                exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

            # 2. Transmon f -> e
            with exp.section(
                uid="fe",
                play_after="sb_unload_rabi_buffer",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            # 3. Readout
            with exp.section(uid="readout", play_after="fe", on_system_grid=True):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=(
                        qubit_parameters["q0"]["cavity_reset_delay"]
                        if reset_delay is None
                        else reset_delay
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

    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs_rabi"] = {}
    sig_freq_map[serial_num][ch]["bs_rabi"]["frequency"] = bs_freq_rabi - lo_bs_rabi
    sig_freq_map[serial_num][ch]["bs_rabi"]["range"] = bs_range_rabi

    sig_freq_map[serial_num][ch]["bs_temp_dump"] = {}
    sig_freq_map[serial_num][ch]["bs_temp_dump"]["frequency"] = (
        bs_temp_dump_freq - lo_temp_dump
    )
    sig_freq_map[serial_num][ch]["bs_temp_dump"]["range"] = bs_temp_dump_range

    sig_freq_map[serial_num][ch]["snail_drive"] = {}
    sig_freq_map[serial_num][ch]["snail_drive"]["frequency"] = (
        snail_drive_freq - lo_snail
    )
    sig_freq_map[serial_num][ch]["snail_drive"]["range"] = snail_drive_range

    ch = "SG3"
    sig_freq_map[serial_num][ch]["bs_ST"] = {}
    sig_freq_map[serial_num][ch]["bs_ST"]["frequency"] = bs_ST_freq - lo_bs_ST
    sig_freq_map[serial_num][ch]["bs_ST"]["range"] = bs_ST_range

    print(f"lo_bs_rabi: {lo_bs_rabi}, lo_bs_ST: {lo_bs_ST}, lo_snail: {lo_snail}")
    print(f"bs_rabi map: {sig_freq_map[serial_num]["SG4"]['bs_rabi']}")
    print(f"bs_ST map: {sig_freq_map[serial_num]["SG3"]['bs_ST']}")
    print(f"snail drive map: {sig_freq_map[serial_num]["SG4"]['snail_drive']}")
    print(f"temp dump map: {sig_freq_map[serial_num]["SG4"]['bs_temp_dump']}\n")

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

    return exp, [lo_bs_rabi, lo_bs_ST, lo_snail]
