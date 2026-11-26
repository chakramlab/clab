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


def bs_rabi_with_snail_dumps(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_rabi_with_snail_dumps",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(uid="swp_param", start=1e-9, stop=10e-6, count=6),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_ramp=None,
    bs_amplitude=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    storage_mode=1,
    sb_delay=0,
    single_shot=False,
    three_state=False,
    chunk_count=1,
    bs_channel=None,
    bs_lo_name=None,
    dump_lo_name=None,
    dump_channel_name=None,
    FT_dump_lo_name=None,
    FT_dump_channel_name=None,
    dump_storage_mode=5,
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

    bs = qubit_params_module.sb_pulses[alice_or_bob][f"bs{storage_mode}"]
    if bs_length is None:
        bs_length = bs.length
    if bs_amplitude is not None:
        bs.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][storage_mode]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][storage_mode]
    if bs_ramp is not None:
        bs.pulse_parameters["ramp"] = bs_ramp

    storage_dump = qubit_params_module.snail_storage_bs_pulses[
        f"bs_snail_{dump_storage_mode}"
    ]
    storage_dump_freq = qubit_parameters["q0"]["bs_snail_freqs"][dump_storage_mode - 1]
    storage_dump_range = qubit_parameters["q0"]["bs_snail_dBm_ranges"][
        dump_storage_mode - 1
    ]

    FT_dump = qubit_params_module.sb_snail_FT_f0g1
    FT_dump_freq = qubit_parameters["q0"]["sb_snail_FT_freq"]
    FT_dump_range = qubit_parameters["q0"]["sb_snail_FT_dBm_range"]

    bs_ch = "SG4" if bs_channel is None else bs_channel
    bs_lo_name = "SG4_LO" if bs_lo_name is None else bs_lo_name
    lo = lo_settings["q0"][serial_num][bs_lo_name]
    lo_range = 500e6  # on either side of the lo
    lo_min = 1e9
    step = 200e6
    new_lo = bs_freq
    new_lo = round(new_lo / step) * step
    print(f"Trying new LO frequency: {new_lo/1e9} GHz")
    if new_lo < lo_min:
        print("Warning: LO frequency is too low. Adjusting to 1 GHz.")
        new_lo = lo_min
    bs_lo = new_lo
    lo_settings["q0"][serial_num][bs_lo_name] = bs_lo
    print(f"Warning: BS LO ({bs_lo_name}) frequency changed to {bs_lo/1e9} GHz")
    lo_change = True

    dump_ch = "SG3" if dump_channel_name is None else dump_channel_name
    dump_lo_name = "SG2_LO" if dump_lo_name is None else dump_lo_name
    lo = lo_settings["q0"][serial_num][dump_lo_name]
    lo_range = 500e6  # on either side of the lo
    lo_min = 1e9
    step = 200e6
    new_lo = storage_dump_freq
    new_lo = round(new_lo / step) * step
    print(f"Trying new LO frequency: {new_lo/1e9} GHz")
    if new_lo < lo_min:
        print("Warning: LO frequency is too low. Adjusting to 1 GHz.")
        new_lo = lo_min
    dump_lo = new_lo
    lo_settings["q0"][serial_num][dump_lo_name] = dump_lo
    print(f"Warning: Dump LO ({dump_lo_name}) frequency changed to {dump_lo/1e9} GHz")

    FT_dump_ch = "SG4" if FT_dump_channel_name is None else FT_dump_channel_name
    FT_dump_lo_name = "SG4_LO" if FT_dump_lo_name is None else FT_dump_lo_name
    lo = lo_settings["q0"][serial_num][FT_dump_lo_name]
    lo_range = 500e6  # on either side of the lo
    # check if the current lo setting is good
    lo_good = (FT_dump_freq > lo - lo_range) & (FT_dump_freq < lo + lo_range)
    if not lo_good:
        raise ValueError(
            f"FT lo {FT_dump_lo_name} is not good. lo value={lo}. FT frequency={FT_dump_freq}"
        )
    else:
        print(f"FT lo {FT_dump_lo_name} is good")
    FT_dump_lo = lo

    if swp_amp:
        bs_amplitude = swp_param
    else:
        bs_length = swp_param

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("storage_dump"),
            ExperimentSignal("FT_dump"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    averaging_mode = AveragingMode.SINGLE_SHOT if single_shot else AveragingMode.CYCLIC
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
            chunk_count=chunk_count,
        ):

            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(
                uid="sb_transition_f0g1_1",
                play_after="ef_excitation",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                )
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)
            with exp.section(
                uid="bs", play_after="sb_transition_f0g1_1", on_system_grid=True
            ):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )
            with exp.section(uid="FT_dump", play_after="bs", on_system_grid=True):
                exp.play(signal="FT_dump", pulse=FT_dump)
            with exp.section(
                uid="storage_dump", play_after="FT_dump", on_system_grid=True
            ):
                exp.play(signal="storage_dump", pulse=storage_dump)
            with exp.section(
                uid="sb_transition_f0g1_2",
                play_after="storage_dump",
                on_system_grid=True,
            ):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                )
            with exp.section(
                uid="ef_excitation_2",
                play_after="sb_transition_f0g1_2",
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
        single_shot=three_state if single_shot else False,
    )

    ch = bs_ch
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - bs_lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    ch = dump_ch
    sig_freq_map[serial_num][ch]["storage_dump"] = {}
    sig_freq_map[serial_num][ch]["storage_dump"]["frequency"] = (
        storage_dump_freq - dump_lo
    )
    sig_freq_map[serial_num][ch]["storage_dump"]["range"] = storage_dump_range

    ch = FT_dump_ch
    sig_freq_map[serial_num][ch]["FT_dump"] = {}
    sig_freq_map[serial_num][ch]["FT_dump"]["frequency"] = FT_dump_freq - FT_dump_lo
    sig_freq_map[serial_num][ch]["FT_dump"]["range"] = FT_dump_range

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

    return exp, bs_lo
