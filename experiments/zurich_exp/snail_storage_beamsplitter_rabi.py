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


def snail_storage_beamsplitter_rabi(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="rabi",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(uid="swp_param", start=0, stop=4e-6, count=17),
    dump_freq=None,
    dump_length=None,
    dump_range=None,
    dump_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    storage_mode=0,
    sb_delay=0,
    dump_lo_name=None,
    dump_channel_name=None,
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
    buff_snail = qubit_params_module.sb_pulses[alice_or_bob]["bs0"]

    buff_snail_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][0]

    dump = qubit_params_module.snail_storage_bs_pulses[f"bs_snail_{storage_mode}"]
    storage_idx = storage_mode - 1
    if dump_freq is None:
        dump_freq = qubit_parameters["q0"]["bs_snail_freqs"][storage_idx]
    if dump_length is None:
        dump_length = dump.length
    if dump_amplitude is not None:
        dump.amplitude = 1
    if dump_range is None:
        dump_range = qubit_parameters["q0"]["bs_snail_dBm_ranges"][storage_idx]

    if dump_lo_name is None:
        dump_lo_name = "SG2_LO"
    if dump_channel_name is None:
        dump_channel_name = "SG3"

    lo_swap = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_dump = lo_settings["q0"][serial_num][dump_lo_name]
    lo_range = 0.5e9
    lo_min = 1e9

    for lo, drive_type, lo_name in zip(
        [lo_swap, lo_dump],
        ["swap", "dump"],
        ["SG4_LO", dump_lo_name],
    ):
        if drive_type == "swap":
            new_lo = buff_snail_freq
            step = 200e6
            new_lo = round(new_lo / step) * step
            if new_lo < lo_min:
                new_lo = lo_min
            if (
                buff_snail_freq < new_lo - lo_range
                or buff_snail_freq > new_lo + lo_range
            ):
                raise ValueError(
                    f"{drive_type} frequency is outside of the allowed LO range."
                )
            lo_swap = new_lo
            lo_settings["q0"][serial_num][lo_name] = lo_swap
            print(
                f"Warning: {drive_type} LO ({lo_name}) frequency changed to {new_lo / 1e9} GHz"
            )
            lo_change = True
        else:
            new_lo = dump_freq
            step = 200e6
            new_lo = round(new_lo / step) * step
            if new_lo < lo_min:
                new_lo = lo_min
            if dump_freq < new_lo - lo_range or dump_freq > new_lo + lo_range:
                raise ValueError(
                    f"{drive_type} frequency is outside of the allowed LO range."
                )
            lo_dump = new_lo
            lo_settings["q0"][serial_num][lo_name] = lo_dump
            print(
                f"Warning: {drive_type} LO ({lo_name}) frequency changed to {new_lo / 1e9} GHz"
            )
            lo_change = True

    transitions = [f"f{i}g{i + 1}" for i in range(max_fock_state)]
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
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("dump"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True
        ):
            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation_1",
                play_after="ge_excitation",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(
                uid="sb_transition_f0g1_1",
                play_after="ef_excitation_1",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                )
            with exp.section(
                uid="sb_delay_pre",
                play_after="sb_transition_f0g1_1",
                on_system_grid=True,
            ):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            with exp.section(
                uid="buff_snail", play_after="sb_delay_pre", on_system_grid=True
            ):
                exp.play(
                    signal="bs",
                    pulse=buff_snail,
                )

            with exp.section(uid="dump", play_after="buff_snail", on_system_grid=True):
                exp.play(
                    signal="dump",
                    pulse=dump,
                    length=swp_param,
                    amplitude=dump_amplitude if dump_amplitude is not None else None,
                )

            with exp.section(uid="snail_buff", play_after="dump", on_system_grid=True):
                exp.play(
                    signal="bs",
                    pulse=buff_snail,
                )

            with exp.section(
                uid="sb_delay_post", play_after="snail_buff", on_system_grid=True
            ):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)
            with exp.section(
                uid="sb_transition_f0g1_2",
                play_after="sb_delay_post",
                on_system_grid=True,
            ):
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

    swap_channel_name = "SG4"
    swap_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][0]
    # check ranges
    ranges_dict = {"swap": swap_range, "dump": dump_range}
    channels_dict = {
        "dump": dump_channel_name,
        "swap": swap_channel_name,
    }
    ranges = list(ranges_dict.values())
    if len(set(ranges)) > 1:
        print(
            "Warning: considering the 2 drives (buff snail swap, dump), not all dBm ranges are the same."
        )
        print("Checking if the mismatched ranges are using the same channel")
        for drive_type, range_val in ranges_dict.items():
            channel = channels_dict[drive_type]
            for drive_type2, range_val2 in ranges_dict.items():
                if drive_type2 != drive_type:
                    channel2 = channels_dict[drive_type2]
                    if channel == channel2 and range_val != range_val2:
                        raise ValueError(
                            f"Channel {channel} has 2 drives: {drive_type} and {drive_type2} with different ranges: {range_val} and {range_val2}"
                        )

    ch = swap_channel_name
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = buff_snail_freq - lo_swap
    sig_freq_map[serial_num][ch]["bs"]["range"] = swap_range

    ch = dump_channel_name
    sig_freq_map[serial_num][ch]["dump"] = {}
    sig_freq_map[serial_num][ch]["dump"]["frequency"] = dump_freq - lo_dump
    sig_freq_map[serial_num][ch]["dump"]["range"] = dump_range

    print(f"lo_swap: {lo_swap}, lo_dump: {lo_dump}")

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

    return exp, lo_dump
