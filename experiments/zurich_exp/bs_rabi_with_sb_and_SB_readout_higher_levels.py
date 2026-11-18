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


def bs_rabi_with_sb_and_SB_readout_higher_levels(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="rabi",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(uid="time_swp_param", start=0, stop=1e-6, count=26),
    swp_amp=False,
    snail_drive_freq=None,
    snail_drive_length=None,
    snail_drive_range=None,
    snail_drive_amplitude=None,
    bs_freq=None,
    bs_length=None,
    bs_range=None,
    bs_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    storage_mode=0,
    sb_delay=0,
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
    snail_drive = qubit_params_module.snail_12_by2

    if snail_drive_freq is None:
        snail_drive_freq = qubit_parameters["q0"]["snail_12_by2_freq"]
    if snail_drive_amplitude is not None:
        snail_drive.amplitude = 1
    if snail_drive_range is None:
        snail_drive_range = qubit_parameters["q0"]["snail_12_by2_dBm_range"]
    if snail_drive_length is None:
        snail_drive_length = snail_drive.length

    bs = qubit_params_module.sb_pulses[alice_or_bob][f"bs{storage_mode}"]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][storage_mode]
    if bs_length is None:
        bs_length = bs.length
    if bs_amplitude is not None:
        bs.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][storage_mode]

    lo_snail = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_bs = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    lo_min = 1e9

    for lo, bs_type, lo_name in zip(
        [lo_snail, lo_bs],
        ["snail", "bs"],
        ["SG2_LO", "SG4_LO"],
    ):
        if bs_type == "bs":
            new_lo = bs_freq
            step = 200e6
            new_lo = round(new_lo / step) * step
            if new_lo < lo_min:
                new_lo = lo_min
            if bs_freq < new_lo - lo_range or bs_freq > new_lo + lo_range:
                raise ValueError("BS frequency is outside of the allowed LO range.")
            lo_bs = new_lo
            lo_settings["q0"][serial_num][lo_name] = lo_bs
            print(
                f"Warning: {bs_type} LO ({lo_name}) frequency changed to {new_lo/1e9} GHz"
            )
            lo_change = True
        else:
            new_lo = snail_drive_freq
            step = 200e6
            new_lo = round(new_lo / step) * step
            if new_lo < lo_min:
                new_lo = lo_min
            if (
                snail_drive_freq < new_lo - lo_range
                or snail_drive_freq > new_lo + lo_range
            ):
                raise ValueError("Snail frequency is outside of the allowed LO range.")
            lo_snail = new_lo
            lo_settings["q0"][serial_num][lo_name] = lo_snail
            print(
                f"Warning: {bs_type} LO ({lo_name}) frequency changed to {new_lo/1e9} GHz"
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
            ExperimentSignal("snail_drive"),
            ExperimentSignal("bs"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    if swp_amp:
        snail_drive_amplitude = swp_param
    else:
        snail_drive_length = swp_param

    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="time_sweep", parameter=swp_param, reset_oscillator_phase=True
        ):
            with exp.section(uid="ge_excitation", play_after=None):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation_1", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="sb_transition_f0g1_1", play_after="ef_excitation_1"):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                )
            with exp.section(uid="sb_delay_pre", play_after="sb_transition_f0g1_1"):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            with exp.section(uid="bs", play_after="sb_delay_pre"):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            with exp.section(uid="snail_drive", play_after="bs"):
                exp.play(
                    signal="snail_drive",
                    pulse=snail_drive,
                    length=snail_drive_length,
                    amplitude=(
                        snail_drive_amplitude
                        if snail_drive_amplitude is not None
                        else None
                    ),
                )

            with exp.section(uid="bs_2", play_after="snail_drive"):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            with exp.section(uid="sb_delay_post", play_after="bs_2"):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            with exp.section(uid="sb_transition_f0g1_2", play_after="sb_delay_post"):
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

            with exp.section(uid="readout", play_after="ef_excitation_2"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=(
                        qubit_parameters["q0"]["cavity_reset_delay"]
                        if storage_mode != 0
                        else 5 * 2 * qubit_parameters["q0"]["snail_01_t1"]
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
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo_bs
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    ch = "SG3"
    sig_freq_map[serial_num][ch]["snail_drive"] = {}
    sig_freq_map[serial_num][ch]["snail_drive"]["frequency"] = (
        snail_drive_freq - lo_snail
    )
    sig_freq_map[serial_num][ch]["snail_drive"]["range"] = snail_drive_range

    print(f"lo_bs: {lo_bs}, lo_snail: {lo_snail}")

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
