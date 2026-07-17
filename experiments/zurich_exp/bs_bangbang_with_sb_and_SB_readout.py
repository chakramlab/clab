from laboneq.simple import (  # Removed the faulty match/case imports!
    AcquisitionType,
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


def bs_bangbang_with_sb_and_SB_readout(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_with_sb_bangbang",
    average_exponent=5,  # 2^n averages
    swp_param=None,  # Expecting a SweepParameter tracking integer pulse counts
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    bs_ramp=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,
    rotate_ro=False,
    thresholds=None,
    storage_mode=1,
    sb_delay=0,
    chunk_count=1,
    pulse_delay=0,
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

    # bs pulse configuration
    if bs_ramp is None:
        bs_ramp = qubit_params_module.sb_pulses[alice_or_bob][
            f"bs{storage_mode}"
        ].pulse_parameters["ramp"]
    bs = qubit_params_module.sb_pulses[alice_or_bob][f"bs{storage_mode}"]
    bs.pulse_parameters["ramp"] = bs_ramp
    if bs_length is None:
        bs_length = bs.length

    if bs_amplitude is not None:
        bs.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][storage_mode]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][storage_mode]

    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    if bs_freq < lo - lo_range or bs_freq > lo + lo_range:
        new_lo = bs_freq
        step = 200e6
        new_lo = round(new_lo / step) * step
        if new_lo < 1e9:
            new_lo = 0
        lo_settings["q0"][serial_num]["SG4_LO"] = new_lo
        lo = new_lo
        print(f"Warning: LO frequency changed to {new_lo/1e9} GHz")

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
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
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):

        with exp.sweep(
            uid="pulse_count_sweep",
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

            # --- MODIFIED: Bang-Bang Pulse Stacking Section ---
            with exp.section(
                uid="bs", play_after="sb_transition_f0g1_1", on_system_grid=True
            ):
                # FIXED: Called as methods on the `exp` object
                with exp.match(sweep_parameter=swp_param):
                    for n in swp_param.values:
                        n_int = int(n)
                        with exp.case(n_int):
                            for _ in range(n_int):
                                exp.play(
                                    signal="bs",
                                    pulse=bs,
                                    length=bs_length,
                                    amplitude=(
                                        bs_amplitude
                                        if bs_amplitude is not None
                                        else None
                                    ),
                                )
                                (
                                    exp.delay(signal="bs", time=pulse_delay)
                                    if _ < n_int - 1
                                    else None
                                )
            # --------------------------------------------------

            with exp.section(
                uid="sb_transition_f0g1_2", play_after="bs", on_system_grid=True
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

    # Setup calibration and signal map for the experiment
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
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    print(sig_freq_map[serial_num][ch]["bs"])

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
