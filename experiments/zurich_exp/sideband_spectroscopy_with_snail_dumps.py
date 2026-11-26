import numpy as np
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


def sideband_spectroscopy_with_snail_dumps(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="sideband_spectroscopy_with_snail_dumps",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=-700e6, stop=700e6, count=6
    ),
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    sb_delay=0,
    single_shot=False,
    three_state=False,
    chunk_count=1,
    dump_storage_mode=5,
    prepare_snail="2",
    dump=True,
    sb_length=None,
    sb_range=None,
    sb_amplitude=None,
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

    if prepare_snail not in ["0", "1", "2", "1+2"]:
        raise ValueError("prepare_snail must be 0, 1, 2, or 1+2")

    # snail 01 by 2
    snail_01 = qubit_params_module.snail_01_by2_X180
    snail_01_freq = qubit_parameters["q0"]["snail_01_by2_freq"]
    snail_01_range = qubit_parameters["q0"]["snail_01_by2_dBm_range"]
    # snail 12 by 2
    snail_12 = (
        qubit_params_module.snail_12_by2_X90
        if prepare_snail == "1+2"
        else qubit_params_module.snail_12_by2_X180
    )
    snail_12_freq = qubit_parameters["q0"]["snail_12_by2_freq"]
    snail_12_range = qubit_parameters["q0"]["snail_12_by2_dBm_range"]

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

    # setting snail-storage dump lo
    dump_ch = "SG3"
    dump_lo_name = "SG2_LO"
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

    # FT dump and snail 01 and 12 all share an lo
    lo_name = "SG4_LO"
    lo = lo_settings["q0"][serial_num][dump_lo_name]
    lo_range = 500e6  # on either side of the lo
    lo_min = 1e9
    step = 200e6
    new_lo = np.mean([snail_01_freq, snail_12_freq, FT_dump_freq])
    new_lo = round(new_lo / step) * step
    print(f"Trying new LO frequency: {new_lo/1e9} GHz")
    if new_lo < lo_min:
        print("Warning: LO frequency is too low. Adjusting to 1 GHz.")
        new_lo = lo_min
    lo_min_max = [new_lo - lo_range, new_lo + lo_range]
    for freq in [snail_01_freq, snail_12_freq, FT_dump_freq]:
        if (freq > lo_min_max[0]) & (freq < lo_min_max[1]):
            pass
        else:
            raise ValueError(f"{freq} not in the LO range: {lo_min_max}")
    FT_dump_lo, snail_01_lo, snail_12_lo = new_lo, new_lo, new_lo

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    sb_lo = lo_settings["q0"][serial_num]["SG0_LO"]
    freq_swp.start -= sb_lo
    freq_swp.stop -= sb_lo

    if sb_length is None:
        sb_length = (
            sb_f0g1_alice.length if alice_or_bob == "alice" else sb_f0g1_bob.length
        )
    if sb_amplitude is None:
        sb_amplitude = (
            sb_f0g1_alice.amplitude
            if alice_or_bob == "alice"
            else sb_f0g1_bob.amplitude
        )
    if sb_range is None:
        sb_range = qubit_parameters["q0"][f"sb_{alice_or_bob}_dBm_range"]

    # Create Experiment
    signals = [
        ExperimentSignal("qb_drive"),
        ExperimentSignal("qb_ef_drive"),
        ExperimentSignal("storage_dump"),
        ExperimentSignal("FT_dump"),
        *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
        ExperimentSignal("measure"),
        ExperimentSignal("acquire"),
    ]
    if prepare_snail != "0":
        signals.append(ExperimentSignal("snail_01"))
        if prepare_snail == "2" or prepare_snail == "snail_12":
            signals.append(ExperimentSignal("snail_12"))
    exp = Experiment(
        uid=exp_id,
        signals=signals,
    )

    averaging_mode = AveragingMode.SINGLE_SHOT if single_shot else AveragingMode.CYCLIC
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
        averaging_mode=averaging_mode,
    ):

        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):

            # prepare snail
            if prepare_snail != "0":
                with exp.section(uid="snail_01", play_after=None, on_system_grid=True):
                    exp.play(signal="snail_01", pulse=snail_01)
                    play_after = "snail_01"
                if prepare_snail == "2" or prepare_snail == "1+2":
                    with exp.section(
                        uid="snail_12", play_after=play_after, on_system_grid=True
                    ):
                        exp.play(signal="snail_12", pulse=snail_12)
                    play_after = "snail_12"
            else:
                play_after = None

            # dump
            if dump:
                with exp.section(
                    uid="FT_dump", play_after=play_after, on_system_grid=True
                ):
                    exp.play(signal="FT_dump", pulse=FT_dump)
                with exp.section(
                    uid="storage_dump", play_after="FT_dump", on_system_grid=True
                ):
                    exp.play(signal="storage_dump", pulse=storage_dump)
                play_after = "storage_dump"

            # transmon prep
            with exp.section(
                uid="ge_excitation", play_after=play_after, on_system_grid=True
            ):
                exp.play(signal="qb_drive", pulse=ge_X180)
            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            # sb spectroscopy
            with exp.section(
                uid="sb_transition_f0g1",
                play_after="ef_excitation",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob,
                )
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            # map back to g-e
            with exp.section(
                uid="ef_excitation_2",
                play_after="sb_transition_f0g1",
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

    ch = dump_ch
    sig_freq_map[serial_num][ch]["storage_dump"] = {}
    sig_freq_map[serial_num][ch]["storage_dump"]["frequency"] = (
        storage_dump_freq - dump_lo
    )
    sig_freq_map[serial_num][ch]["storage_dump"]["range"] = storage_dump_range

    ch = "SG4"
    sig_freq_map[serial_num][ch]["FT_dump"] = {}
    sig_freq_map[serial_num][ch]["FT_dump"]["frequency"] = FT_dump_freq - FT_dump_lo
    sig_freq_map[serial_num][ch]["FT_dump"]["range"] = FT_dump_range

    if prepare_snail != "0":
        sig_freq_map[serial_num][ch]["snail_01"] = {}
        sig_freq_map[serial_num][ch]["snail_01"]["frequency"] = (
            snail_01_freq - snail_01_lo
        )
        sig_freq_map[serial_num][ch]["snail_01"]["range"] = snail_01_range

    if prepare_snail == "2" or prepare_snail == "1+2":
        sig_freq_map[serial_num][ch]["snail_12"] = {}
        sig_freq_map[serial_num][ch]["snail_12"]["frequency"] = (
            snail_12_freq - snail_12_lo
        )
        sig_freq_map[serial_num][ch]["snail_12"]["range"] = snail_12_range

    ch = "SG1"
    sig_freq_map[serial_num][ch] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["frequency"] = freq_swp
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]][
        "amplitude"
    ] = sb_amplitude  # amp_swp
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["range"] = sb_range
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["length"] = sb_length

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

    return exp, sb_lo
