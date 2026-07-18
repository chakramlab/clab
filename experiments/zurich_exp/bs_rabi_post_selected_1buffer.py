from laboneq.simple import (
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


def bs_rabi_post_selected_1buffer(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_rabi_post_selected_1buffer",
    average_exponent=5,
    swp_param=LinearSweepParameter(
        uid="swp_param", start=1e-9, stop=10e-6, count=6
    ),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    buffer="alice",
    max_fock_state=1,
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    storage_mode=1,
    sb_delay=0
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    
    sb_f0g1 = qubit_params_module.sb_pulses[buffer]["f0g1"]

    # bs pulse
    bs_pulse = qubit_params_module.sb_pulses[buffer][f'bs{storage_mode}']

    if bs_length is None:
        bs_length = bs_pulse.length
    if bs_amplitude is not None:
        bs_pulse.amplitude = 1
        
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{buffer}_dBm_ranges"][storage_mode]
        
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{buffer}_freqs"][storage_mode]

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

    if swp_amp:
        bs_amplitude_rabi = swp_param
        bs_length_rabi = bs_length
    else:
        bs_length_rabi = swp_param
        bs_amplitude_rabi = bs_amplitude if bs_amplitude is not None else None

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    for transition in transitions:
        sb_drive_lines[transition] = f"sb_drive_{buffer}_{transition}"

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
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True,
        ):
            # Initial excitation to transmon e, then f, then to buffer
            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(uid="ef_excitation", play_after="ge_excitation", on_system_grid=True):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="sb_load_buffer", play_after="ef_excitation", on_system_grid=True):
                exp.play(signal=sb_drive_lines["f0g1"], pulse=sb_f0g1)
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            # Rabi sweep
            with exp.section(uid="bs_rabi_sweep", play_after="sb_load_buffer", on_system_grid=True):
                exp.play(
                    signal="bs", pulse=bs_pulse,
                    length=bs_length_rabi, amplitude=bs_amplitude_rabi
                )

            # Post-selection / Erasure Check (1-buffer scheme)
            # 1. Unload Buffer -> Transmon f
            with exp.section(uid="sb_unload_buffer_1", play_after="bs_rabi_sweep", on_system_grid=True):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)
                exp.play(signal=sb_drive_lines["f0g1"], pulse=sb_f0g1)
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            # 2. Transmon f -> e
            with exp.section(uid="ef_swap", play_after="sb_unload_buffer_1", on_system_grid=True):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            # 3. Move Storage to Buffer (standard calibrated pi pulse)
            with exp.section(uid="bs_move_storage_to_buffer", play_after="ef_swap", on_system_grid=True):
                exp.play(signal="bs", pulse=bs_pulse)

            # 4. Unload Buffer -> Transmon f
            with exp.section(uid="sb_unload_buffer_2", play_after="bs_move_storage_to_buffer", on_system_grid=True):
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)
                exp.play(signal=sb_drive_lines["f0g1"], pulse=sb_f0g1)
                exp.delay(signal=sb_drive_lines["f0g1"], time=sb_delay)

            # 5. Readout
            with exp.section(uid="readout", play_after="sb_unload_buffer_2", on_system_grid=True):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"] if reset_delay is None else reset_delay,
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
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    print("bs map:", sig_freq_map[serial_num][ch]["bs"])

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
