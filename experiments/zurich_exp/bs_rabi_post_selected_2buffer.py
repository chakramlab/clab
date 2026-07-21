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


def bs_rabi_post_selected_2buffer(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_rabi_post_selected_2buffer",
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
    rabi_buffer="alice",
    ro_buffer="bob",
    max_fock_state=1,
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    storage_mode=1,
    sb_delay=0,
    averaging_mode=AveragingMode.SINGLE_SHOT,
    final_mapping="no pi pulse"
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    
    sb_f0g1_rabi = qubit_params_module.sb_pulses[rabi_buffer]["f0g1"]
    sb_f0g1_ro   = qubit_params_module.sb_pulses[ro_buffer]["f0g1"]

    # bs pulses
    bs_rabi_pulse = qubit_params_module.sb_pulses[rabi_buffer][f'bs{storage_mode}']
    bs_ro_pulse = qubit_params_module.sb_pulses[ro_buffer][f'bs{storage_mode}']

    if bs_length is None:
        bs_length = bs_rabi_pulse.length
    if bs_amplitude is not None:
        bs_rabi_pulse.amplitude = 1
        
    if bs_range is None:
        bs_range_rabi = qubit_parameters["q0"][f"bs_{rabi_buffer}_dBm_ranges"][storage_mode]
    else:
        bs_range_rabi = bs_range
        
    if bs_freq is None:
        bs_freq_rabi = qubit_parameters["q0"][f"bs_{rabi_buffer}_freqs"][storage_mode]
    else:
        bs_freq_rabi = bs_freq

    bs_range_ro = qubit_parameters["q0"][f"bs_{ro_buffer}_dBm_ranges"][storage_mode]
    bs_freq_ro = qubit_parameters["q0"][f"bs_{ro_buffer}_freqs"][storage_mode]

    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    
    requires_shift = False
    if bs_freq_rabi < lo - lo_range or bs_freq_rabi > lo + lo_range:
        requires_shift = True
    if bs_freq_ro < lo - lo_range or bs_freq_ro > lo + lo_range:
        requires_shift = True

    if requires_shift:
        new_lo = (bs_freq_rabi + bs_freq_ro) / 2
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
    sb_drive_lines_rabi = {}
    sb_drive_lines_ro = {}
    for transition in transitions:
        sb_drive_lines_rabi[transition] = f"sb_drive_{rabi_buffer}_{transition}"
        sb_drive_lines_ro[transition] = f"sb_drive_{ro_buffer}_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs_rabi"),
            ExperimentSignal("bs_ro"),
            *[ExperimentSignal(sb_drive_lines_rabi[_]) for _ in transitions],
            *[ExperimentSignal(sb_drive_lines_ro[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type, averaging_mode=averaging_mode
    ):
        with exp.sweep(
                uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True,
            ):
            if final_mapping=="no pi pulse":
                # Initial excitation to transmon e, then f, then to rabi buffer
                with exp.section(uid="ge_excitation_0", play_after=None, on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(uid="ef_excitation_0", play_after="ge_excitation_0", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                with exp.section(uid="sb_load_rabi_buffer_0", play_after="ef_excitation_0", on_system_grid=True):
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                # Rabi sweep
                with exp.section(uid="bs_rabi_sweep_0", play_after="sb_load_rabi_buffer_0", on_system_grid=True):
                    exp.play(
                        signal="bs_rabi", pulse=bs_rabi_pulse,
                        length=bs_length_rabi, amplitude=bs_amplitude_rabi
                    )

                # Post-selection / Erasure Check (2-buffer scheme)
                # 1. Move Storage to Bob (ro_buffer)
                with exp.section(uid="bs_move_storage_to_ro_buffer_0", play_after="bs_rabi_sweep_0", on_system_grid=True):
                    exp.play(signal="bs_ro", pulse=bs_ro_pulse)

                # 2. Sideband Alice (rabi_buffer) -> Transmon f
                with exp.section(uid="sb_unload_rabi_buffer_0", play_after="bs_move_storage_to_ro_buffer_0", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

                # 3. Transmon f -> e
                with exp.section(uid="ef_swap_0", play_after="sb_unload_rabi_buffer_0", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                # 4. Sideband Bob (ro_buffer) -> Transmon f
                with exp.section(uid="sb_unload_ro_buffer_0", play_after="ef_swap_0", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_ro["f0g1"], pulse=sb_f0g1_ro)
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)

                # 5. Readout
                with exp.section(uid="readout_0", play_after="sb_unload_ro_buffer_0", on_system_grid=True):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        handle="ac_0",
                        reset_delay=qubit_parameters["q0"]["cavity_reset_delay"] if reset_delay is None else reset_delay,
                        acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                    )

            elif final_mapping=="pi pulse":
                # Initial excitation to transmon e, then f, then to rabi buffer
                with exp.section(uid="ge_excitation_1", play_after=None, on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(uid="ef_excitation_1", play_after="ge_excitation_1", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                with exp.section(uid="sb_load_rabi_buffer_1", play_after="ef_excitation_1", on_system_grid=True):
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

                # Rabi sweep
                with exp.section(uid="bs_rabi_sweep_1", play_after="sb_load_rabi_buffer_1", on_system_grid=True):
                    exp.play(
                        signal="bs_rabi", pulse=bs_rabi_pulse,
                        length=bs_length_rabi, amplitude=bs_amplitude_rabi
                    )

                # Post-selection / Erasure Check (2-buffer scheme)
                # 1. Move Storage to Bob (ro_buffer)
                with exp.section(uid="bs_move_storage_to_ro_buffer_1", play_after="bs_rabi_sweep_1", on_system_grid=True):
                    exp.play(signal="bs_ro", pulse=bs_ro_pulse)

                # 2. Sideband Alice (rabi_buffer) -> Transmon f
                with exp.section(uid="sb_unload_rabi_buffer_1", play_after="bs_move_storage_to_ro_buffer_1", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

                # 3. Transmon f -> e
                with exp.section(uid="ef_swap_1", play_after="sb_unload_rabi_buffer_1", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                # 4. Sideband Bob (ro_buffer) -> Transmon f
                with exp.section(uid="sb_unload_ro_buffer_1", play_after="ef_swap_1", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_ro["f0g1"], pulse=sb_f0g1_ro)
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)

                with exp.section(uid="ge_1_0", play_after="sb_unload_ro_buffer_1", on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                # 5. Readout
                with exp.section(uid="readout_1", play_after="ge_1_0", on_system_grid=True):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        handle="ac_1",
                        reset_delay=qubit_parameters["q0"]["cavity_reset_delay"] if reset_delay is None else reset_delay,
                        acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                    )

            elif final_mapping=="pi pulse 2":
                # Initial excitation to transmon e, then f, then to rabi buffer
                with exp.section(uid="ge_excitation_2", play_after=None, on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(uid="ef_excitation_2", play_after="ge_excitation_2", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                with exp.section(uid="sb_load_rabi_buffer_2", play_after="ef_excitation_2", on_system_grid=True):
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

                # Rabi sweep
                with exp.section(uid="bs_rabi_sweep_2", play_after="sb_load_rabi_buffer_2", on_system_grid=True):
                    exp.play(
                        signal="bs_rabi", pulse=bs_rabi_pulse,
                        length=bs_length_rabi, amplitude=bs_amplitude_rabi
                    )

                # Post-selection / Erasure Check (2-buffer scheme)
                # 1. Move Storage to Bob (ro_buffer)
                with exp.section(uid="bs_move_storage_to_ro_buffer_2", play_after="bs_rabi_sweep_2", on_system_grid=True):
                    exp.play(signal="bs_ro", pulse=bs_ro_pulse)

                # 2. Sideband Alice (rabi_buffer) -> Transmon f
                with exp.section(uid="sb_unload_rabi_buffer_2", play_after="bs_move_storage_to_ro_buffer_2", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_rabi["f0g1"], pulse=sb_f0g1_rabi)
                    exp.delay(signal=sb_drive_lines_rabi["f0g1"], time=sb_delay)

                # 3. Transmon f -> e
                with exp.section(uid="ef_swap_2", play_after="sb_unload_rabi_buffer_2", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)

                # 4. Sideband Bob (ro_buffer) -> Transmon f
                with exp.section(uid="sb_unload_ro_buffer_2", play_after="ef_swap_2", on_system_grid=True):
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)
                    exp.play(signal=sb_drive_lines_ro["f0g1"], pulse=sb_f0g1_ro)
                    exp.delay(signal=sb_drive_lines_ro["f0g1"], time=sb_delay)

                with exp.section(uid="ge_2_0", play_after="sb_unload_ro_buffer_2", on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(uid="ef_2_0", play_after="ge_2_0", on_system_grid=True):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)
                
                with exp.section(uid="ge_2_1", play_after="ef_2_0", on_system_grid=True):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                # 5. Readout
                with exp.section(uid="readout_2", play_after="ge_2_1", on_system_grid=True):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        handle="ac_2",
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
    sig_freq_map[serial_num][ch]["bs_rabi"] = {}
    sig_freq_map[serial_num][ch]["bs_rabi"]["frequency"] = bs_freq_rabi - lo
    sig_freq_map[serial_num][ch]["bs_rabi"]["range"] = bs_range_rabi

    sig_freq_map[serial_num][ch]["bs_ro"] = {}
    sig_freq_map[serial_num][ch]["bs_ro"]["frequency"] = bs_freq_ro - lo
    sig_freq_map[serial_num][ch]["bs_ro"]["range"] = bs_range_ro

    print("bs_rabi map:", sig_freq_map[serial_num][ch]["bs_rabi"])
    print("bs_ro map:", sig_freq_map[serial_num][ch]["bs_ro"])

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
