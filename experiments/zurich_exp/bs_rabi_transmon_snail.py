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


def bs_transmon_snail_rabi(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_transmon_snail_rabi",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(uid="swp_param", start=1e-9, stop=10e-6, count=6),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    reset_delay=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    channel=None,
    lo_name=None,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180

    bs = qubit_params_module.bs_transmon_snail_eg_ge
    if bs_length is None:
        bs_length = bs.length
    if bs_amplitude is not None:
        bs.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_dBm_range"]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]

    ch = "SG4" if channel is None else channel
    lo_name = "SG4_LO" if lo_name is None else lo_name
    lo = lo_settings["q0"][serial_num][lo_name]

    lo_range = 500e6  # on either side of the lo
    lo_min = 1e9
    step = 200e6

    new_lo = bs_freq
    new_lo = round(new_lo / step) * step
    print(f"Trying new LO frequency: {new_lo/1e9} GHz")

    if new_lo < lo_min:
        print("Warning: LO frequency is too low. Adjusting to 1 GHz.")
        new_lo = lo_min

    lo = new_lo
    lo_settings["q0"][serial_num][lo_name] = lo
    print(f"Warning: LO ({lo_name}) frequency changed to {lo/1e9} GHz")
    lo_change = True

    if swp_amp:
        bs_amplitude = swp_param
    else:
        bs_length = swp_param

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):

        with exp.sweep(
            uid="time_or_amp_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
        ):

            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(uid="bs", play_after="ge_excitation", on_system_grid=True):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            with exp.section(uid="readout", play_after="bs", on_system_grid=True):
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
