from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    SectionAlignment,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def rabi_ef_chevron(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="rabi_ef",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(
        uid="sweep_param", start=40e-9, stop=400e-9, count=11
    ),
    acquisition_type=AcquisitionType.INTEGRATION,
    amplitude_rabi=False,
    time_rabi=False,
    rotate_ro=False,
    thresholds=None,
    pulse_type="const",  # "const" or "gaussian"
    play_ge=True,  # Whether to play the ge transition pulse
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param",
        start=4.1e9,
        stop=4.2e9,
        count=101,
    ),
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ef_X180 = qubit_params_module.ef_X180
    ge_X180 = qubit_params_module.ge_X180

    lo = lo_settings["q0"][serial_num]['SG0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    if not (time_rabi ^ amplitude_rabi):
        raise ValueError(
            "Please select either time_rabi or amplitude_rabi to be True and the other to be False."
        )


    with exp.sweep(
                uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True
            ):
        with exp.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            acquisition_type=acquisition_type,
        ): 
            with exp.sweep(uid="spect_sweep", parameter=freq_swp):
            
                with exp.section(uid="ge_transition", alignment=SectionAlignment.RIGHT):
                    exp.play(
                        signal="qb_drive",
                        pulse=ge_X180,
                        amplitude=ge_X180.amplitude if play_ge else 0,
                    )
                with exp.section(
                    uid="ef_transition", play_after="ge_transition", on_system_grid=True
                ):
                    if time_rabi:
                        exp.play(signal="qb_ef_drive", pulse=ef_X180, length=swp_param)
                    elif amplitude_rabi:
                        exp.play(
                            signal="qb_ef_drive", pulse=ef_X180, amplitude=swp_param
                        )
                with exp.section(uid="eg_transition", play_after="ef_transition"):
                    exp.play(signal="qb_drive", pulse=ge_X180)
                with exp.section(uid="readout", play_after="eg_transition"):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        handle="ac_0",
                        reset_delay= 2 * qubit_parameters["q0"]["reset_delay"],
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
    sig_freq_map[serial_num]["SG0"]["qb_ef_drive"]["frequency"] = freq_swp

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

    return exp
