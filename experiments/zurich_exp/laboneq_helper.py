import os
import pickle
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython.display import clear_output
from laboneq.simple import *

sys.path.append(os.getcwd())
import json

# sys.path.append(r'C:\_Lib\python\clab\experiments\qick_exp\exp_code')


# from helper_files.fitting_helper import *
def get_dict(data, name):
    """
    Convert Zurich data object to a dictionary
    """
    xpts = list(data.get_axis(name)[0])
    amp = data.get_data(name)
    amp_real = list(amp.real)
    amp_imag = list(amp.imag)

    return {"xpts": xpts, "avgi": amp_real, "avgq": amp_imag}


# from helper_files.fitting_helper import *
def get_dict_custom(xpts=None, amp=None):
    """
    Convert Zurich data object to a dictionary
    """
    amp_real = list(amp.real)
    amp_imag = list(amp.imag)

    return {"xpts": xpts, "avgi": amp_real, "avgq": amp_imag}


def get_dict_2D(data, name):
    """
    Convert Zurich data object to a dictionary
    """
    xpts = list(data.get_axis(name)[0])
    ypts = list(data.get_axis(name)[1])
    amp = data.get_data(name)
    amp_real = list(amp.real)
    amp_imag = list(amp.imag)

    return {"xpts": xpts, "ypts": ypts, "avgi": amp_real, "avgq": amp_imag}


def get_dict_2D_custom(xpts=None, ypts=None, amp=None):
    """
    Convert Zurich data object to a dictionary
    """

    amp_real = list(amp.real)
    amp_imag = list(amp.imag)

    return {"xpts": xpts, "ypts": ypts, "avgi": amp_real, "avgq": amp_imag}


# def save_data(data_path, filename, arrays={}, pts={}, save_config=True, config=None):
#     """
#     config is a dicionary with {"lo_settings": lo_settings, "qubit_parameters": qubit_parameters}
#     arrays: dictionary of the arrays to save; key is name to save array as and value is the array
#     pts: dictionary of pts to save; same format as arrays
#     """
#     file_path = data_path + "\\" + get_next_filename(data_path, filename, ".h5")

#     with SlabFile(file_path, "a") as f:
#         for i in arrays:
#             f.append_line(i, arrays[i])
#         for i in pts:
#             f.append_pt(i, pts[i])
#         if save_config:
#             f.attrs["config"] = json.dumps(config)
#     print("File saved at", file_path)


def get_next_filename(data_path, file_name, file_type="h5"):
    pattern = re.compile(rf"^(\d+)_({re.escape(file_name)})\.{re.escape(file_type)}$")
    max_index = -1
    for entry in os.scandir(data_path):
        match = pattern.match(entry.name)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    return f"{max_index + 1:05d}_{file_name}.{file_type}"


def save_data(data_path, file_name, data, config=None):
    file_path = os.path.join(data_path, get_next_filename(data_path, file_name, "h5"))
    with h5py.File(file_path, "w") as f:
        xpts = data.get("xpts", None)
        avgi = data.get("avgi", None)
        avgq = data.get("avgq", None)
        f.create_dataset("xpts", data=xpts)
        f.create_dataset("avgi", data=avgi)
        f.create_dataset("avgq", data=avgq)
        if "ypts" in data:
            ypts = data.get("ypts", None)
            f.create_dataset("ypts", data=ypts)
        if config:
            f.attrs["config"] = json.dumps(config)
    print("File saved at", file_path)


def save_calibrated_data(data_path, file_name, data, config=None):
    file_path = os.path.join(data_path, get_next_filename(data_path, file_name, "h5"))
    with h5py.File(file_path, "w") as f:
        xpts = data.get("xpts", None)
        P_e = data.get("P_e", None)
        avgi = data.get("avgi", None)
        avgq = data.get("avgq", None)
        r_g = data.get("r_g", None)
        r_e = data.get("r_e", None)
        r_gp = data.get("r_gp", None)
        r_ep = data.get("r_ep", None)
        avgi_rot = data.get("avgi_rot", None)
        avgq_rot = data.get("avgq_rot", None)
        theta = data.get("theta", None)

        f.create_dataset("xpts", data=xpts)
        f.create_dataset("P_e", data=P_e)
        f.create_dataset("avgi", data=avgi)
        f.create_dataset("avgq", data=avgq)
        f.create_dataset("r_g", data=r_g)
        f.create_dataset("r_e", data=r_e)
        f.create_dataset("r_gp", data=r_gp)
        f.create_dataset("r_ep", data=r_ep)
        f.create_dataset("avgi_rot", data=avgi_rot)
        f.create_dataset("avgq_rot", data=avgq_rot)
        f.create_dataset("theta", data=theta)

        if "ypts" in data:
            ypts = data.get("ypts", None)
            if ypts is not None:
                f.create_dataset("ypts", data=ypts)
        if "zpts" in data:
            zpts = data.get("zpts", None)
            if zpts is not None:
                f.create_dataset("zpts", data=zpts)

        if config:
            f.attrs["config"] = json.dumps(config)

    print("File saved at", file_path)


def save_results(
    results=None, data_path=None, filename="lenrabi", calibrate=False, save_config=True
):

    data = get_dict(results, "ac_0")

    if not calibrate:
        save_data(data_path=data_path, filename=filename, arrays=data, save_config=True)

    else:
        data_calib = get_dict(results, "ac_0_calib")

        # combine data and data_calib dictionaries
        data_combined = {}
        for key in data:
            data_combined[key] = np.concatenate((data[key], data_calib[key]), axis=0)

        save_data(
            data_path=data_path,
            filename=filename,
            arrays=data_combined,
            save_config=save_config,
        )


# function for generating YAML descriptor files
# due to inconvenience in defining open channels for each device, we will just be generating maximal number of signal lines possible.
def generate_descriptor_yaml(
    filename="laboneq_helper/default_descriptor.yaml",
    devices={"PQSC": [], "HDAWG": [], "SHFQC": [], "SHFQA": [], "SHFSG": []},
    n_qubits=1,
):
    descriptor = {"instruments": {}, "connections": {}}
    pqsc_present = False
    # deal with PQSC before other devices.
    if "PQSC" in devices.keys():
        pqsc_present = True
        pqsc_uid = "PQSC_" + devices["PQSC"]["serial"].lower()
        descriptor["instruments"]["PQSC"] = [
            {"address": devices["PQSC"]["serial"].lower(), "uid": pqsc_uid}
        ]
        descriptor["connections"][pqsc_uid] = []
        if ("external_clock" in devices["PQSC"].keys()) and devices["PQSC"][
            "external_clock"
        ]:
            descriptor["connections"][pqsc_uid].append("external_clock_signal")
        else:
            descriptor["connections"][pqsc_uid].append("internal_clock_signal")
    # deal with the remaining devices
    for key in devices.keys():
        if key != "PQSC":
            descriptor["instruments"][key] = []
            for device in devices[key]:
                # setup instruments section
                dev_uid = key + "_" + device["serial"].lower()
                descriptor["instruments"][key].append(
                    {"address": device["serial"].lower(), "uid": dev_uid}
                )
                if "usb" in device.keys():
                    if device["usb"]:
                        descriptor["instruments"][key][-1]["interface"] = "usb"
                # take care of zsync connections
                if ("zsync" in device.keys()) and pqsc_present:
                    descriptor["connections"][pqsc_uid].append(
                        {"to": dev_uid, "port": f'ZSYNCS/{device["zsync"]}'}
                    )
                elif not (not ("zsync" in device.keys()) and not (pqsc_present)):
                    print(
                        f"something wrong with zsync PQSC is {pqsc_present}, and "
                        + dev_uid
                        + f" has zsync {'zsync' in device}"
                    )
                    return
                # set up total number of channels and logical signals to enable in connections.
                if key == "HDAWG":
                    ch_setting = {
                        "total_sg_channels": 8,
                        "logical_signals_per_channel": 1,
                        "total_qa_channels": 0,
                        "ro_multiplex": 0,
                    }
                elif key == "SHFQC":
                    ch_setting = {
                        "total_sg_channels": 6,
                        "logical_signals_per_channel": 8,
                        "total_qa_channels": 1,
                        "ro_multiplex": 16,
                    }
                elif key == "SHFSG":
                    ch_setting = {
                        "total_sg_channels": 8,
                        "logical_signals_per_channel": 8,
                        "total_qa_channels": 0,
                        "ro_multiplex": 0,
                    }
                elif key == "SHFQA":
                    ch_setting = {
                        "total_sg_channels": 0,
                        "logical_signals_per_channel": 0,
                        "total_qa_channels": 4,
                        "ro_multiplex": 16,
                    }
                else:
                    print("Invalid Device type intered: ", key)
                    return
                for ch_key in ch_setting.keys():
                    if ch_key in device.keys():
                        ch_setting[ch_key] = min(ch_setting[ch_key], device[ch_key])
                # setup connections section
                descriptor["connections"][dev_uid] = []
                if "external_clock" in device.keys():
                    if device["external_clock"]:
                        descriptor["connections"][dev_uid].append(
                            "external_clock_signal"
                        )
                for qi in range(n_qubits):
                    for j in range(ch_setting["total_sg_channels"]):
                        for k in range(ch_setting["logical_signals_per_channel"]):
                            if key == "HDAWG":
                                descriptor["connections"][dev_uid].append(
                                    {
                                        "rf_signal": f'q{qi}/{device["serial"].lower()}_drive_{j}_{k}_line',
                                        "ports": f"SIGOUTS/{j}",
                                    }
                                )
                            elif key[:3] == "SHF":
                                descriptor["connections"][dev_uid].append(
                                    {
                                        "iq_signal": f'q{qi}/{device["serial"].lower()}_drive_{j}_{k}_line',
                                        "ports": f"SGCHANNELS/{j}/OUTPUT",
                                    }
                                )
                    for m in range(ch_setting["total_qa_channels"]):
                        for n in range(ch_setting["ro_multiplex"]):
                            descriptor["connections"][dev_uid].append(
                                {
                                    "iq_signal": f'q{qi}/{device["serial"].lower()}_measure_{m}_{n}_line',
                                    "ports": f"QACHANNELS/{m}/OUTPUT",
                                }
                            )
                            descriptor["connections"][dev_uid].append(
                                {
                                    "acquire_signal": f'q{qi}/{device["serial"].lower()}_acquire_{m}_{n}_line',
                                    "ports": f"QACHANNELS/{m}/INPUT",
                                }
                            )

    with open(filename, "w") as yaml_file:
        yaml.safe_dump(descriptor, yaml_file, sort_keys=False)
    return filename


# # version prior to 2024 10 31
# # uncomment and use it if needed
# # function for generating mapping and calibration
# def default_signal_map_and_calibration(sig_freq_map, metadata, qubit_index = "q0"):
#     exp_calibration = Calibration()
#     sig_map = {}
#     qa_pair = ["measure", "acquire"]
#     device_setup = metadata["device_setup"]
#     lo_settings = metadata["lo_settings"]
#     qubit_parameters = metadata["qubit_parameters"]

#     # find device
#     for devnum in sig_freq_map.keys():
#         # Signal Mapping
#         for ch in sig_freq_map[devnum]:
#             if ch[:2].upper() == "SG":
#                 for i in range(len(sig_freq_map[devnum][ch])):
#                     sig_map[sig_freq_map[devnum][ch][i]["logical_signal"]] = device_setup.logical_signal_groups[qubit_index].logical_signals[f"{devnum.lower()}_drive_{ch[-1]}_{i}_line"]
#             elif ch[:2].upper() == "QA":
#                 for i in range(len(sig_freq_map[devnum][ch])):
#                     for j in range(len(qa_pair)):
#                         sig_map[sig_freq_map[devnum][ch][i]["logical_signal"][j]] = device_setup.logical_signal_groups[qubit_index].logical_signals[f"{devnum.lower()}_{qa_pair[j]}_{ch[-1]}_{i}_line"]

#     # Calibration
#     for devnum in sig_freq_map.keys():
#         for ch in sig_freq_map[devnum]:
#             added_outputs = []
#             ch_num = int(ch[-1])
#             # SG channels
#             if ch[:2].upper() == "SG":
#                 for i in range(len(sig_freq_map[devnum][ch])):
#                     # Setting output router
#                     if "route" in sig_freq_map[devnum][ch][i].keys():
#                         for _ in sig_freq_map[devnum][ch][i]["route"]:
#                             added_outputs.append(OutputRoute(source=sig_map[_["src"]],amplitude_scaling=_["amp_scaling"],phase_shift=_["ph_shift"]))
#                     # Handling when LO not defined. For SHF, LO better be provided. This exception handling is for HDAWG
#                     try:
#                         sg_LO = Oscillator(uid = f"{devnum.lower()}_{ch[:2].upper()}{ch_num-ch_num%2}_lo", frequency = lo_settings[qubit_index][devnum][ch[:2].upper()+str(ch_num-ch_num%2)+'_LO'])
#                     except:
#                         sg_LO = None
#                     exp_calibration[sig_freq_map[devnum][ch][i]["logical_signal"]] = SignalCalibration(
#                         oscillator = Oscillator(frequency = sig_freq_map[devnum][ch][i]["frequency"], #uid = 'ch'+str(i)+'_'+str(j)+'_osc',
#                             modulation_type=ModulationType.HARDWARE
#                         ),
#                         local_oscillator = sg_LO,
#                         port_mode = PortMode.LF if (sg_LO == None or lo_settings[qubit_index][devnum][ch[:2].upper()+str(ch_num-ch_num%2)+'_LO'] == 0) else None ,
#                         range = 0 if "range" not in sig_freq_map[devnum][ch][i].keys() else sig_freq_map[devnum][ch][i]["range"],
#                         added_outputs = added_outputs,
#                         automute = False if (("automute" not in sig_freq_map[devnum][ch][i].keys()) or lo_settings[qubit_index][devnum][ch[:2].upper()+str(ch_num-ch_num%2)+'_LO'] == 0) else sig_freq_map[devnum][ch][i]["automute"],
#                     )
#             # QA channels
#             if ch[:2].upper() == "QA":
#                 for i in range(len(sig_freq_map[devnum][ch])):
#                     spect = (("spectroscopy" in sig_freq_map[devnum][ch][i].keys()) and sig_freq_map[devnum][ch][i]["spectroscopy"])
#                     rotate_ro = (("rotate_ro" in sig_freq_map[devnum][ch][i].keys()) and sig_freq_map[devnum][ch][i]["rotate_ro"])
#                     thresholds = [None,None] if "thresholds" not in sig_freq_map[devnum][ch][i].keys() else [None,sig_freq_map[devnum][ch][i]["thresholds"]]
#                     # Set specific settings for SPECTROSCOPY mode
#                     modtype = ModulationType.HARDWARE if spect else ModulationType.SOFTWARE
#                     meas_port_delay = 0 if spect else qubit_parameters[qubit_index]['ro_delay']
#                     # add an offset between the readout pulse and the start of the data acquisition - to compensate for round-trip and ring-up time of readout pulse
#                     acq_port_delay = 0 if spect else qubit_parameters[qubit_index]["ro_delay"] + qubit_parameters[qubit_index]["ro_int_delay"]
#                     port_delay = [meas_port_delay, acq_port_delay]
#                     for j in range(len(qa_pair)):
#                         exp_calibration[sig_freq_map[devnum][ch][i]["logical_signal"][j]] = SignalCalibration(
#                             # Oscillator settings - Dependant on SPECTROSCOPY mode, readout rotation needed for discrimination kernal calculation
#                             oscillator = Oscillator(frequency = sig_freq_map[devnum][ch][i]["frequency"], modulation_type=modtype) if ((not (spect or rotate_ro)) or (j == 0)) else None,
#                             local_oscillator = Oscillator(uid = f"{devnum.lower()}_{ch.upper()}_lo",frequency = lo_settings[qubit_index][devnum][ch.upper()+'_LO']) if ((not (spect or rotate_ro)) or (j == 0)) else None,
#                             port_delay = port_delay[j],
#                             port_mode = PortMode.LF if lo_settings[qubit_index][devnum][ch.upper()+'_LO'] == 0 else None,
#                             range = 0 if "range" not in sig_freq_map[devnum][ch][i].keys() else sig_freq_map[devnum][ch][i]["range"][j],
#                             automute = False if ((("automute" not in sig_freq_map[devnum][ch][i].keys()) or j == 1) or lo_settings[qubit_index][devnum][ch.upper()+'_LO'] == 0) else sig_freq_map[devnum][ch][i]["automute"],
#                             threshold = thresholds[j],
#                         )
#     return exp_calibration, sig_map


# version after 2024 10 31
# function for generating mapping and calibration
def default_signal_map_and_calibration(sig_freq_map, metadata, qubit_index="q0"):
    exp_calibration = Calibration()
    sig_map = {}
    device_setup = metadata["device_setup"]
    lo_settings = metadata["lo_settings"]
    qubit_parameters = metadata["qubit_parameters"]

    # find device
    for devnum in sig_freq_map.keys():
        # Signal Mapping
        for ch in sig_freq_map[devnum]:
            if ch[:2].upper() == "SG":
                i = 0
                for key in sig_freq_map[devnum][ch].keys():
                    sig_map[key] = device_setup.logical_signal_groups[
                        qubit_index
                    ].logical_signals[f"{devnum.lower()}_drive_{ch[-1]}_{i}_line"]
                    i += 1
            elif ch[:2].upper() == "QA":
                i = 0
                for key in sig_freq_map[devnum][ch].keys():
                    qa_pair = key.split("/")
                    sig_map[qa_pair[0]] = device_setup.logical_signal_groups[
                        qubit_index
                    ].logical_signals[f"{devnum.lower()}_measure_{ch[-1]}_{i}_line"]
                    sig_map[qa_pair[1]] = device_setup.logical_signal_groups[
                        qubit_index
                    ].logical_signals[f"{devnum.lower()}_acquire_{ch[-1]}_{i}_line"]
                    i += 1

    # Calibration
    for devnum in sig_freq_map.keys():
        for ch in sig_freq_map[devnum]:
            added_outputs = []
            ch_num = int(ch[-1])
            # SG channels
            if ch[:2].upper() == "SG":
                for key in sig_freq_map[devnum][ch].keys():
                    # Setting output router
                    if "route" in sig_freq_map[devnum][ch][key].keys():
                        for _ in sig_freq_map[devnum][ch][key]["route"]:
                            added_outputs.append(
                                OutputRoute(
                                    source=sig_map[_["src"]],
                                    amplitude_scaling=_["amp_scaling"],
                                    phase_shift=_["ph_shift"],
                                )
                            )
                    # Handling when LO not defined. For SHF, LO better be provided. This exception handling is for HDAWG
                    try:
                        sg_LO = Oscillator(
                            uid=f"{devnum.lower()}_{ch[:2].upper()}{ch_num-ch_num%2}_lo",
                            frequency=lo_settings[qubit_index][devnum][
                                ch[:2].upper() + str(ch_num - ch_num % 2) + "_LO"
                            ],
                        )
                    except:
                        sg_LO = None
                    exp_calibration[key] = SignalCalibration(
                        oscillator=Oscillator(
                            frequency=sig_freq_map[devnum][ch][key][
                                "frequency"
                            ],  # uid = 'ch'+str(i)+'_'+str(j)+'_osc',
                            modulation_type=(
                                ModulationType.HARDWARE
                                if (
                                    "SW_override"
                                    not in sig_freq_map[devnum][ch][key].keys()
                                    or not sig_freq_map[devnum][ch][key]["SW_override"]
                                )
                                else ModulationType.SOFTWARE
                            ),
                        ),
                        local_oscillator=sg_LO,
                        port_mode=(
                            PortMode.LF
                            if (
                                sg_LO == None
                                or lo_settings[qubit_index][devnum][
                                    ch[:2].upper() + str(ch_num - ch_num % 2) + "_LO"
                                ]
                                == 0
                            )
                            else None
                        ),
                        range=(
                            0
                            if "range" not in sig_freq_map[devnum][ch][key].keys()
                            else sig_freq_map[devnum][ch][key]["range"]
                        ),
                        added_outputs=added_outputs,
                        automute=(
                            False
                            if (
                                ("automute" not in sig_freq_map[devnum][ch][key].keys())
                                or lo_settings[qubit_index][devnum][
                                    ch[:2].upper() + str(ch_num - ch_num % 2) + "_LO"
                                ]
                                == 0
                            )
                            else sig_freq_map[devnum][ch][key]["automute"]
                        ),
                    )
            # QA channels
            if ch[:2].upper() == "QA":
                for key in sig_freq_map[devnum][ch].keys():
                    qa_pair = key.split("/")
                    spect = (
                        "spectroscopy" in sig_freq_map[devnum][ch][key].keys()
                    ) and sig_freq_map[devnum][ch][key]["spectroscopy"]
                    rotate_ro = (
                        "rotate_ro" in sig_freq_map[devnum][ch][key].keys()
                    ) and sig_freq_map[devnum][ch][key]["rotate_ro"]
                    thresholds = (
                        [None, None]
                        if "thresholds" not in sig_freq_map[devnum][ch][key].keys()
                        else [None, sig_freq_map[devnum][ch][key]["thresholds"]]
                    )
                    # Set specific settings for SPECTROSCOPY mode
                    modtype = (
                        ModulationType.HARDWARE if spect else ModulationType.SOFTWARE
                    )
                    meas_port_delay = (
                        0 if spect else qubit_parameters[qubit_index]["ro_delay"]
                    )
                    # add an offset between the readout pulse and the start of the data acquisition - to compensate for round-trip and ring-up time of readout pulse
                    # maybe okay not to force it to 0 for spectroscopy mode.
                    acq_port_delay = (
                        0
                        if spect
                        else qubit_parameters[qubit_index]["ro_delay"]
                        + qubit_parameters[qubit_index]["ro_int_delay"]
                    )
                    port_delay = [meas_port_delay, acq_port_delay]
                    for j in range(len(qa_pair)):  # len of qa pair must be 2
                        exp_calibration[qa_pair[j]] = SignalCalibration(
                            # Oscillator settings - Dependant on SPECTROSCOPY mode, readout rotation needed for discrimination kernal calculation
                            oscillator=(
                                Oscillator(
                                    frequency=sig_freq_map[devnum][ch][key][
                                        "frequency"
                                    ],
                                    modulation_type=modtype,
                                )
                                if ((not (spect or rotate_ro)) or (j == 0))
                                else None
                            ),
                            local_oscillator=(
                                Oscillator(
                                    uid=f"{devnum.lower()}_{ch.upper()}_lo",
                                    frequency=lo_settings[qubit_index][devnum][
                                        ch.upper() + "_LO"
                                    ],
                                )
                                if ((not (spect or rotate_ro)) or (j == 0))
                                else None
                            ),
                            # local_oscillator = Oscillator(uid = f"{devnum.lower()}_{ch.upper()}_lo",frequency = lo_settings[qubit_index][devnum][ch.upper()+'_LO']) if qa_pair[j] == 'measure' else None,
                            port_delay=port_delay[j],
                            port_mode=(
                                PortMode.LF
                                if lo_settings[qubit_index][devnum][ch.upper() + "_LO"]
                                == 0
                                else None
                            ),
                            range=(
                                0
                                if "range" not in sig_freq_map[devnum][ch][key].keys()
                                else sig_freq_map[devnum][ch][key]["range"][j]
                            ),
                            automute=(
                                False
                                if (
                                    (
                                        (
                                            "automute"
                                            not in sig_freq_map[devnum][ch][key].keys()
                                        )
                                        or j == 1
                                    )
                                    or lo_settings[qubit_index][devnum][
                                        ch.upper() + "_LO"
                                    ]
                                    == 0
                                )
                                else sig_freq_map[devnum][ch][key]["automute"]
                            ),
                            threshold=thresholds[j],
                        )
    return exp_calibration, sig_map


def save_kernels_and_thresholds_to_csv(filepath, kernels, thresholds, tag=None):
    for _ in kernels:
        np.savetxt(
            filepath + f"/kernels_{tag}.csv",
            [_.samples for _ in kernels],
            delimiter=",",
            fmt="%s",
        )
        np.savetxt(
            filepath + f"/thresholds_{tag}.csv", thresholds, delimiter=",", fmt="%s"
        )
    return


def load_kernels_and_thresholds_from_csv(filepath, tag=None):
    data_from_csv = np.loadtxt(
        filepath + f"/kernels_{tag}.csv", delimiter=",", dtype=str
    )
    thresholds = np.loadtxt(
        filepath + f"/thresholds_{tag}.csv", delimiter=",", dtype=str
    )

    states = ["g", "e", "f"]
    kernels = []
    if data_from_csv.ndim == 1:
        kernels.append(
            pulse_library.PulseSampledComplex(
                uid=f"kernel_{states[0]}", samples=data_from_csv.astype(np.complex128)
            )
        )
        thresholds = [np.float64(thresholds)]
    else:
        for i in range(len(data_from_csv)):
            kernels.append(
                pulse_library.PulseSampledComplex(
                    uid=f"kernel_{states[i]}",
                    samples=data_from_csv[i].astype(np.complex128),
                )
            )
        thresholds = [np.float64(_) for _ in thresholds]
    return kernels, thresholds


def parsing_single_alice_bob_cav_and_sb_transitions(
    alice_or_bob,
    cav_alice=None,
    cav_bob=None,
    sb_pulses=None,
    max_fock_state=1,
):
    """
    outputs cav_name(str), transitions(list), sb_lines_parsed(dict), sb_pulses_parsed(dict),
    cav_line(str), cav_pulse(pulse) in this order.
    """
    if not (("a" in alice_or_bob.lower()) ^ ("b" in alice_or_bob.lower())):
        raise ValueError("choose either alice or bob, not both or neither.")
    if "a" in alice_or_bob.lower():
        cav_name = "alice"
    else:
        cav_name = "bob"
    cav_pulse = cav_alice if cav_name == "alice" else cav_bob
    cav_line = f"cav_drive_{cav_name}" if cav_pulse != None else None
    transitions = []
    for i in range(max_fock_state):
        transitions.append(f"f{i}g{i+1}")
    sb_pulses_parsed = None
    sb_lines_parsed = None
    if sb_pulses != None:
        sb_pulses_parsed = {}
        sb_lines_parsed = {}
        for transition in transitions:
            try:
                sb_pulses_parsed[transition] = sb_pulses[cav_name][transition]
                sb_lines_parsed[transition] = f"sb_drive_{cav_name}_{transition}"
            except KeyError:
                print(f"Warning: {cav_name, transition} not found in sb_pulses")

    return cav_name, transitions, sb_lines_parsed, sb_pulses_parsed, cav_line, cav_pulse


def convert_iq_swp_into_amp_phase_swp(i_sweep, q_sweep):
    if (type(i_sweep) == type(None)) or (type(q_sweep) == type(None)):
        raise ValueError("Both i_sweep and q_sweep must be provided.")
    z_sweep = []
    for q in q_sweep:
        for i in i_sweep:
            z_sweep.append(i + 1j * q)
    z_sweep = np.array(z_sweep)
    phase_swp = SweepParameter(uid="phase_swp", values=np.angle(z_sweep))
    amp_swp = SweepParameter(uid="amp_swp", values=np.abs(z_sweep))
    return amp_swp, phase_swp


def convert_iq_values_into_amp_phase_swp(iq_values):
    if (type(iq_values) == type(None)) or (len(iq_values) == 0):
        raise ValueError("iq_sweep must be provided.")
    phase_swp = SweepParameter(uid="phase_swp", values=np.angle(iq_values))
    amp_swp = SweepParameter(uid="amp_swp", values=np.abs(iq_values))
    return amp_swp, phase_swp


def update_sweep_plot(
    session,
    exp,
    fn_prefix="",
    tot_average_exponent: int | None = None,
    chunk_average_exponent: int | None = None,
    wrap_data: (
        int | None
    ) = None,  ## this is when you want to make a 2d plot out of 1d data with wrap_data as the width of the 2d data.
    plot_im=False,  ## allowed only in 1D sweep
    handle_2D: str | None = None,
    data_dump_list: list | None = None,
    press_enter: bool = True,
):
    if tot_average_exponent == None:
        tot_average_exponent = int(np.log2(exp.get_rt_acquire_loop().count))
    if chunk_average_exponent == None:
        chunk_average_exponent = (
            tot_average_exponent - 5 if tot_average_exponent > 5 else 0
        )
    if chunk_average_exponent > tot_average_exponent:
        raise ValueError(
            "chunk_average_exponent must be less than tot_average_exponent. chunk size is,",
            pow(2, chunk_average_exponent),
            " and total size is ",
            pow(2, tot_average_exponent),
        )
    exp.get_rt_acquire_loop().count = pow(2, chunk_average_exponent)
    compiled_exp = session.compile(exp)
    print(
        "Finished compiling. Press Enter to run the experiment.\n\
          The experiment will run for ",
        pow(2, tot_average_exponent),
        " times and\n\
          will update plot every ",
        pow(2, chunk_average_exponent),
        " runs",
    )
    print(
        f"{compiled_exp.estimated_runtime}s estimated runtime per chunk, excluding python, communication, and near-time operations overheads"
    )
    if press_enter:
        input()

    ##### compilation done. Now run the experiment and collect data
    handles = []
    result_axes = []
    averaged_data = {}
    data_dump = []
    repeat_chunk = pow(2, tot_average_exponent - chunk_average_exponent)
    for i in range(repeat_chunk):
        print(
            "Running chunk ",
            (i + 1) * pow(2, chunk_average_exponent),
            " of ",
            pow(2, tot_average_exponent),
        )
        data_exp_chunk = {}
        run_exp = session.run(compiled_exp)
        if len(handles) == 0:
            for key in run_exp.acquired_results.keys():
                handles.append(key)
            for result_axes_items in run_exp.get_axis(handles[0]):
                result_axes.append(result_axes_items)
            # result_axes.append(run_exp.get_axis(handles[0])[0])
        for handle in handles:
            data_exp_chunk[handle] = run_exp.get_data(handle)
            if i == 0:
                averaged_data[handle] = data_exp_chunk[handle]
            else:
                averaged_data[handle] = (
                    averaged_data[handle] * i + data_exp_chunk[handle]
                ) / (i + 1)
        data_dump.append(data_exp_chunk)
        if data_dump_list != None:
            data_dump_list.append(data_exp_chunk)
            fname = fn_prefix + "_" + exp.uid

            with open(fname, "wb") as handle_pickle:
                pickle.dump(
                    data_dump_list, handle_pickle, protocol=pickle.HIGHEST_PROTOCOL
                )

        ##### data collection for a single chunk is done. plot / update data
        ##### 1D plot case
        print(len(result_axes))
        if (len(result_axes) == 1) and (wrap_data == None):
            print("1D plot")
            fig, ax = plt.subplots(len(handles), 2, figsize=(12, 4 * len(handles)))
            ax = ax.flatten()
            fig.suptitle(
                f"run {(i+1)*pow(2,chunk_average_exponent)} of {pow(2, tot_average_exponent)}"
            )
            for j in range(len(handles)):
                if np.array(result_axes[0]).ndim == 1:
                    if plot_im:
                        ax[2 * j].plot(
                            result_axes[0],
                            np.imag(data_exp_chunk[handles[j]]),
                            ".C1-",
                            label="im",
                        )
                    ax[2 * j + 1].plot(
                        result_axes[0],
                        np.real(averaged_data[handles[j]]),
                        ".C0-",
                        label="real",
                    )
                    if plot_im:
                        ax[2 * j + 1].plot(
                            result_axes[0],
                            np.imag(averaged_data[handles[j]]),
                            ".C1-",
                            label="im",
                        )
                    ax[2 * j].plot(
                        result_axes[0],
                        np.real(data_exp_chunk[handles[j]]),
                        ".C0-",
                        label="real",
                    )
                else:
                    if plot_im:
                        ax[2 * j].plot(
                            np.imag(data_exp_chunk[handles[j]]), ".C1-", label="im"
                        )
                    ax[2 * j + 1].plot(
                        np.real(averaged_data[handles[j]]), ".C0-", label="real"
                    )
                    if plot_im:
                        ax[2 * j + 1].plot(
                            np.imag(averaged_data[handles[j]]), ".C1-", label="im"
                        )
                    ax[2 * j].plot(
                        np.real(data_exp_chunk[handles[j]]), ".C0-", label="real"
                    )
                ax[2 * j].set_title(f"Chunk Data, {handles[j]}")
                ax[2 * j + 1].set_title(f"Averaged Data, {handles[j]}")
                ax[2 * j].set_xlabel("Sweep parameter value")
                ax[2 * j + 1].set_xlabel("Sweep parameter value")
                ax[2 * j].set_ylabel("A (V * integrated sample #)")
                ax[2 * j + 1].set_ylabel("A (V * integrated sample #)")
                ax[2 * j].grid()
                ax[2 * j + 1].grid()
                ax[2 * j].legend()
                ax[2 * j + 1].legend()
        ##### 2D plot case
        elif ((len(result_axes) == 1) and (wrap_data != None)) or (
            len(result_axes) == 2
        ):
            print("2D plot")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax = ax.flatten()
            fig.suptitle(
                f"run {(i+1)*pow(2,chunk_average_exponent)} of {pow(2, tot_average_exponent)}"
            )
            if handle_2D == None:
                handle_2D = handles[-1]
            if len(result_axes) == 1:
                plot_array = np.reshape(
                    data_exp_chunk[handle_2D],
                    (int(len(data_exp_chunk[handle_2D]) / wrap_data), wrap_data),
                )
                plot_array_av = np.reshape(
                    averaged_data[handle_2D],
                    (int(len(averaged_data[handle_2D]) / wrap_data), wrap_data),
                )
            elif len(result_axes) == 2:
                plot_array = data_exp_chunk[handle_2D]
                plot_array_av = averaged_data[handle_2D]
            if plot_im:
                print("plotting imaginary part")
                ax[0].pcolormesh(np.imag(plot_array), cmap="RdBu_r")
                ax[1].pcolormesh(np.imag(plot_array_av), cmap="RdBu_r")
            else:
                print("plotting real part")
                ax[0].pcolormesh(np.real(plot_array), cmap="RdBu_r")
                ax[1].pcolormesh(np.real(plot_array_av), cmap="RdBu_r")
            cbar0 = plt.colorbar(ax[0].collections[0], ax=ax[0])
            cbar1 = plt.colorbar(ax[1].collections[0], ax=ax[1])
            ax[0].set_title(f"Chunk Data, {handle_2D}, imaginary: {plot_im}")
            ax[1].set_title(f"Averaged Data, {handle_2D}, imaginary: {plot_im}")
            ax[0].set_xlabel("Sweep parameter value")
            ax[1].set_xlabel("Sweep parameter value")
            ax[0].set_ylabel("A (V * integrated sample #)")
            ax[1].set_ylabel("A (V * integrated sample #)")
            ax[0].grid()
            ax[1].grid()
        else:
            raise ValueError(
                "Too many axes in the acquired results. Only 1 or 2 axes are allowed at this moment."
            )
        plt.tight_layout()
        clear_output(True)
        plt.show(fig)
        print(
            "Finished chunk ",
            (i + 1) * pow(2, chunk_average_exponent),
            " of ",
            pow(2, tot_average_exponent),
        )
    return data_dump, averaged_data, result_axes, handles


# def create_custom_kernel_threshold(run_exp, frequency, handles = ["ac_g", "ac_e", "ac_f"]):
#     kernels = []
#     thresholds = []
#     iq_data = {}
#     for handle in handles:
#         iq_data[handle] = run_exp.get_data(handle)
#     qsd_ge = qubit_state_discriminator(train_g_states=iq_data["ac_g"], train_e_states=iq_data["ac_e"])
#     qsd_ge.train()
#     qsd_gf = qubit_state_discriminator(train_g_states=iq_data["ac_g"], train_e_states=iq_data["ac_f"])
#     qsd_gf.train()
#     qsd_ef = qubit_state_discriminator(train_g_states=iq_data["ac_e"], train_e_states=iq_data["ac_f"])
#     qsd_ef.train()

#     avg_g = np.average(iq_data["ac_g"])
#     avg_e = np.average(iq_data["ac_e"])
#     avg_f = np.average(iq_data["ac_f"])

#     samples = np.arange(4000)
#     time = samples*0.5e-9
#     phase_offset_ge = qsd_ge.angle + np.pi
#     phase_offset_gf = qsd_gf.angle + np.pi

#     threshold_ge = - qsd_ge.threshold
#     threshold_gf = - qsd_gf.threshold
#     threshold_ef = - qsd_ef.threshold
#     print("angles and thresholds", qsd_ge.angle, qsd_gf.angle, qsd_ef.angle,)
#     print("angles and thresholds", threshold_ge, threshold_gf, threshold_ef)
#     ge_vector = avg_g - avg_e
#     gf_vector = avg_g - avg_f

#     amp_ge = np.abs(ge_vector)
#     amp_gf = np.abs(gf_vector)
#     print("amplitudes", amp_ge, amp_gf)

#     manual_kernel_samples_ge = amp_ge*np.exp(-1.j*(2*np.pi*time*frequency - phase_offset_ge))
#     manual_kernel_samples_gf = amp_gf*np.exp(-1.j*(2*np.pi*time*frequency - phase_offset_gf))

#     manual_kernel_ge = pulse_library.PulseSampledComplex(uid = 'manual_kernel_ge', samples = manual_kernel_samples_ge/ np.max(np.abs(manual_kernel_samples_ge)))
#     manual_kernel_gf = pulse_library.PulseSampledComplex(uid = 'manual_kernel_gf', samples = manual_kernel_samples_gf/ np.max(np.abs(manual_kernel_samples_gf)))

#     kernels = [manual_kernel_ge, manual_kernel_gf]
#     thresholds = [threshold_ge, threshold_gf, threshold_ef]

#     return kernels, thresholds
