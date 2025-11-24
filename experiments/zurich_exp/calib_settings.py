from laboneq.dsl.enums import AcquisitionType


def _transition_to_index(transition: str) -> int:
    """Return the integer index (the f-number) for a label like 'f2g3'."""
    try:
        f_part, g_part = transition.split("g", 1)
        if not f_part.startswith("f"):
            raise ValueError
        f_idx = int(f_part[1:])
        g_idx = int(g_part)
    except Exception as exc:  # noqa: BLE001 - make error message clearer downstream
        raise ValueError(f"Invalid transition label '{transition}'") from exc
    if g_idx - f_idx != 1:
        raise ValueError(f"Unsupported transition label '{transition}'")
    return f_idx


def _get_sb_value(
    qparams: dict, buffer_mode: str, transition: str, kind: str
) -> float:
    """Lookup sideband values using arrays with legacy fallback."""
    idx = _transition_to_index(transition)
    array_keys = {
        "freq": f"sb_{buffer_mode}_freqs",
        "flat_len": f"sb_{buffer_mode}_flat_lens",
        "ramp_len": f"sb_{buffer_mode}_ramp_lens",
        "amp": f"sb_{buffer_mode}_amps",
    }
    array_key = array_keys.get(kind)
    if array_key is not None:
        values = qparams.get(array_key)
        if isinstance(values, (list, tuple)) and idx < len(values):
            return values[idx]

    legacy_suffix = {
        "freq": "_freq",
        "flat_len": "_flat_len",
        "ramp_len": "_ramp_len",
        "amp": "_amp",
    }[kind]
    legacy_key = f"sb_{transition}_{buffer_mode}{legacy_suffix}"
    if legacy_key in qparams:
        return qparams[legacy_key]
    raise KeyError(
        f"Missing sideband parameter '{legacy_key}' (buffer_mode={buffer_mode}, transition={transition})"
    )


def _get_bs_value(
    qparams: dict, buffer_mode: str, index: int, kind: str
) -> float:
    """Lookup beamsplitter values using arrays with legacy fallback."""
    array_keys = {
        "freq": f"bs_{buffer_mode}_freqs",
        "flat_len": f"bs_{buffer_mode}_flat_lens",
        "ramp_len": f"bs_{buffer_mode}_ramp_lens",
        "range": f"bs_{buffer_mode}_dBm_ranges",
        "amp": f"bs_{buffer_mode}_amps",
    }
    array_key = array_keys.get(kind)
    if array_key is not None:
        values = qparams.get(array_key)
        if isinstance(values, (list, tuple)) and index < len(values):
            return values[index]

    legacy_suffix_map = {
        "freq": "_freq",
        "flat_len": "_flat_len",
        "ramp_len": "_ramp_len",
        "range": "_dBm_range",
        "amp": "_amp",
    }
    legacy_suffix = legacy_suffix_map[kind]
    legacy_keys = (
        f"bs{index}_{buffer_mode}{legacy_suffix}",
        f"sb_bs{index}_{buffer_mode}{legacy_suffix}",
    )
    for key in legacy_keys:
        if key in qparams:
            return qparams[key]
    # final fallback: allow a single shared range if present (e.g. sb_alice_dBm_range)
    if kind == "range":
        shared_key = f"sb_{buffer_mode}_dBm_range"
        if shared_key in qparams:
            return qparams[shared_key]
    raise KeyError(
        f"Missing beamsplitter parameter for buffer_mode={buffer_mode}, index={index}, kind={kind}"
    )


def create_default_map_and_calibration(
    exp,
    serial_num,
    qubit_parameters,
    lo_settings,
    max_fock_state=8,
    rotate_ro=False,
    thresholds=None,
    automute=True,
    sb_SW_override=False,
    qb_drive_SW_override=False,
):

    los = lo_settings["q0"][serial_num]
    ro_lo = los["QA0_LO"]
    qb_lo = los["SG0_LO"]
    cavity_lo = los["SG2_LO"]
    sb_lo = los["SG0_LO"]
    bs_lo = los["SG4_LO"]

    spectroscopy = False
    if exp.get_rt_acquire_loop().acquisition_type == AcquisitionType.SPECTROSCOPY:
        spectroscopy = True

    transitions = []
    for i in range(max_fock_state):
        transitions.append(f"f{i}g{i+1}")
    if max_fock_state > 8:
        raise ValueError(
            "Max fock state should be less than or equal to 8. use 'enable_router_for_sb' function for higher numbers."
        )

    sig_freq_map = {
        serial_num: {
            "SG0": {},
            "SG1": {},
            "SG2": {},
            "SG3": {},
            "SG4": {},
            "SG5": {},
            "QA0": {},
        },
    }
    if "qb_drive" in exp.signals:
        ch = "SG0"
        sig_freq_map[serial_num][ch]["qb_drive"] = {}
        sig_freq_map[serial_num][ch]["qb_drive"]["frequency"] = (
            qubit_parameters["q0"]["qb_freq"] - qb_lo
        )
        sig_freq_map[serial_num][ch]["qb_drive"]["range"] = qubit_parameters["q0"][
            "qb_drive_dBm_range"
        ]
        sig_freq_map[serial_num][ch]["qb_drive"]["automute"] = automute
        sig_freq_map[serial_num][ch]["qb_drive"]["SW_override"] = qb_drive_SW_override

    if "qb_drive_resolved" in exp.signals:
        ch = "SG0"
        sig_freq_map[serial_num][ch]["qb_drive_resolved"] = {}
        sig_freq_map[serial_num][ch]["qb_drive_resolved"]["frequency"] = (
            qubit_parameters["q0"]["qb_resolved_freq"] - qb_lo
        )
        sig_freq_map[serial_num][ch]["qb_drive_resolved"]["range"] = qubit_parameters["q0"][
            "qb_drive_resolved_dBm_range"
        ]
        sig_freq_map[serial_num][ch]["qb_drive_resolved"]["automute"] = automute
        sig_freq_map[serial_num][ch]["qb_drive_resolved"]["SW_override"] = qb_drive_SW_override

    if "qb_ef_drive" in exp.signals:
        ch = "SG0"
        sig_freq_map[serial_num][ch]["qb_ef_drive"] = {}
        sig_freq_map[serial_num][ch]["qb_ef_drive"]["frequency"] = (
            qubit_parameters["q0"]["qb_ef_freq"] - qb_lo
        )
        sig_freq_map[serial_num][ch]["qb_ef_drive"]["range"] = qubit_parameters["q0"][
            "qb_drive_dBm_range"
        ]
        sig_freq_map[serial_num][ch]["qb_ef_drive"]["automute"] = automute
        sig_freq_map[serial_num][ch]["qb_ef_drive"][
            "SW_override"
        ] = qb_drive_SW_override

    if "cav_drive_alice" in exp.signals:
        ch = "SG3"
        sig_freq_map[serial_num][ch]["cav_drive_alice"] = {}
        sig_freq_map[serial_num][ch]["cav_drive_alice"]["frequency"] = (
            qubit_parameters["q0"]["cav_alice_freq"] - cavity_lo
        )
        sig_freq_map[serial_num][ch]["cav_drive_alice"]["range"] = qubit_parameters[
            "q0"
        ]["cav_alice_dBm_range"]
        sig_freq_map[serial_num][ch]["cav_drive_alice"]["automute"] = automute

    if "cav_drive_bob" in exp.signals:
        ch = "SG3"
        sig_freq_map[serial_num][ch]["cav_drive_bob"] = {}
        sig_freq_map[serial_num][ch]["cav_drive_bob"]["frequency"] = (
            qubit_parameters["q0"]["cav_bob_freq"] - cavity_lo
        )
        sig_freq_map[serial_num][ch]["cav_drive_bob"]["range"] = qubit_parameters["q0"][
            "cav_bob_dBm_range"
        ]
        sig_freq_map[serial_num][ch]["cav_drive_bob"]["automute"] = automute

    # beamsplitter signals
    for buffer_mode in ("alice", "bob"):
        prefix = f"sb_drive_{buffer_mode}_bs"
        for signal_name in (sig for sig in exp.signals if sig.startswith(prefix)):
            try:
                bs_index = int(signal_name[len(prefix) :])
            except ValueError:
                continue
            ch = "SG4"
            sig_freq_map[serial_num][ch][signal_name] = {}
            sig_freq_map[serial_num][ch][signal_name]["frequency"] = (
                _get_bs_value(qubit_parameters["q0"], buffer_mode, bs_index, "freq")
                - bs_lo
            )
            sig_freq_map[serial_num][ch][signal_name]["range"] = _get_bs_value(
                qubit_parameters["q0"], buffer_mode, bs_index, "range"
            )
            sig_freq_map[serial_num][ch][signal_name]["automute"] = automute
            sig_freq_map[serial_num][ch][signal_name]["SW_override"] = sb_SW_override
            sig_freq_map[serial_num][ch][signal_name]["length"] = (
                _get_bs_value(qubit_parameters["q0"], buffer_mode, bs_index, "flat_len")
            )

    # sideband signals
    for transition in transitions:
        for buffer_mode in ("alice", "bob"):
            signal_name = f"sb_drive_{buffer_mode}_{transition}"
            if signal_name not in exp.signals:
                continue
            ch = "SG1"
            sig_freq_map[serial_num][ch][signal_name] = {}
            sig_freq_map[serial_num][ch][signal_name]["frequency"] = (
                _get_sb_value(qubit_parameters["q0"], buffer_mode, transition, "freq")
                - sb_lo
            )
            sig_freq_map[serial_num][ch][signal_name]["range"] = qubit_parameters[
                "q0"
            ][f"sb_{buffer_mode}_dBm_range"]
            sig_freq_map[serial_num][ch][signal_name]["length"] = _get_sb_value(
                qubit_parameters["q0"], buffer_mode, transition, "flat_len"
            )
            sig_freq_map[serial_num][ch][signal_name]["automute"] = automute
            sig_freq_map[serial_num][ch][signal_name]["SW_override"] = sb_SW_override

    # measure and acquire should be added to signals in pairs.
    if ("measure" in exp.signals) or ("acquire" in exp.signals):
        ch = "QA0"
        sig_freq_map[serial_num][ch]["measure/acquire"] = {}
        sig_freq_map[serial_num][ch]["measure/acquire"]["frequency"] = (
            qubit_parameters["q0"]["ro_freq"] - ro_lo
        )
        sig_freq_map[serial_num][ch]["measure/acquire"]["range"] = [
            qubit_parameters["q0"]["ro_drive_dBm_range"],
            qubit_parameters["q0"]["ro_acq_dBm_range"],
        ]
        sig_freq_map[serial_num][ch]["measure/acquire"]["spectroscopy"] = spectroscopy
        sig_freq_map[serial_num][ch]["measure/acquire"]["rotate_ro"] = rotate_ro
        sig_freq_map[serial_num][ch]["measure/acquire"]["thresholds"] = thresholds
        sig_freq_map[serial_num][ch]["measure/acquire"]["automute"] = automute

    return sig_freq_map


def enable_router_for_sb(
    exp,
    serial_num,
    qubit_parameters,
    sig_freq_map,
    max_fock_state=32,
    alice_or_bob="a",
    unused_channels=[
        "SG1",
        "SG5",
        "SG2",
    ],  # note that SG5 and SG2 are initially allocated for bob. If bob is used, use ['SG1','SG3','SG4']
    sb_SW_override=False,
    automute=True,
):

    if max_fock_state < 9:
        raise ValueError(
            "do not use 'enable_router_for_sb' function for max_fock_state less than or equal to 8."
        )
    if max_fock_state > ((len(unused_channels) + 1) * 8):
        raise ValueError(
            f"Provide enough unused channels to support max_fock_state of {max_fock_state}. Each unused channel can support up to 8 fock states. Maximum max_fock_state is 32."
        )
    transitions = []
    for i in range(max_fock_state):
        transitions.append(f"f{i}g{i+1}")
    if "a" in alice_or_bob.lower():
        cav_name = "alice"
        cav_ch = "SG4"
    elif "b" in alice_or_bob.lower():
        cav_name = "bob"
        cav_ch = "SG2"
    else:
        raise ValueError(f"Invalid alice_or_bob argument ({alice_or_bob})")
    for i in range(8):
        sig_freq_map[serial_num][cav_ch][f"sb_drive_{cav_name}_{transitions[i]}"][
            "automute"
        ] = False
    sig_freq_map[serial_num][cav_ch][f"sb_drive_{cav_name}_f0g1"]["route"] = []
    ch_idx = None
    for i in range(8, len(transitions)):
        if f"sb_drive_{cav_name}_{transitions[i]}" in exp.signals:

            if i == 8:
                ch_idx = 0
            elif i == 16:
                ch_idx = 1
            elif i == 24:
                ch_idx = 2

            sig_freq_map[serial_num][unused_channels[ch_idx]][
                f"sb_drive_{cav_name}_{transitions[i]}"
            ] = {}
            sig_freq_map[serial_num][unused_channels[ch_idx]][
                f"sb_drive_{cav_name}_{transitions[i]}"
            ]["frequency"] = _get_sb_value(
                qubit_parameters["q0"], cav_name, transitions[i], "freq"
            )
            sig_freq_map[serial_num][unused_channels[ch_idx]][
                f"sb_drive_{cav_name}_{transitions[i]}"
            ]["range"] = qubit_parameters["q0"][f"sb_{cav_name}_dBm_range"]
            sig_freq_map[serial_num][unused_channels[ch_idx]][
                f"sb_drive_{cav_name}_{transitions[i]}"
            ]["automute"] = automute
            sig_freq_map[serial_num][unused_channels[ch_idx]][
                f"sb_drive_{cav_name}_{transitions[i]}"
            ]["SW_override"] = sb_SW_override

            if i in [8, 16, 24]:
                sig_freq_map[serial_num][cav_ch][f"sb_drive_{cav_name}_f0g1"][
                    "route"
                ].append(
                    {
                        "src": f"sb_drive_{cav_name}_{transitions[i]}",
                        "amp_scaling": 1,
                        "ph_shift": 0,
                    }
                )
                sig_freq_map[serial_num][unused_channels[ch_idx]][
                    f"sb_drive_{cav_name}_{transitions[i]}"
                ]["route"] = [
                    {
                        "src": f"sb_drive_{cav_name}_f0g1",
                        "amp_scaling": 0,
                        "ph_shift": 0,
                    }
                ]

        else:
            raise ValueError(
                f"Experiment signals do not contain all required sideband transitions. Specifically missing sb_drive_{cav_name}_{transitions[i]}"
            )

    return sig_freq_map
