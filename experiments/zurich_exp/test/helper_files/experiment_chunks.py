from laboneq.simple import *
import numpy as np

# convenient functions for some commonly used sections
def trig_sect_for_debug(exp, trig_logical_line, trig_length = 0.12e-6, play_after = None):
    '''
    Output trigger from the marker channel of "trig_logical_line" for debugging purposes
    No other signal is allowed to be played during triggering.
    '''
    current_section = "trig_from_func"
    with exp.section(
        uid = "trig_from_func",
        play_after = play_after,
        trigger = {trig_logical_line: {"state": 1}},
        length = trig_length,
        on_system_grid = True,
    ):
        for k in exp.signals:
            exp.reserve(signal = k)
    return current_section

def add_qb_state_calibration(
        exp,
        previous_section,
        states = "gef",
        ):
    # prepare state g
    # measure
    # prepare state e
    # measure
    # prep state f
    # measure
    # return last section uid and handles
    return

def prepare_cav_state(exp, 
                      state = '0',
                      previous_section = None, 
                      ge_X180 = None, 
                      ge_X90 = None, 
                      ef_X180 = None,
                      sb_pulses = None,
                      alice_or_bob = 'a',
                      function_index = 0,
                      ):
    '''
    state can take any n integer, depending on the sb transition pulses you have defined.
    in this version, n < 30
    state can also take 0+n
    '''
    if 'a' in alice_or_bob.lower():
        sb_name = "alice"
    elif 'b' in alice_or_bob.lower():
        sb_name = "bob"
    if ef_X180 == None:
        raise ValueError(f"ef_X180 must be defined to create state, {state}")
    if sb_pulses == None:
        raise ValueError(f"sb_pulses must be defined to create state, {state}")
    if ('+' not in state) and (len(state) == 1):
        if state == '0':
            pass
        else:
            if ge_X180 == None:
                raise ValueError(f"ge_X180 must be defined to create state, {state}")
            for ii in range(int(state)):
                i = ii+1
                with exp.section(uid = f"ge_transition_{i}_{function_index}", play_after = previous_section, alignment=SectionAlignment.RIGHT, on_system_grid=True):
                    exp.play(signal = "qb_drive", pulse = ge_X180)
                with exp.section(uid = f"ef_transition_{i}_{function_index}", play_after = f"ge_transition_{i}_{function_index}", on_system_grid=True):
                    exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                with exp.section(uid = f"sb_transition_{i}_{function_index}", play_after = f"ef_transition_{i}_{function_index}"):
                    exp.play(signal = f"sb_drive_{sb_name}_f{ii}g{i}", pulse = sb_pulses[sb_name][f"f{ii}g{i}"])
                previous_section = f"sb_transition_{i}_{function_index}"
    elif '+' in state: # separated out in case want to do shelfing.
        if state[0] != '0' or len(state) != 3:
            raise ValueError("state must be in the form of 'n' or '0+n'")
        else:
            if ge_X90 == None:
                raise ValueError(f"ge_X90 must be defined to create state, {state}")
            for ii in range(int(state[-1])):
                i = ii+1
                with exp.section(uid = f"ge_transition_{i}_{function_index}", play_after = previous_section, alignment=SectionAlignment.RIGHT, on_system_grid=True):
                    exp.play(signal = "qb_drive", pulse = ge_X90)
                with exp.section(uid = f"ef_transition_{i}_{function_index}", play_after = f"ge_transition_{i}_{function_index}", on_system_grid=True):
                    exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                with exp.section(uid = f"sb_transition_{i}_{function_index}", play_after = f"ef_transition_{i}_{function_index}"):
                    exp.play(signal = f"sb_drive_{sb_name}_f{ii}g{i}", pulse = sb_pulses[sb_name][f"f{ii}g{i}"])
                previous_section = f"sb_transition_{i}_{function_index}"
    else:
        raise ValueError("No state is prepared in  'prepare_cav_state' function. Check experiment.")
    return previous_section

def prepare_0pn_state_with_phase_offset(exp, 
                      state = '0',
                      previous_section = None, 
                      ge_X180 = None, 
                      ge_X90 = None, 
                      ef_X180 = None,
                      sb_pulses = None,
                      alice_or_bob = 'a',
                      function_index = 0,
                      phase_offset = 0,
                      ge_pi_over_2_phase_offset: bool = False,
                      ef_pi_phase_offset: bool = False,
                      sb_phase_offset: bool = False,
                      ):
    '''
    state can take any n integer, depending on the sb transition pulses you have defined.
    in this version, n < 30
    state can also take 0+n
    '''
    if 'a' in alice_or_bob.lower():
        sb_name = "alice"
    elif 'b' in alice_or_bob.lower():
        sb_name = "bob"
    if ef_X180 == None:
        raise ValueError(f"ef_X180 must be defined to create state, {state}")
    if sb_pulses == None:
        raise ValueError(f"sb_pulses must be defined to create state, {state}")
    if '+' in state: # separated out in case want to do shelfing.
        if state[0] != '0' or len(state) != 3:
            raise ValueError("state must be in the form of 'n' or '0+n'")
        if state[-1] != '1':
            raise ValueError("state must be in the form of '0+n' where n is 1")
        else:
            if ge_X90 == None:
                raise ValueError(f"ge_X90 must be defined to create state, {state}")
            for ii in range(int(state[-1])):
                i = ii+1
                with exp.section(uid = f"ge_transition_{i}_{function_index}", play_after = previous_section, alignment=SectionAlignment.RIGHT, on_system_grid=True):
                    if ge_pi_over_2_phase_offset:
                        exp.play(signal = "qb_drive", pulse = ge_X90, phase = phase_offset)
                    else:
                        exp.play(signal = "qb_drive", pulse = ge_X90)
                with exp.section(uid = f"ef_transition_{i}_{function_index}", play_after = f"ge_transition_{i}_{function_index}", on_system_grid=True):
                    if ef_pi_phase_offset:
                        exp.play(signal = "qb_ef_drive", pulse = ef_X180, phase = phase_offset)
                    else:
                        exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                with exp.section(uid = f"sb_transition_{i}_{function_index}", play_after = f"ef_transition_{i}_{function_index}"):
                    if sb_phase_offset:
                        exp.play(signal = f"sb_drive_{sb_name}_f{ii}g{i}", pulse = sb_pulses[sb_name][f"f{ii}g{i}"], phase = phase_offset)
                    else:
                        exp.play(signal = f"sb_drive_{sb_name}_f{ii}g{i}", pulse = sb_pulses[sb_name][f"f{ii}g{i}"])
                previous_section = f"sb_transition_{i}_{function_index}"
    else:
        raise ValueError("improper state is inputted. please input 0+1 state")
    return previous_section

def prepare_qb_state(exp, 
                     qubit_parameters,
                     state = 'g',
                     previous_section = None, 
                     ge_X180 = None, 
                     ef_X180 = None,
                     function_index = 0,
                     ):
    '''
    Prepare either g e or f state of the qubit.
    Aware that g state preparation will add a delay length of 'reset_delay' to passively relax the qubit.
    Note that function index is pretty important. It is used to keep track of the section uids.
    For each function that you call in the experiment script, you should give a unique function index.
    '''
    if len(state) != 1:
        raise ValueError("state must be a single character, either 'g', 'e', or 'f'")
    if state == 'g':
        with exp.section(uid = f"prep_g_{function_index}", play_after = previous_section):
            # exp.delay(signal = "qb_drive", time = qubit_parameters["q0"]["reset_delay"])
            pass
        previous_section = f"prep_g_{function_index}"
    elif state == 'e':
        with exp.section(uid = f"prep_e_{function_index}", play_after = previous_section):
            exp.play(signal = "qb_drive", pulse = ge_X180)
        previous_section = f"prep_e_{function_index}"
    elif state == 'f':
        with exp.section(uid = f"prep_e_{function_index}", play_after = previous_section, alignment=SectionAlignment.RIGHT, on_system_grid = True):
            exp.play(signal = "qb_drive", pulse = ge_X180)
        with exp.section(uid = f"prep_f_{function_index}", play_after = f"prep_e_{function_index}", on_system_grid= True):
            exp.play(signal = "qb_ef_drive", pulse = ef_X180)
        previous_section = f"prep_f_{function_index}"
    else:
        raise ValueError("No state is prepared in  'prepare_qb_state' function. Check experiment.")
    return previous_section

