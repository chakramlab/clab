from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
import time
from slab.instruments import *
from slab.instruments import InstrumentManager
#from slab.instruments.labbrick import labbrick




def single_tone_CW(config):
    nwa=config['instrument']
    expt_path=config['expt_path']
    prefix = config['prefix']
    read_freq = config['read_freq']
    span = config['span']
    ifbw = config['ifbw']
    read_power = config['read_power']
    sweep_pts = config['sweep_pts']
    avgs = config['avgs']
    avgs_state = config['avgs_state']
    measurement=config['measurement']
    delay=config['elec_delay']
    phase_offset=config['phase_offset']

    nwa.set_timeout(10E5)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)

    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    nwa.setup_take(averages=avgs)
    nwa.set_average_state(avgs_state)

    read_freq_start = read_freq - span/2.0
    read_freq_stop = read_freq + span/2.0


    print ("Setting up CW parameters")

    nwa.set_ifbw(ifbw)

    nwa.write('SENSE:FOM:STATE 1')
    nwa.write("sense:fom:range2:coupled 1")
    nwa.write("sense:fom:range3:coupled 1")
    nwa.write("sense:fom:range4:coupled 0")
    nwa.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % read_freq_start)
    nwa.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % read_freq_stop)

    nwa.write('SENSE:FOM:RANGE4:FREQUENCY:START %f' % 1E9)
    nwa.write('SENSE:FOM:RANGE4:FREQUENCY:STOP %f' % 1E9)

    # Turning off pulsed aspect of the measurement (just in case it was run before)
    # Turning the leveling of the ports to 'Internal'.   'open-loop' is for pulsed
    nwa.write("source1:power1:alc:mode INTERNAL")
    nwa.write("source1:power3:alc:mode INTERNAL")
    # Setting up the proper trigger mode.   Want to trigger on the point
    nwa.write("TRIG:SOUR EXT")
    nwa.write("SENS:SWE:TRIG:MODE POINT")

    # Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
    nwa.write("SENS:BWID:TRAC ON")

    # turning on the pulses
    nwa.write("SENS:PULS0 0")
    nwa.write("SENS:PULS1 0")
    nwa.write("SENS:PULS2 0")
    nwa.write("SENS:PULS3 0")
    nwa.write("SENS:PULS4 0")
    # turning off the inverting
    nwa.write("SENS:PULS1:INV 0")
    nwa.write("SENS:PULS2:INV 0")
    nwa.write("SENS:PULS3:INV 0")
    nwa.write("SENS:PULS4:INV 0")


    pnax = nwa
    # hacked to include switches
    # # turning off the pulses
    pnax.write("SENS:PULS0 0")  # automatically sync ADC to pulse gen
    pnax.write("SENS:PULS1 0")
    pnax.write("SENS:PULS2 0")
    pnax.write("SENS:PULS3 0")
    pnax.write("SENS:PULS4 0")
    # turning on the inverting
    pnax.write("SENS:PULS1:INV 1")
    pnax.write("SENS:PULS2:INV 1")
    pnax.write("SENS:PULS3:INV 1")
    pnax.write("SENS:PULS4:INV 1")


    #setting the port powers and decoupling the powers
    nwa.write("SOUR:POW:COUP OFF")
    nwa.write(":SOURCE:POWER1 %f" % (read_power))
    nwa.write(":SOURCE1:POWER1:MODE ON")
    print ("Read Now On")

    nwa.write(":SOURCE1:POWER3:MODE OFF")

    data = nwa.take_one_in_mag_phase()
    print('hey')
    #print(data, len(data))
    fpoints = data[0]
    mags = data[1]
    phases = data[2]
    print ("finished downloading")


    with SlabFile(fname, 'w') as f:
        f.append_line('mags', mags)
        f.append_line('phases', phases)
        f.append_line('freq', fpoints)
        f.append_pt('read_power', read_power)

    print (fname)

def wide_sweep_CW(expt_params):
    nwa=expt_params['instrument']
    ifbw=expt_params['ifbw']
    sweep_pts=expt_params['sweep_pts']
    measurement=expt_params['measurement']
    delay=expt_params['elec_delay']
    read_power=expt_params['read_power']
    expt_path=expt_params['expt_path']
    prefix=expt_params['prefix']
    span=expt_params['search_window']
    search_range=expt_params['search_range']
    rt_cent_freq=expt_params['read_freq']
    avgs=expt_params['avgs']
    avgs_state=expt_params['avgs_state']
    phase_offset=expt_params['phase_offset']
    fname=expt_params['prefix']
    expt_path=expt_params['expt_path']


    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)
    nwa.set_phase_offset(phase_offset, channel=1)

    nwa.write('SENSE:FOM:STATE 1')
    nwa.write("sense:fom:range2:coupled 1")
    nwa.write("sense:fom:range3:coupled 1")
    nwa.write("sense:fom:range4:coupled 0")

    nwa.write('SENSE:FOM:RANGE4:FREQUENCY:START %f' % 0)
    nwa.write('SENSE:FOM:RANGE4:FREQUENCY:STOP %f' % 0)

    # Turning off pulsed aspect of the measurement (just in case it was run before)
    # Turning the leveling of the ports to 'Internal'.   'open-loop' is for pulsed
    nwa.write("source1:power1:alc:mode INTERNAL")
    nwa.write("source1:power3:alc:mode INTERNAL")
    # Setting up the proper trigger mode.   Want to trigger on the point
    nwa.write("TRIG:SOUR EXT")
    nwa.write("SENS:SWE:TRIG:MODE POINT")

    print('Setting base params')
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    search_bins = int(np.around(search_range / span))
    mode_search_freqs = np.linspace(rt_cent_freq - search_range / 2, rt_cent_freq + search_range / 2, search_bins+1)
    freq=np.array([])
    mags=np.array([])
    phases=np.array([])
    i=1
    for read_freq_center in mode_search_freqs:
        print("Scan number %i out of %i"%(i, search_bins))
        read_freq_start = read_freq_center - span/2.0
        read_freq_stop = read_freq_center + span/2.0
        nwa.setup_take(averages=avgs)
        nwa.set_average_state(avgs_state)
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % read_freq_start)
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % read_freq_stop)
        nwa.write(":SOURCE:POWER1 %f" % (read_power))
        nwa.write(":SOURCE1:POWER1:MODE ON")
        data = nwa.take_one_in_mag_phase()
        freq = np.concatenate([freq, data[0]])
        mags = np.concatenate([mags,data[1]])
        phases = np.concatenate([phases, data[2]])
        i+=1
    print('Total data length %s pts'%len(freq))
    with SlabFile(fname, 'w') as f:
        f.append_pt('power', nwa.get_power())
        f.append_line('fpoints', freq)
        f.append_line('mags', mags)
        f.append_line('phases', phases)
    print('Data Saved')
    return

def two_tone_CW(config):
    nwa=config['instrument']
    prefix=config['prefix']
    read_freq=config['read_freq']
    probe_freq_center=config['probe_freq_center']
    span=config['span']
    ifbw=config['ifbw']
    read_power=config['read_power']
    probe_power=config['probe_power']
    sweep_pts=config['sweep_pts']
    avgs=config['avgs']
    avgs_state=config['avgs_state']
    expt_path=config['expt_path']

    if config['EF_probe']==True:
        ge_drive(config)
    else:
        pass

    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    rfreq =  read_freq

    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement("S21")
    #nwa.set_electrical_delay(delay, channel=1)


    nwa.setup_take(averages=avgs)
    nwa.set_average_state(avgs_state)

    probe_start_freq = probe_freq_center-span/2.0
    probe_stop_freq = probe_freq_center+span/2.0
    print ("Setting up pulsed parameters")

    nwa.set_ifbw(ifbw)

    nwa.write('SENSE:FOM:STATE 1')
    nwa.write("sense:fom:range2:coupled 0")
    nwa.write("sense:fom:range3:coupled 0")
    nwa.write("sense:fom:range4:coupled 1")
    nwa.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % probe_start_freq)
    nwa.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % probe_stop_freq)

    nwa.write('SENSE:FOM:RANGE2:FREQUENCY:START %f' % rfreq)
    nwa.write('SENSE:FOM:RANGE2:FREQUENCY:STOP %f' % rfreq)
    nwa.write('SENSE:FOM:RANGE3:FREQUENCY:START %f' % rfreq)
    nwa.write('SENSE:FOM:RANGE3:FREQUENCY:STOP %f' % rfreq)

    print ("set read freq")

    # Turning off pulsed aspect of the measurement (just in case it was run before)
    # Turning the leveling of the ports to 'Internal'.   'open-loop' is for pulsed
    nwa.write("source1:power1:alc:mode INTERNAL")
    nwa.write("source1:power3:alc:mode INTERNAL")
    # Setting up the proper trigger mode.   Want to trigger on the point
    nwa.write("TRIG:SOUR EXT")
    nwa.write("SENS:SWE:TRIG:MODE POINT")

    # Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
    nwa.write("SENS:BWID:TRAC ON")

    # turning on the pulses
    nwa.write("SENS:PULS0 0")
    nwa.write("SENS:PULS1 0")
    nwa.write("SENS:PULS2 0")
    nwa.write("SENS:PULS3 0")
    nwa.write("SENS:PULS4 0")
    # turning off the inverting
    nwa.write("SENS:PULS1:INV 0")
    nwa.write("SENS:PULS2:INV 0")
    nwa.write("SENS:PULS3:INV 0")
    nwa.write("SENS:PULS4:INV 0")

    # setting the port powers and decoupling the powers
    nwa.write("SOUR:POW:COUP OFF")

    nwa.write(":SOURCE:POWER1 %f" % (read_power))
    nwa.write(":SOURCE1:POWER1:MODE ON")
    print ("Read Now On")

    nwa.write(":SOURCE:POWER3 %f" % (probe_power))
    nwa.write(":SOURCE1:POWER3:MODE ON")
    print ("Qubit Drive Now ON")

    data = nwa.take_one_in_mag_phase()
    fpoints = data[0]
    mags = data[1]
    phases = data[2]
    print ("finished downloading at frequency %.3f GHz" %(fpoints[0]/10**9))

    with SlabFile(fname, 'w') as f:
        f.append_line('freq', fpoints)
        f.append_line('mags', mags)
        f.append_line('phases', phases)
        f.append_pt('read_power', read_power)
        f.append_pt('probe_power', probe_power)

def ge_drive(config):
    set_ext_ref=config['LSG_ext_ref']
    freq=config['ge_freq']
    power=config['ge_power']

    LSG.set_output(False)
    LSG.set_use_internal_reference(set_ext_ref)
    LSG.set_frequency(int(freq))
    LSG.set_power(power)
    LSG.set_output(True)

    print('GE drive power: %.1f'%LSG.get_power())
    print('GE drive frequency %.1f'%LSG.get_frequency())
    print('GE drive output: on')

def single_tone_power_sweep(config):

    read_power_list = config["read_power_list"]
    read_freq_center = config["read_freq_center"]
    span = config["span"]
    ifbw = config["ifbw"] 
    sweep_pts = config["sweep_pts"]
    avgs = config["avgs"]
    avgs_state=config["avgs_state"]
    measurement = config["measurement"]
    expt_path = config["expt_path"]
    nwa = config["instrument"]
    delay = config['elec_delay']
    prefix = config["prefix"]
    is_qubitdrive = 0
    qfreq = 0
    
    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)


    vary_param = "single_tone_CW"

    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    fpoints = []
    mags = []
    phases = []
    read_power_col = []


    #setting the port powers and decoupling the powers

    for read_power in read_power_list:
        nwa.setup_take(averages=avgs)
        nwa.set_average_state(avgs_state)

        # if read_power < -30:
        #     nwa.setup_take(averages=5)
        # elif read_power < -25:
        #     nwa.setup_take(averages=5)
        # else:
        #     nwa.setup_take(averages=1)



        read_freq_start = read_freq_center - span / 2.0
        read_freq_stop = read_freq_center + span / 2.0

        print("Setting up CW parameters")

        nwa.write('SENSE:FOM:STATE 1')
        nwa.write("sense:fom:range2:coupled 1")
        nwa.write("sense:fom:range3:coupled 1")
        nwa.write("sense:fom:range4:coupled 0")
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % read_freq_start)
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % read_freq_stop)

        nwa.write('SENSE:FOM:RANGE4:FREQUENCY:START %f' % qfreq)
        nwa.write('SENSE:FOM:RANGE4:FREQUENCY:STOP %f' % qfreq)

        # Turning off pulsed aspect of the measurement (just in case it was run before)
        # Turning the leveling of the ports to 'Internal'.   'open-loop' is for pulsed
        nwa.write("source1:power1:alc:mode INTERNAL")
        nwa.write("source1:power3:alc:mode INTERNAL")
        # Setting up the proper trigger mode.   Want to trigger on the point
        nwa.write("TRIG:SOUR EXT")
        nwa.write("SENS:SWE:TRIG:MODE POINT")

        # Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
        nwa.write("SENS:BWID:TRAC ON")

        # turning on the pulses
        nwa.write("SENS:PULS0 0")
        nwa.write("SENS:PULS1 0")
        nwa.write("SENS:PULS2 0")
        nwa.write("SENS:PULS3 0")
        nwa.write("SENS:PULS4 0")
        # turning off the inverting
        nwa.write("SENS:PULS1:INV 0")
        nwa.write("SENS:PULS2:INV 0")
        nwa.write("SENS:PULS3:INV 0")
        nwa.write("SENS:PULS4:INV 0")

        pnax = nwa
        # hacked to include switches
        # # turning off the pulses
        pnax.write("SENS:PULS0 0")  # automatically sync ADC to pulse gen
        pnax.write("SENS:PULS1 0")
        pnax.write("SENS:PULS2 0")
        pnax.write("SENS:PULS3 0")
        pnax.write("SENS:PULS4 0")
        # turning on the inverting
        pnax.write("SENS:PULS1:INV 1")
        pnax.write("SENS:PULS2:INV 1")
        pnax.write("SENS:PULS3:INV 1")
        pnax.write("SENS:PULS4:INV 1")

        nwa.write("SOUR:POW:COUP OFF")
        nwa.write(":SOURCE:POWER1 %f" % (read_power))
        nwa.write(":SOURCE1:POWER1:MODE ON")
        print ("Read Now On")

        nwa.write(":SOURCE1:POWER3:MODE OFF")
        print ("Qubit Drive Now Off")

        data = nwa.take_one_in_mag_phase()
        fpoints.append(data[0])
        mags.append(data[1])
        phases.append(data[2])
        read_power_col.append(read_power)

        print ("finished downloading")


    with SlabFile(fname, 'w') as f:
        f.append_data(f, 'mags', mags)
        f.append_data(f, 'phases', phases)
        f.append_data(f, 'freq', fpoints)
        f.append_data(f, 'read_power', read_power_col)

def two_tone_power_sweep(config):
    #vary_param = "qubit_twotone_CW"

    # print("Swept Parameter: %s" % (vary_param))

    probe_power_list = config["probe_power_list"]
    read_freq = config["read_freq"]
    probe_freq_center = config['probe_freq_center']
    span = config["span"]
    ifbw = config["ifbw"] 
    read_power = config["read_power"]
    sweep_pts = config["sweep_pts"]
    avgs = config["avgs"]
    avgs_state=config["avgs_state"]
    measurement = config["measurement"]
    expt_path = config["expt_path"]
    nwa = config["instrument"]
    delay = config['elec_delay']
    prefix = config["prefix"]

    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    rfreq =  read_freq

    fpoints = []
    mags = []
    phases = []
    read_power_col = []
    probe_power_col = []
    
    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)
    nwa.setup_take(averages=avgs)
    nwa.set_average_state(avgs_state)

    for probe_power in probe_power_list:
        nwa.setup_take(averages=avgs)

        probe_start_freq = probe_freq_center-span/2.0
        probe_stop_freq = probe_freq_center+span/2.0
    #
        print ("Setting up pulsed parameters")

        nwa.set_ifbw(ifbw)

        nwa.write('SENSE:FOM:STATE 1')
        nwa.write("sense:fom:range2:coupled 0")
        nwa.write("sense:fom:range3:coupled 0")
        nwa.write("sense:fom:range4:coupled 1")
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % probe_start_freq)
        nwa.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % probe_stop_freq)

        nwa.write('SENSE:FOM:RANGE2:FREQUENCY:START %f' % rfreq)
        nwa.write('SENSE:FOM:RANGE2:FREQUENCY:STOP %f' % rfreq)
        nwa.write('SENSE:FOM:RANGE3:FREQUENCY:START %f' % rfreq)
        nwa.write('SENSE:FOM:RANGE3:FREQUENCY:STOP %f' % rfreq)

        print ("set read freq")

        # Turning off pulsed aspect of the measurement (just in case it was run before)
        # Turning the leveling of the ports to 'Internal'.   'open-loop' is for pulsed
        nwa.write("source1:power1:alc:mode INTERNAL")
        nwa.write("source1:power3:alc:mode INTERNAL")
        # Setting up the proper trigger mode.   Want to trigger on the point
        nwa.write("TRIG:SOUR EXT")
        nwa.write("SENS:SWE:TRIG:MODE POINT")

        # Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
        nwa.write("SENS:BWID:TRAC ON")

        # turning on the pulses
        nwa.write("SENS:PULS0 0")
        nwa.write("SENS:PULS1 0")
        nwa.write("SENS:PULS2 0")
        nwa.write("SENS:PULS3 0")
        nwa.write("SENS:PULS4 0")
        # turning off the inverting
        nwa.write("SENS:PULS1:INV 0")
        nwa.write("SENS:PULS2:INV 0")
        nwa.write("SENS:PULS3:INV 0")
        nwa.write("SENS:PULS4:INV 0")

        # setting the port powers and decoupling the powers
        nwa.write("SOUR:POW:COUP OFF")

        nwa.write(":SOURCE:POWER1 %f" % (read_power))
        nwa.write(":SOURCE1:POWER1:MODE ON")
        print ("Read Now On")

        nwa.write(":SOURCE:POWER3 %f" % (probe_power))
        nwa.write(":SOURCE1:POWER3:MODE ON")
        print ("Qubit Drive Now ON")


        data = nwa.take_one_in_mag_phase()
        fpoints.append(data[0])
        mags.append(data[1])
        phases.append(data[2])
        probe_power_col.append(probe_power)
        read_power_col.append(read_power)

        # print ("finished downloading at frequency %.3f GHz" %(fpoints[0]/10**9))

        with SlabFile(fname, 'w') as f:
            f.append_data(f, 'freq', fpoints)
            f.append_data(f, 'mags', mags)
            f.append_data(f, 'phases', phases)
            f.append_data(f, 'read_power', read_power_col)
            f.append_data(f, 'probe_power', probe_power_col)

