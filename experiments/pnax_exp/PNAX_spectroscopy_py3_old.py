# -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nate E
"""

from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
import time
from slab.instruments import *
from slab.instruments import InstrumentManager

im = InstrumentManager()
nwa =  im['PNAX']
print(nwa.get_id())
print('Deviced Connected')

expt_path = os.getcwd() + '\data'

# initial NWA configuration values
ifbw = 500
read_power = -40
probe_power = 5
sweep_pts = 500
avgs = 20
avgs_state = 1
delay = 0.0

prefix = "PNAX_%s_flute cavity"

print("Configuring the NWA")



print ("NWA Configured. IFBW = %f Hz, Read_Power = %f, Drive_Power = %f, NumPts = %d, Avgs = %d " %(ifbw,read_power,probe_power,sweep_pts,avgs))

def single_tone_CW(read_freq_center,span,read_power = read_power,ifbw = ifbw, sweep_pts= sweep_pts,avgs=avgs,avgs_state=avgs_state,is_qubitdrive= 0,qfreq= 0,measurement="S21"):

    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)


    vary_param = "single_tone_CW"
    print("Swept Parameter: %s" % (vary_param))
    prefix = "Readout_%s" % vary_param
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    nwa.setup_take(averages=avgs)
    nwa.set_average_state(avgs_state)

    read_freq_start = read_freq_center - span/2.0
    read_freq_stop =  read_freq_center + span/2.0


    print ("Setting up CW parameters")

    nwa.set_ifbw(ifbw)

    nwa.write('SENSE:FOM:STATE 0')
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


    #setting the port powers and decoupling the powers
    nwa.write("SOUR:POW:COUP OFF")
    if measurement == 'S23':
        print ("Went here to power 3")
        nwa.write(":SOURCE:POWER3 %f" % (read_power))
        nwa.write(":SOURCE1:POWER3:MODE ON")
    else:
        nwa.write(":SOURCE:POWER1 %f" % (read_power))
        nwa.write(":SOURCE1:POWER1:MODE ON")
    print ("Read Now On")

    if measurement == 'S21':
        nwa.write(":SOURCE:POWER3 %f" % (probe_power))
    if is_qubitdrive:
        nwa.write(":SOURCE1:POWER3:MODE ON")
        print ("Qubit Drive Now ON")
    else:
        nwa.write(":SOURCE1:POWER3:MODE OFF")
        print ("Qubit Drive Now Off")

    data = nwa.take_one_in_mag_phase()
    fpoints = data[0]
    mags = data[1]
    phases = data[2]
    print ("finished downloading")


    with SlabFile(fname) as f:
        f.append_line('mags', mags)
        f.append_line('phases', phases)
        f.append_line('freq', fpoints)
        f.append_pt('read_power', read_power)
        f.append_pt('probe_power', probe_power)

    print (fname)

def single_tone_power_sweep(read_power_list,read_freq_center,span,ifbw = ifbw, sweep_pts= sweep_pts,avgs=avgs,avgs_state=avgs_state,is_qubitdrive= 0,qfreq= 0,measurement="S21"):

    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement(measurement)
    nwa.set_electrical_delay(delay, channel=1)


    vary_param = "single_tone_CW"

    prefix = "Single_tone_nearReadoutFreq_Transmission"
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)



    #setting the port powers and decoupling the powers

    for read_power in read_power_list:
        nwa.setup_take(averages=avgs)
        nwa.set_average_state(avgs_state)

        if read_power < -30:
            nwa.setup_take(averages=5)
        elif read_power < -25:
            nwa.setup_take(averages=5)
        else:
            nwa.setup_take(averages=1)



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
        fpoints = data[0]
        mags = data[1]
        phases = data[2]
        print ("finished downloading")


        with SlabFile(fname) as f:
            f.append_line('mags', mags)
            f.append_line('phases', phases)
            f.append_line('freq', fpoints)
            f.append_pt('read_power', read_power)

def two_tone_CW(read_freq,probe_freq_center,span,ifbw = ifbw, read_power = read_power,probe_power = probe_power,sweep_pts= sweep_pts,avgs=avgs,avgs_state=avgs_state):
    vary_param = "twotone_CW"

    print("Swept Parameter: %s" % (vary_param))
    prefix = "%s_" % vary_param
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    rfreq =  read_freq

    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement("S21")
    nwa.set_electrical_delay(delay, channel=1)


    nwa.setup_take(averages=avgs)
    nwa.set_average_state(avgs_state)

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
    fpoints = data[0]
    mags = data[1]
    phases = data[2]
    print ("finished downloading at frequency %.3f GHz" %(fpoints[0]/10**9))

    with SlabFile(fname) as f:
        f.append_line('freq', fpoints)
        f.append_line('mags', mags)
        f.append_line('phases', phases)
        f.append_pt('read_power', read_power)
        f.append_pt('probe_power', probe_power)

def two_tone_power_sweep(probe_power_list,read_freq,probe_freq_center,span,ifbw = ifbw, read_power = read_power,sweep_pts= sweep_pts,avgs=avgs,avgs_state=avgs_state):
    #vary_param = "qubit_twotone_CW"

   # print("Swept Parameter: %s" % (vary_param))
    prefix = "twotoneVsPowerTransmission"
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print(fname)
    fname = os.path.join(expt_path, fname)

    rfreq =  read_freq

    nwa.set_timeout(10E5)
    nwa.set_ifbw(ifbw)
    nwa.set_sweep_points(sweep_pts)
    nwa.clear_traces()
    nwa.setup_measurement("S21")
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
        fpoints = data[0]
        mags = data[1]
        phases = data[2]

        print ("finished downloading at frequency %.3f GHz" %(fpoints[0]/10**9))

        with SlabFile(fname) as f:
            f.append_line('freq', fpoints)
            f.append_line('mags', mags)
            f.append_line('phases', phases)
            f.append_pt('read_power', read_power)
            f.append_pt('probe_power', probe_power)


read_freq  =   7.784411643008987e9
single_tone_CW(read_freq, 5e6, measurement="S21")
#/
# read_power_list = arange(-21,5,1)
# single_tone_power_sweep(read_power_list,read_freq_center = read_freq,span = 100e6)
#
# read_freq  =   7.767947746052817e9
# single_tone_power_sweep(read_power_list,read_freq_center = read_freq,span = 5e6)
#
#
# single_tone_CW(read_freq, 6.17e6, measurement="S21",is_qubitdrive=False,qfreq = 3.810241016669044e9)



# two_tone_CW(read_freq,probe_freq_center=4e9, span =  2e9)
# two_tone_power_sweep(probe_power_list = arange(-55,0,1),read_freq=read_freq,probe_freq_center=6.0e9,span=2.2e9)