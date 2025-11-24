__author__ = 'Alex Ma/Nate Earnest'

from slab import *
from numpy import *
import os
from tqdm import tqdm
from slab.instruments import PNAX
from slab.dataanalysis import get_next_filename

im = InstrumentManager(config_path='C:\_Lib\python\slab\instruments\instruments.cfg')
def polar2mag(xs, ys):
    return 20*np.log10(np.sqrt(xs ** 2 + ys ** 2)), np.arctan2(ys, xs) * 180/np.pi

class PnaxPulseExperiment(Experiment):
    '''pnax pulsed experiments'''

    def __init__(self, expt_name, config_file = '..\\config_pnax.json', filename = None, path=None, **kwargs):

        self.__dict__.update(kwargs)

        if path is None:
            path = os.getcwd() #+ '\Data'

        if filename is None:
            filename="pnax_pulsed_" + expt_name

        super(PnaxPulseExperiment, self).__init__(path=path, prefix=filename, config_file=config_file, liveplot_enabled=False, **kwargs)   
            
        print('Writing to datafile:', self.fname)

        print(self.cfg, 'printing')

        # this formats the config file into a single line....
        # self.save_config()
        # self.pnax = im['PNAX2']
        self.pnax = im['PNAX']
        # previous line already loads the devices in 'aliases'

        print('PNAX Connected:')
        print(self.pnax.get_id())
        # print(self.dcflux.get_id())
        # print(self.drive2.get_id())

        # if self.cfg['flux']['set_flux']:
        #     self.dcflux.set_mode('current')
        #     # self.dcflux.set_range(self.cfg['flux']['range'])
        #     # self.dcflux.set_output(True)
        #     print("Dcflux range set to %.4f mA" % (1000*self.cfg['flux']['range']))
        #     print("Ramping dcflux to %.4f mA" % (1000*self.cfg['flux']['current']))
        #     self.dcflux.set_current(self.cfg['flux']['current'])

        # initialize all common pulse settings
        self.initialize_pnax()
        print('Readout power is %.2f'%self.cfg['readout']['power'])

        try:
            self.LSG = im['LB1']
            self.LSG_state=True
            print('LSG connected')
        except:
            print('Labbrick not connected, cannot do EF pulse probe')
            self.LSG_state=False

        if expt_name.lower() == 'vacuum_rabi_focus_scan':
            self.vacuum_rabi_focus_scan(expt_name, **kwargs)

        elif expt_name.lower() == 'vacuum_rabi':
            self.vacuum_rabi(expt_name, **kwargs)

        elif expt_name.lower() == 'cavity_ringdown':
            self.cavity_ringdown(expt_name, **kwargs)

        elif expt_name.lower() == 'pulse_probe':
            self.pulse_probe(expt_name, **kwargs)

        elif expt_name.lower() == 'rabi':
            self.rabi(expt_name, **kwargs)

        elif expt_name.lower() == 't1':
            self.t1(expt_name, **kwargs)

        elif expt_name.lower() == 'ramsey':
            self.ramsey(expt_name, **kwargs)

        elif expt_name.lower() == 'spin_echo':
            self.spin_echo(expt_name, **kwargs)

        elif expt_name.lower() == 'histogram':
            self.histogram(expt_name, **kwargs)

        elif expt_name.lower() == 'raman':
            self.raman(expt_name, **kwargs)

        elif expt_name.lower() == 'raman_rabi':
            self.raman_rabi(expt_name, **kwargs)

        elif expt_name.lower() == 'raman_ramsey':
            self.raman_ramsey(expt_name, **kwargs)

        elif expt_name.lower() == 'raman_t1':
            self.raman_t1(expt_name, **kwargs)

        elif expt_name.lower() == 'alex_test':
            self.alex_test(expt_name, **kwargs)

        elif expt_name.lower() == 'ef_pulse_probe':
            self.ef_pulse_probe(expt_name, **kwargs)

        else:
            print('No experiment named', expt_name.lower(), '! process aborted.')


    def focus_pts(self, f0, span, steps, spacing):
        steps=int(steps/2)
        dx = 1.0 / (steps - 1)
        us = [f0 + (i * dx) ** spacing * span / 2 for i in range(steps + 1)][1::]
        ls = [f0 - (i * dx) ** spacing * span / 2 for i in range(steps)]
        ls.reverse()
        return np.array(ls + us[0:-1])

    def initialize_pnax(self):

        print("\nConfiguring the PNAX for pulse measurements")
        self.pnax.set_timeout(10E3)
        self.pnax.clear_traces()
        self.pnax.setup_measurement("S21")
        # right now ifbw cover the whole readout period, so average is done using set_sweep_points
        avgs = 1
        self.pnax.setup_take(averages_state=True)
        self.pnax.set_averages_and_group_count(avgs, True)
        # setting average - will be reset in each exp function
        self.pnax.set_sweep_points(100)

        self.pnax.write('SENSE:FOM:STATE 1')
        # todo: maybe set al ranges to uncoupled and setting freq manually??
        self.pnax.write("sense:fom:range2:coupled 0") # source 1
        self.pnax.write("sense:fom:range3:coupled 0") # receivers
        self.pnax.write("sense:fom:range4:coupled 1") # source 2

        # since range 4 is coupled, setting range1 freq effectively sets range 4 as well
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % self.cfg['qubit']['freq'])
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % self.cfg['qubit']['freq'])

        self.pnax.write('SENSE:FOM:RANGE2:FREQUENCY:START %f' % (self.cfg['readout']['freq']))
        self.pnax.write('SENSE:FOM:RANGE2:FREQUENCY:STOP %f' % self.cfg['readout']['freq'])
        self.pnax.write('SENSE:FOM:RANGE3:FREQUENCY:START %f' % self.cfg['readout']['freq'])
        self.pnax.write('SENSE:FOM:RANGE3:FREQUENCY:STOP %f' % self.cfg['readout']['freq'])

        # setting the port powers and decoupling the powers
        self.pnax.write("SOUR:POW:COUP OFF")
        self.pnax.write(":SOURCE:POWER1 %f" % (self.cfg['readout']['power']))
        self.pnax.write(":SOURCE1:POWER1:MODE ON")
        self.pnax.write(":SOURCE:POWER3 %f" % (self.cfg['drive']['power']))
        self.pnax.write(":SOURCE1:POWER3:MODE ON")

        #Turning the leveling of the ports to 'open-loop'. This is required when trying to pulsed modulated measurements otherwise the ALC will try to level the source with the detected power level with pulse on and off, causing a source unleveled error
        self.pnax.write("source1:power1:alc:mode openloop")
        self.pnax.write("source1:power3:alc:mode openloop")
        #Setting up the proper trigger mode. Want to trigger on the point

        # ext sour is not necessary, pulse0 ensure int. sync? - alex ma
        #self.pnax.write("TRIG:SOUR EXT")
        self.pnax.write("SENS:SWE:TRIG:MODE POINT")

        #Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
        self.pnax.write("SENS:BWID:TRAC OFF")

        #turning on the pulses
        self.pnax.write("SENS:PULS0 1") # automatically sync ADC to pulse gen
        self.pnax.write("SENS:PULS1 1")
        self.pnax.write("SENS:PULS2 1")
        self.pnax.write("SENS:PULS3 1")
        self.pnax.write("SENS:PULS4 1")
        #turning off the inverting
        self.pnax.write("SENS:PULS1:INV 0")
        self.pnax.write("SENS:PULS2:INV 0")
        self.pnax.write("SENS:PULS3:INV 0")
        self.pnax.write("SENS:PULS4:INV 0")

        # setting the exp period
        self.pnax.write("SENS:PULS:PERiod %.12f" % (self.cfg['expt_trigger']['period_ns']*1e-9))

        # This is setting the ifbw to be wide enough to ensure that the measurement occurs within the readout pulse
        ifbw = 1.0/(0.99*self.cfg['readout']['width']*1e-9)
        self.pnax.set_ifbw(ifbw)
        # todo attention: only discrete values of ifbw
        print('target ifbw =', ifbw)
        print('actual ifbw =', self.pnax.get_ifbw())

        self.pnax.write("SENS:PULS1:WIDT %.12f" % (self.cfg['readout']['width']*1e-9))
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (self.cfg['drive']['width']*1e-9))
        self.pnax.write("SENS:PULS3:WIDT %.12f" % (0.0)) # redundant
        self.pnax.write("SENS:PULS4:WIDT %.12f" % (0.0)) # powering Nate's box

        # delays of various pulses

        # take care of the case where readout is inside drive pulse
        # start delay of pulse 1
        self.read_delay = 0.99 * self.cfg['expt_trigger']['period_ns'] - \
                     max(self.cfg['readout']['delay'], self.cfg['readout']['width'])

        drive_end_delay_unadjust = self.read_delay - self.cfg['readout']['delay']
        # adjust for box response, ns
        self.pulse2end_delay = drive_end_delay_unadjust + self.cfg['pulse_box_delay_adjust']['pulse2']
        self.pulse3end_delay = drive_end_delay_unadjust + self.cfg['pulse_box_delay_adjust']['pulse3']
        self.pulse4end_delay = drive_end_delay_unadjust + self.cfg['pulse_box_delay_adjust']['pulse4']

        drive1_delay = self.pulse2end_delay - self.cfg['drive']['width']

        self.pnax.write("SENS:PULS0:DEL %.12f" % ( (self.read_delay + self.cfg['readout']['ADC_delay']) *1e-9) )
        self.pnax.write("SENS:PULS1:DEL %.12f" % (self.read_delay*1e-9) )
        self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay*1e-9) )
        self.pnax.write("SENS:PULS3:DEL %.12f" % (0) )
        self.pnax.write("SENS:PULS4:DEL %.12f" % (0) )

    def vacuum_rabi_focus_scan(self, expt_name, **kwargs):
        ''' change readout-delay and pulse lengths for pseudo CW experiment '''

        # setting average
        self.pnax.set_sweep_points(dict(self.cfg)[expt_name]['averages'])

        t_pt = time.time()

        # couple receiver to source 1
        self.pnax.write("sense:fom:range2:coupled 1")
        self.pnax.write("sense:fom:range3:coupled 1")
        self.pnax.write("sense:fom:range4:coupled 0")

        #follows a power law that becomes more sparse the further away from center
        center=self.cfg[expt_name]['center']    #Hz
        sweep_range=self.cfg[expt_name]['range']      #Hz
        steps=self.cfg[expt_name]['steps']
        spacing=self.cfg[expt_name]['spacing']
        sweep_avg=self.cfg[expt_name]['sweep_avg']
        print(center, range, steps, spacing, sweep_avg)


        scan_points=self.focus_pts(center, sweep_range, steps, spacing)

        nn = len(scan_points)

        if self.cfg[expt_name]['pi_pulse']:

            drive_ns = self.cfg['pulse_info']['pi_time']
            self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive_ns * 1e-9))
            drive1_delay = self.pulse2end_delay - drive_ns
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write(":SOURCE1:POWER3:MODE ON")
        else:
            self.pnax.write(":SOURCE1:POWER3:MODE OFF")
        
        #save settings initially
        with SlabFile(self.fname, 'a') as f:
                    #f.append_pt('t_pts', t_pt)
                    f.append_pt('flux_pts', self.cfg['flux']['current'])
                    f.append_line('readfreq_pts', scan_points)
                    f.append_pt('read_power', self.cfg['readout']['power'])
                    f.append_pt('drive_power', self.cfg['drive']['power'])
                    # only partial info in here..
                    f.save_settings(self.pnax.get_settings())
                    
        for avg_num in range(0, sweep_avg):
            print(f'Now running average #{avg_num+1}')
            for ii, scan_pt in tqdm(enumerate(scan_points)):
                # now for vaccum rabi, range 2,3 are coupled, which are simult. set
                self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % scan_pt)
                self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % scan_pt)
                data = self.pnax.take()
                fpoints_dummy = data[0]
                I = data[1]
                Q = data[2]
                # print(".. taking %d of %d at freq %.6f GHz" % (ii+1, nn+1, scan_pt*1e-9))
                if avg_num+1>1:
                    with SlabFile(self.fname, 'r') as f:
                        I_prev=np.array(f['imag'])
                        Q_prev=np.array(f['real'])
                    I_averaged=average(avg_num*[I_prev[ii]]+[average(I)])
                    Q_averaged=average(avg_num*[Q_prev[ii]]+[average(Q)])
                else:
                    I_averaged = average(I)
                    Q_averaged = average(Q)
                # print(average(mags), mags_averaged, avg_num)
                with SlabFile(self.fname, 'a') as f:
                    f.attrs.modify('trace_avgs', avg_num+1)
                    if avg_num+1>1:
                        f['imag'][ii]=I_averaged
                        f['real'][ii]=Q_averaged
                        mag, phase=polar2mag(I_averaged, Q_averaged)
                        f['mags'][ii]=mag
                        f['phases'][ii]=phase
                    else:
                        f.append_pt('imag', I_averaged)
                        f.append_pt('real', Q_averaged)
                        mag, phase=polar2mag(I_averaged, Q_averaged)
                        f.append_pt('mags', mag)
                        f.append_pt('phases', phase)
                    # f.append_pt(f'raw_phase_{avg_num}', phases)
                    # f.append_pt(f'raw_mags_{avg_num}', average(mags))

        print('\nData file saved:')
        print(self.fname)


    def vacuum_rabi(self, expt_name, **kwargs):

        ''' change readout-delay and pulse lengths for pseudo CW experiment '''

        # setting average
        self.pnax.set_sweep_points(dict(self.cfg)[expt_name]['averages'])

        t_pt = time.time()

        # couple receiver to source 1
        self.pnax.write("sense:fom:range2:coupled 1")
        self.pnax.write("sense:fom:range3:coupled 1")
        self.pnax.write("sense:fom:range4:coupled 0")

        start = self.cfg[expt_name]['start']    # * 1e-9
        stop = self.cfg[expt_name]['stop']      # * 1e-9
        step = self.cfg[expt_name]['step']      # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn*step+start, nn+1)

        if self.cfg[expt_name]['pi_pulse']:

            drive_ns = self.cfg['pulse_info']['pi_time']
            self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive_ns * 1e-9))
            drive1_delay = self.pulse2end_delay - drive_ns
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write(":SOURCE1:POWER3:MODE ON")
        else:
            self.pnax.write(":SOURCE1:POWER3:MODE OFF")

        for ii, scan_pt in enumerate(scan_points):

            # now for vaccum rabi, range 2,3 are coupled, which are simult. set
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % scan_pt)
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % scan_pt)
            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at freq %.6f GHz" % (ii+1, nn+1, scan_pt*1e-9))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                if ii==0:
                    f.append_pt('flux_pts', self.cfg['flux']['current'])
                    f.append_line('readfreq_pts', scan_points)
                    f.append_pt('read_power', self.cfg['readout']['power'])
                    f.append_pt('drive_power', self.cfg['drive']['power'])
                    # only partial info in here..
                    f.save_settings(self.pnax.get_settings())
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)

        print('\nData file saved:')
        print(self.fname)

    def cavity_ringdown(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()
        self.pnax.write(":SOURCE1:POWER3:MODE OFF")


        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)
        # scan_points = logspace(-2,1.6,25)*1e6

        for ii, scan_pt in enumerate(scan_points):

            self.pnax.write("SENS:PULS0:DEL %.12f" % ((self.read_delay + self.cfg['readout']['ADC_delay']+scan_pt) * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at t1 wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:

                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    def pulse_probe(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        t_pt = time.time()

        #follows a power law that becomes more sparse the further away from center
        center=self.cfg[expt_name]['center']    #Hz
        sweep_range=self.cfg[expt_name]['range']      #Hz
        steps=self.cfg[expt_name]['steps']
        spacing=self.cfg[expt_name]['spacing']
        # sweep_avg=self.cfg[expt_name]['sweep_avg']
        print(center, range, steps, spacing)


        # scan_points=self.focus_pts(center, sweep_range, steps, spacing)
        scan_points = linspace(center-sweep_range/2.0,center+sweep_range/2.0,steps)

        for ii, scan_pt in tqdm(enumerate(scan_points)):

            # now range 4 (Source2) is coupled, so qfreq is set by changing range1
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % scan_pt)
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % scan_pt)

            data = self.pnax.take_one_in_mag_phase()
            mags = data[1]
            phases = data[2]
            # print(".. taking %d of %d at freq %.6f GHz" % (ii + 1, scan_pt * 1e-9))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_line('drivefreq_pts', scan_points)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        return self.fname

    def rabi(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        t_pt = time.time()

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            self.pnax.write("SENS:PULS2:WIDT %.12f" % (scan_pt * 1e-9))

            drive1_delay = self.pulse2end_delay - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at rabi duration %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    def t1(self, expt_name, **kwargs):

        # setting average
        print(self.cfg, 'test10')
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        if self.cfg[expt_name]['use_pi_time']:
            drive_width = self.cfg['pulse_info']['pi_time']
        else:
            drive_width = self.cfg['drive']['width']

        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive_width * 1e-9)) #setting pulse width

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        sweep_avg=self.cfg[expt_name]['sweep_avg']
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)
        # scan_points = logspace(-2,1.6,25)*1e6
        with SlabFile(self.fname, 'a') as f:
            #f.append_pt('t_pts', t_pt)
            f.append_pt('flux_pts', self.cfg['flux']['current'])
            f.append_line('time_pts', scan_points)
            f.append_pt('read_power', self.cfg['readout']['power'])
            f.append_pt('drive_power', self.cfg['drive']['power'])
            # only partial info in here..
            f.save_settings(self.pnax.get_settings())


        for avg_num in range(0, sweep_avg):
            print(f'Now running average #{avg_num+1}')
            for ii, scan_pt in tqdm(enumerate(scan_points)):

                drive1_delay = self.pulse2end_delay - drive_width - scan_pt
                self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))

                data = self.pnax.take()
                fpoints_dummy = data[0]
                I = data[1]
                Q = data[2]
                # print(".. taking %d of %d at freq %.6f GHz" % (ii+1, nn+1, scan_pt*1e-9))
                if avg_num+1>1:
                    with SlabFile(self.fname, 'r') as f:
                        I_prev=np.array(f['imag'])
                        Q_prev=np.array(f['real'])
                    I_averaged=average(avg_num*[I_prev[ii]]+[average(I)])
                    Q_averaged=average(avg_num*[Q_prev[ii]]+[average(Q)])
                else:
                    I_averaged = average(I)
                    Q_averaged = average(Q)
                # print(average(mags), mags_averaged, avg_num)
                with SlabFile(self.fname, 'a') as f:
                    f.attrs.modify('trace_avgs', avg_num+1)
                    if avg_num+1>1:
                        f['imag'][ii]=I_averaged
                        f['real'][ii]=Q_averaged
                        mag, phase=polar2mag(I_averaged, Q_averaged)
                        f['mags'][ii]=mag
                        f['phases'][ii]=phase
                    else:
                        f.append_pt('imag', I_averaged)
                        f.append_pt('real', Q_averaged)
                        mag, phase=polar2mag(I_averaged, Q_averaged)
                        f.append_pt('mags', mag)
                        f.append_pt('phases', phase)
        print('\nData file saved:')
        print(self.fname)

    def ramsey(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        drive_freq = self.cfg['qubit']['freq'] +  self.cfg[expt_name]['ramsey_freq']

        # now range 4 (Source2) is coupled, so qfreq is set by changing range1
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % drive_freq)
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % drive_freq)

        drive1_width = self.cfg['pulse_info']['half_pi_time']
        drive2_width = self.cfg['pulse_info']['half_pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))
        self.pnax.write("SENS:PULS3:WIDT %.12f" % (drive2_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive1_width
            drive2_delay = self.pulse3end_delay - drive1_width - scan_pt - drive2_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS3:DEL %.12f" % (drive2_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at ramsey wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    def ef_pulse_probe(self, expt_name):

        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        drive_width=self.cfg[expt_name]["ge_pulse_width"]
        drive_freq=self.cfg[expt_name]["ge_freq"]
        drive_power=self.cfg[expt_name]["ge_power"]

        ge_delay=self.pulse3end_delay - self.cfg['drive']['width']-self.cfg[expt_name]['ge_pulse_width']

        self.pnax.write("SENS:PULS3:WIDT %.12f" %(drive_width*1e-9))
        self.pnax.write("SENS:PULS3:DEL %.12f" %(ge_delay*1e-9))

        if self.LSG_state==True:
            self.LSG.set_output(False)
            self.LSG.set_use_internal_reference(False)
            self.LSG.set_frequency(int(drive_freq))
            self.LSG.set_power(drive_power)
            time.sleep(1.0)
            self.LSG.set_output(True)

            print('g-e drive power: %.1f' % self.LSG.get_power())
            print('g-e drive frequency %.1f' % self.LSG.get_frequency())
            print('g-e drive output: on')
        else:
            raise Exception("Labbrick is not initialized, cannot generate ge drive.")

        for ii, scan_pt in enumerate(scan_points):

            # now range 4 (Source2) is coupled, so qfreq is set by changing range1
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % scan_pt)
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % scan_pt)

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at freq %.6f GHz" % (ii + 1, nn + 1, scan_pt * 1e-9))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_line('drivefreq_pts', scan_points)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        self.LSG.set_output(False)
        print('\nData file saved:')
        print(self.fname)



    def spin_echo(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        drive_freq = self.cfg['qubit']['freq'] + self.cfg[expt_name]['ramsey_freq']

        # now range 4 (Source2) is coupled, so qfreq is set by changing range1
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % drive_freq)
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % drive_freq)

        drive1_width = self.cfg['pulse_info']['half_pi_time']
        drive2_width = self.cfg['pulse_info']['pi_time']
        drive3_width = self.cfg['pulse_info']['half_pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))
        self.pnax.write("SENS:PULS3:WIDT %.12f" % (drive2_width * 1e-9))
        self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive1_width
            drive2_delay = self.pulse3end_delay - drive1_width - scan_pt/2.0 - drive2_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt - drive2_width - drive3_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS3:DEL %.12f" % (drive2_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at spin echo total wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    def histogram(self, expt_name, **kwargs):
        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['num_pts'])
        t_pt = time.time()

        if self.cfg[expt_name]['use_pi_time']:
            drive_width = self.cfg['pulse_info']['pi_time']
            print("Drive On. Using Pi Pulse")
        else:
            drive_width = self.cfg['drive']['width']
            print("Drive On. Not Using Pi Pulse")
        if self.cfg[expt_name]['drive_state_off']:
            drive_width = 0.0
            print("Drive OFF")

        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9





        drive1_delay = self.pulse2end_delay - drive_width - start
        self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
        print("Taking Data now")
        data = self.pnax.take_one_in_mag_phase()
        fpoints_dummy = data[0]
        mags = data[1]
        phases = data[2]
        print("Finished Taking Data")


        with SlabFile(self.fname, 'a') as f:
            #f.append_pt('t_pts', t_pt)
            f.append_pt('flux_pts', self.cfg['flux']['current'])
            # f.append_pt('time_pts', scan_pt)
            f.append_line('mags', mags)
            f.append_line('phases', phases)
            f.append_pt('read_power', self.cfg['readout']['power'])
            f.append_pt('drive_power', self.cfg['drive']['power'])
            f.append_pt('readout_time', self.cfg['readout']['width'])
            # only partial info in here..
            f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    def raman(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        t_pt = time.time()

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)
        pump_freq = self.cfg['pump_drive']['freq']
        pump_power = self.cfg['pump_drive']['power']

        self.drive2.set_output(True)
        self.drive2.set_frequency(pump_freq)
        self.drive2.set_power(pump_power)

        for ii, scan_pt in enumerate(scan_points):
            # now range 4 (Source2) is coupled, so qfreq is set by changing range1
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % scan_pt)
            self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % scan_pt)

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at freq %.6f GHz" % (ii + 1, nn + 1, scan_pt * 1e-9))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_line('drivefreq_pts', scan_points)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])
                f.append_pt('pump_power',self.cfg['pump_drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        self.drive2.set_output(False)
        print(self.fname)

    def raman_rabi(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        pump_freq = self.cfg['pump_drive']['freq']
        pump_power = self.cfg['pump_drive']['power']

        self.drive2.set_output(True)
        self.drive2.set_frequency(pump_freq)
        self.drive2.set_power(pump_power)

        t_pt = time.time()

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            self.pnax.write("SENS:PULS2:WIDT %.12f" % (scan_pt * 1e-9))

            drive1_delay = self.pulse2end_delay - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at rabi duration %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])
                f.append_pt('pump_power',self.cfg['pump_drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        self.drive2.set_output(False)
        print(self.fname)

    def raman_t1(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        if self.cfg[expt_name]['use_pi_time']:
            drive_width = self.cfg['pulse_info']['pi_time']
        else:
            drive_width = self.cfg['drive']['width']

        print ("drive width = ",drive_width,"ns")

        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)
        # scan_points = logspace(-2,1.6,25)*1e6

        pump_freq = self.cfg['pump_drive']['freq']
        pump_power = self.cfg['pump_drive']['power']

        self.drive2.set_output(True)
        self.drive2.set_frequency(pump_freq)
        self.drive2.set_power(pump_power)

        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive_width - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at t1 wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])
                f.append_pt('pump_power',self.cfg['pump_drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        self.drive2.set_output(False)
        print(self.fname)

    def raman_ramsey(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        drive_freq = self.cfg['qubit']['freq'] +  self.cfg[expt_name]['ramsey_freq']

        # now range 4 (Source2) is coupled, so qfreq is set by changing range1
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % drive_freq)
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % drive_freq)

        drive1_width = self.cfg['pulse_info']['half_pi_time']
        drive2_width = self.cfg['pulse_info']['half_pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))
        self.pnax.write("SENS:PULS3:WIDT %.12f" % (drive2_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        pump_freq = self.cfg['pump_drive']['freq']
        pump_power = self.cfg['pump_drive']['power']


        self.drive2.set_output(True)
        self.drive2.set_frequency(pump_freq)
        self.drive2.set_power(pump_power)


        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive1_width
            drive2_delay = self.pulse3end_delay - drive1_width - scan_pt - drive2_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS3:DEL %.12f" % (drive2_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at ramsey wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])
                f.append_pt('pump_power',self.cfg['pump_drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        self.drive2.set_output(False)
        print(self.fname)

    def raman_spin_echo(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])
        t_pt = time.time()

        drive_freq = self.cfg['qubit']['freq'] + self.cfg[expt_name]['ramsey_freq']

        # now range 4 (Source2) is coupled, so qfreq is set by changing range1
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:START %f' % drive_freq)
        self.pnax.write('SENSE:FOM:RANGE1:FREQUENCY:STOP %f' % drive_freq)

        drive1_width = self.cfg['pulse_info']['half_pi_time']
        drive2_width = self.cfg['pulse_info']['pi_time']
        drive3_width = self.cfg['pulse_info']['half_pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))
        self.pnax.write("SENS:PULS3:WIDT %.12f" % (drive2_width * 1e-9))
        self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        pump_freq = self.cfg['pump_drive']['freq']
        pump_power = self.cfg['pump_drive']['power']

        self.drive2.set_output(True)
        self.drive2.set_frequency(pump_freq)
        self.drive2.set_power(pump_power)

        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive1_width
            drive2_delay = self.pulse3end_delay - drive1_width - scan_pt/2.0 - drive2_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt - drive2_width - drive3_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS3:DEL %.12f" % (drive2_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at spin echo total wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                #f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])
                f.append_pt('pump_power',self.cfg['pump_drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        self.drive2.set_output(False)
        print(self.fname)

    # the following tests are done with pulse4 connected to ttl of rf generator used for rf flux modulation
    # this is turning rf on, then off, wait and drive pi pulse at LSS
    def alex_test_1(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        drive1_width = self.cfg['pulse_info']['pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))

        drive3_width = self.cfg[expt_name]['p4_width']
        self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive1_delay = self.pulse2end_delay - drive1_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt - drive3_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at alex_test wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    # same as test1, but keep start of rf mod pulse fixed (shorten while scan_pt increases)
    def alex_test_1b(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        drive1_width = self.cfg['pulse_info']['pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive3_width = self.cfg[expt_name]['p4_width'] - scan_pt
            self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

            drive1_delay = self.pulse2end_delay - drive1_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt - drive3_width
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at alex_test wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    # this is turnig rf mod on, wait, pi and readout then off
    def alex_test_2(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        drive1_width = self.cfg['pulse_info']['pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive3_width = scan_pt + drive1_width + \
                           self.cfg['readout']['width'] + self.cfg['readout']['delay']
            self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

            drive1_delay = self.pulse2end_delay - drive1_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at alex_test wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    # this is opposite of above: turnig rf mod off, wait, pi and readout then on.
    # i.e. rf mod is always on except when for pi_cal and readout, none by inverting the ttl
    def alex_test_3(self, expt_name, **kwargs):

        # invert pulse4 (rf mod)
        self.pnax.write("SENS:PULS4:INV 1")

        # setting average
        self.pnax.set_sweep_points(self.cfg[expt_name]['averages'])

        drive1_width = self.cfg['pulse_info']['pi_time']
        self.pnax.write("SENS:PULS2:WIDT %.12f" % (drive1_width * 1e-9))

        start = self.cfg[expt_name]['start']  # * 1e-9
        stop = self.cfg[expt_name]['stop']  # * 1e-9
        step = self.cfg[expt_name]['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):

            drive3_width = scan_pt + drive1_width + \
                           self.cfg['readout']['width'] + self.cfg['readout']['delay']
            self.pnax.write("SENS:PULS4:WIDT %.12f" % (drive3_width * 1e-9))

            drive1_delay = self.pulse2end_delay - drive1_width
            drive3_delay = self.pulse4end_delay - drive1_width - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))
            self.pnax.write("SENS:PULS4:DEL %.12f" % (drive3_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at alex_test wait of %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)

    # this is modified rabi
    # added pulse 4 to control rfmod
    def alex_test(self, expt_name, **kwargs):

        # setting average
        self.pnax.set_sweep_points(self.cfg['rabi']['averages'])

        t_pt = time.time()

        start = self.cfg['rabi']['start']  # * 1e-9
        stop = self.cfg['rabi']['stop']  # * 1e-9
        step = self.cfg['rabi']['step']  # * 1e-9
        nn = round((stop - start) / step)
        scan_points = linspace(start, nn * step + start, nn + 1)

        for ii, scan_pt in enumerate(scan_points):
            self.pnax.write("SENS:PULS2:WIDT %.12f" % (scan_pt * 1e-9))

            drive1_delay = self.pulse2end_delay - scan_pt
            self.pnax.write("SENS:PULS2:DEL %.12f" % (drive1_delay * 1e-9))

            data = self.pnax.take_one_in_mag_phase()
            fpoints_dummy = data[0]
            mags = data[1]
            phases = data[2]
            print(".. taking %d of %d at rabi duration %.6f ns" % (ii + 1, nn + 1, scan_pt))
            mags_averaged = average(mags)
            phases_averaged = average(phases)

            with SlabFile(self.fname, 'a') as f:
                # f.append_pt('t_pts', t_pt)
                f.append_pt('flux_pts', self.cfg['flux']['current'])
                f.append_pt('time_pts', scan_pt)
                f.append_pt('mags', mags_averaged)
                f.append_pt('phases', phases_averaged)
                f.append_pt('read_power', self.cfg['readout']['power'])
                f.append_pt('drive_power', self.cfg['drive']['power'])

                # only partial info in here..
                f.save_settings(self.pnax.get_settings())

        print('\nData file saved:')
        print(self.fname)