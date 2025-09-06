import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class QubitCavityTomographyTestProgram(AveragerProgram):
    def initialize(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # Readout parameters
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_ch_nyquist = cfg.device.soc.resonator.nyqist
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.readout_ch = cfg.device.soc.readout.ch
        self.readout_freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.res_gain = cfg.device.soc.resonator.gain
        self.adc_trig_offset = self.us2cycles(cfg.device.soc.readout.adc_trig_offset)
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # Qubit parameters
        self.qubit_ch=cfg.device.soc.qubit.ch
        self.q_ch = self.qubit_ch
        self.q_reg_page =self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.qubit_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.qubit_ch, "phase")
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist

        # Cavity drive parameters
        self.cavdr_ch=cfg.device.soc.storage.ch
        self.cavdr_reg_page =self.ch_page(self.cavdr_ch)     # get register page for cavdr_ch
        self.cavdr_freq_reg = self.sreg(self.cavdr_ch, "freq")   # get frequency register for cavdr_ch
        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        if self.cfg.expt.pulse_info_from_json == True:
            print ("pulse_info_from_config_json = True")
            self.cavdr_length = self.cfg.device.soc.storage.tomography_pulse_lens[self.cfg.expt.mode]
            self.cavdr_freq = self.cfg.device.soc.storage.freqs[self.cfg.expt.mode]
            self.cavdr_pulse_type = self.cfg.device.soc.storage.tomography_pulse_types[self.cfg.expt.mode]
            self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
            print ("for mode ", self.cfg.expt.mode, "with freq = ", self.cavdr_freq, "MHz", "chi_e = ", self.chi_e, "MHz")
            print ("cavdr_length = ", self.cavdr_length, "us")
            print ("cavdr_pulse_type = ", self.cavdr_pulse_type)
            self.qubit_pi2_waittime = self.us2cycles(1/2/np.abs(self.chi_e))
        else:
            self.cavdr_length = cfg.expt.length
            self.cavdr_freq = cfg.expt.cavity_drive_freq
            self.cavdr_pulse_type = self.cfg.expt['cavdr_pulse_type']
            self.qubit_pi2_waittime = self.us2cycles(cfg.expt.waittime)
            print ("for mode ", self.cfg.expt.mode, "with freq = ", self.cavdr_freq, "MHz")
            print ("cavdr_length = ", self.cavdr_length, "us")
            print ("cavdr_pulse_type = ", self.cavdr_pulse_type)

        # Sideband drive parameters

        self.sideband_ch = cfg.device.soc.sideband.ch
        self.sideband_nyquist =cfg.device.soc.sideband.nyqist

        # Wigner tomography sweep
        self.cavdr_gain = cfg.expt.cavdr_gain_temp
        self.cavdr_phase = cfg.expt.cavdr_phase_temp

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)
        self.declare_gen(ch=self.cavdr_ch, nqz=2)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)
        
        try:
            self.delay_before_tomography = self.us2cycles(cfg.expt.delay_before_tomography)
        except:
            self.delay_before_tomography = self.us2cycles(0.0)

        #initializing pulse register
        # add qubit and readout pulses to respective channels
            
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        # taking chi_e and chi_ef just from the experiment config

        if self.cfg.expt.shift_qubit_pulses == True:
            self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
            self.chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
            self.chi_ef = self.chi_f - self.chi_e
            print("chi_e = ", self.chi_e, "chi_f = ", self.chi_f, "MHz")

        else:
            self.chi_e = 0 
            self.chi_f = 0
            self.chi_ef = 0


        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
        if self.cfg.expt.qubit_prep_pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_prep", sigma=self.us2cycles(self.cfg.expt.qubit_prep_length), length=self.us2cycles(self.cfg.expt.qubit_prep_length) * 4)
        if self.cavdr_pulse_type == 'gauss':
            print ("cavdr_pulse_type = gauss")
            self.add_gauss(ch=self.cavdr_ch, name="cavdr", sigma=self.us2cycles(self.cavdr_length), length=self.us2cycles(self.cavdr_length)* 4)

        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(self.cfg.expt.sb_ramp_sigma), length=self.us2cycles(self.cfg.expt.sb_ramp_sigma * 4))
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2)
        self.add_bump_func(ch=self.sideband_ch, name="sb_flat_top_bump", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2, k=2, flat_top_fraction=0.0)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.readout_freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)
        
        print('new settings! 5')
        self.synci(500)  # give processor some time to configure pulses

    def play_pige_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
                    length=self.sigma_ge)
            
        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.qubit_ch)

    def play_piby2ge(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.sigma_ge2)
            
        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.qubit_ch)

    def play_pief_pulse(self, phase = 0, shift = 0):
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                    length=self.sigma_ef)
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        self.pulse(ch=self.qubit_ch)   

    def play_cavity_drive(self, gain = 0, length = 1, phase = 0):
            
        if self.cavdr_pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.cavdr_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cavdr_freq, gen_ch=self.cavdr_ch),
                    phase=self.deg2reg(phase, gen_ch=self.cavdr_ch),
                    gain=gain, 
                    length= self.us2cycles(length))
            
        if self.cavdr_pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.cavdr_ch,
                style="arb",
                freq=self.freq2reg(self.cavdr_freq, gen_ch=self.cavdr_ch),
                phase=self.deg2reg(phase,gen_ch=self.cavdr_ch),
                gain=gain,
                waveform="cavdr")
        
        self.pulse(ch=self.cavdr_ch)
    
    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='bump', ramp_sigma=0.01, phase=0, shift=0):

        if pulse_type == 'const':
            
            print('Sideband const')
            self.set_pulse_registers(
                    ch=self.sideband_ch, 
                    style="const", 
                    freq=self.freq2reg(freq+shift), 
                    phase=self.deg2reg(phase),
                    gain=gain, 
                    length=self.us2cycles(length))
        
        if pulse_type == 'flat_top':
            
            if ramp_type == 'sin_squared':
                # print('Sideband flat top sin squared')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_sin_squared")

            elif ramp_type == 'bump':
                print('Sideband flat top bump')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_bump")
                
            elif ramp_type == 'gaussian':
                print('Sideband flat top gaussian')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_gaussian")
        
        self.pulse(ch=self.sideband_ch)

    def body(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)
        
        # Reset |1> in cavity

        # System Reset

        if cfg.expt.cavity_reset_beginning:

            self.cfg.device.soc.readout.reset_cavity_n =  1

            for ii in range(cfg.device.soc.readout.reset_cycles):

                print('Resetting System,', 'Cycle', ii)

                # Transmon Reset

                # f0g1 to readout mode

                sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                
                self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                self.sync_all()

                # pi_ef

                self.play_pief_pulse()
                self.sync_all()

                # f0g1 to readout mode

                sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                
                self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                self.sync_all()

                # Cavity Reset

                for ii in range(self.cfg.device.soc.readout.reset_cavity_n-1, -1, -1):
                    
                    if self.cfg.expt.chi_correction:

                        print('chi_ge_cor', self.chi_e * ii)
                        print('chi_ef_cor', (self.chi_ef * ii))
                        chi_ge_cor = self.chi_e * ii
                        chi_ef_cor = self.chi_ef * ii
                    else:
                        chi_ge_cor = 0
                        chi_ef_cor = 0

                    print('Resetting cavity for n =', ii)

                    # setup and play f,n g,n+1 sideband pi pulse

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][ii]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][ii]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][ii]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][ii]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
                    print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                    self.sync_all()

                    # Transmon Reset

                    # f0g1 to readout mode

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                    # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    
                    self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                    self.sync_all()

                    # pi_ef

                    self.play_pief_pulse(shift=chi_ef_cor)
                    self.sync_all()

                    # f0g1 to readout mode

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                    # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    
                    self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                    self.sync_all()

        # State Preparation

        print("State Preparation")
        
        # 1. pi2_ge

        self.play_piby2ge()
        self.sync_all()

        # # 3. pi_ge

        # print("pi_ge")
        # print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        # print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        # print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
        
        # self.play_pige_pulse()
        # self.sync_all()

        # 2. pi_ef
        

        self.play_pief_pulse(phase = 0)
        self.sync_all()

        # 3. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
        
        self.play_pige_pulse()
        self.sync_all()

        # 4. pi_f0g1

        print("pi_f0g1")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][0])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][0])
        
        self.play_sb(
            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][0], 
            length=self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][0],
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][0],
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][0])

        self.sync_all()


        # # 1. pi2_ge

        # self.play_piby2ge()
        # self.sync_all()

        # # 1. pi2_ge

        # self.play_piby2ge()
        # self.sync_all()
        
        # 3. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
        
        self.play_pige_pulse()
        self.sync_all()

     

        self.sync_all(self.delay_before_tomography)  # added a delay before tomography
        
        # Post-selection measurement of transmon
        if self.cfg.expt.tomography_pulsetype == 'pi2_x':

            if self.cfg.expt.pi_flip:
                self.play_pige_pulse(phase=0)
                self.sync_all()

            self.play_piby2ge(phase = 0)
            self.sync_all()

        elif self.cfg.expt.tomography_pulsetype == 'pi2_y':

            if self.cfg.expt.pi_flip:
                self.play_pige_pulse(phase=90)
                self.sync_all()

            self.play_piby2ge(phase = 90)
            self.sync_all()
        
        else:
            
            if self.cfg.expt.pi_flip:
                self.play_pige_pulse(phase=0)
                self.sync_all()

        # Readout kick pulse

        if self.cfg.device.soc.readout.kick_pulse:
            print('Playing kick pulse')
            self.set_pulse_registers(
                ch=self.cfg.device.soc.resonator.ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.readout.kick_pulse_gain,
                length=self.us2cycles(self.cfg.device.soc.readout.kick_pulse_length))
            
            self.pulse(ch=self.cfg.device.soc.resonator.ch)
            self.sync_all()

        # Readout 

        self.set_pulse_registers(
            ch=self.cfg.device.soc.resonator.ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.resonator.gain,
            length=self.us2cycles(self.cfg.device.soc.readout.length, gen_ch=self.cfg.device.soc.resonator.ch))
        
        self.measure(pulse_ch=self.cfg.device.soc.resonator.ch,
                     adcs=[0],
                     adc_trig_offset=self.us2cycles(self.cfg.device.soc.readout.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg.device.soc.readout.readout_reset_wait_time))  # sync all channels
        
        # Reset of readout cavity
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.readout_freq, 
            phase=self.deg2reg(cfg.device.soc.readout.readout_reset_phase, gen_ch=self.res_ch), # 0 degrees
            gain=self.cfg.device.soc.readout.readout_reset_gain, 
            length=self.us2cycles(cfg.device.soc.readout.readout_reset_length))
        self.pulse(ch=self.res_ch)
        self.sync_all(self.us2cycles(cfg.device.soc.readout.post_selection_wait_time))
        print(
            'Readout reset relax time: ', cfg.device.soc.readout.post_selection_wait_time, 'us',
            'Readout reset wait time: ', cfg.device.soc.readout.readout_reset_wait_time, 'us',
            'Readout reset phase: ', cfg.device.soc.readout.readout_reset_phase, 
            'Readout reset gain: ', cfg.device.soc.readout.readout_reset_gain, 
            'Readout reset length: ', cfg.device.soc.readout.readout_reset_length, 'us')
        
        # Cavity displacement
            
        self.play_cavity_drive(gain = self.cavdr_gain, length = self.cavdr_length, phase = self.cavdr_phase)
        # self.set_pulse_registers(
        #     ch=self.cavdr_ch,
        #     style="const",
        #     freq=self.cavdr_freq,
        #     phase=self.deg2reg(self.cavdr_phase, gen_ch=self.cavdr_ch), # 0 degrees
        #     gain=self.cavdr_gain,
        #     length=self.cavdr_length)
        
        # self.pulse(ch=self.cavdr_ch)  
        self.sync_all()
        
        # Parity Measurement
        
        # Qubit pi/2 pulse
        self.play_piby2ge()
        self.sync_all(self.qubit_pi2_waittime)  # Wait for time tau = chi/2

        self.pulse(ch=self.q_ch)  # Qubit pi/2 pulse
        self.sync_all()

        # Readout kick pulse

        if self.cfg.device.soc.readout.kick_pulse:
            print('Playing kick pulse')
            self.set_pulse_registers(
                ch=self.cfg.device.soc.resonator.ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.readout.kick_pulse_gain,
                length=self.us2cycles(self.cfg.device.soc.readout.kick_pulse_length))
            
            self.pulse(ch=self.cfg.device.soc.resonator.ch)
            self.sync_all()

        # Readout 

        self.set_pulse_registers(
            ch=self.cfg.device.soc.resonator.ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.resonator.gain,
            length=self.us2cycles(self.cfg.device.soc.readout.length, gen_ch=self.cfg.device.soc.resonator.ch))
        
        self.measure(pulse_ch=self.cfg.device.soc.resonator.ch,
                     adcs=[0],
                     adc_trig_offset=self.us2cycles(self.cfg.device.soc.readout.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg.device.soc.readout.relax_delay))  # sync all channels

    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg

        shots_i0 = self.di_buf[0].reshape((cfg.expt.reps, cfg.expt.n_meas)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        shots_q0 = self.dq_buf[0].reshape((cfg.expt.reps, cfg.expt.n_meas)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        return shots_i0, shots_q0
    
class QubitCavityTomographyTestExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='WignerTomography', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        cavdr_gains = list(self.cfg.expt.cavdr_gains)
        cavdr_phases = list(self.cfg.expt.cavdr_phases)

        avgi_col = []
        avgq_col = []
        i_shots_col = []
        q_shots_col = []
        avgi_pi_flip_col = []
        avgq_pi_flip_col = []
        i_shots_pi_flip_col = []
        q_shots_pi_flip_col = []
        avgi_x_col = []
        avgq_x_col = []
        i_shots_x_col = []
        q_shots_x_col = []
        avgi_x_pi_flip_col = []
        avgq_x_pi_flip_col = []
        i_shots_x_pi_flip_col = []
        q_shots_x_pi_flip_col = []
        avgi_y_col = []
        avgq_y_col = []
        i_shots_y_col = []
        q_shots_y_col = []
        avgi_y_pi_flip_col = []
        avgq_y_pi_flip_col = []
        i_shots_y_pi_flip_col = []
        q_shots_y_pi_flip_col = []

        for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
            self.cfg.expt.cavdr_gain_temp = i
            self.cfg.expt.cavdr_phase_temp = j
            self.cfg.expt.tomography_pulsetype = 'I'
            self.cfg.expt.pi_flip = False
            print('Gain = ', i, 'Phase = ', j)
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
            i_shots, q_shots = wigtom.collect_shots()
            i_shots_col.append(i_shots)
            q_shots_col.append(q_shots)
            avgi_col.append(avgi[0])
            avgq_col.append(avgq[0]) 
        
        # pi_ge before post-selection measurement
        for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
            self.cfg.expt.cavdr_gain_temp = i
            self.cfg.expt.cavdr_phase_temp = j
            self.cfg.expt.tomography_pulsetype = 'I'
            self.cfg.expt.pi_flip = True
            print('Gain = ', i, 'Phase = ', j)
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
            i_shots, q_shots = wigtom.collect_shots()
            i_shots_pi_flip_col.append(i_shots)
            q_shots_pi_flip_col.append(q_shots)
            avgi_pi_flip_col.append(avgi[0])
            avgq_pi_flip_col.append(avgq[0]) 

        for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
            self.cfg.expt.cavdr_gain_temp = i
            self.cfg.expt.cavdr_phase_temp = j
            self.cfg.expt.tomography_pulsetype = 'pi2_x'
            self.cfg.expt.pi_flip = False
            print('Gain = ', i, 'Phase = ', j)
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
            i_shots, q_shots = wigtom.collect_shots()
            i_shots_x_col.append(i_shots)
            q_shots_x_col.append(q_shots)
            avgi_x_col.append(avgi[0])
            avgq_x_col.append(avgq[0])  

        # # pi_ge before post-selection measurement
        # for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
        #     self.cfg.expt.cavdr_gain_temp = i
        #     self.cfg.expt.cavdr_phase_temp = j
        #     self.cfg.expt.tomography_pulsetype = 'pi2_x'
        #     self.cfg.expt.pi_flip = True
        #     print('Gain = ', i, 'Phase = ', j)
        #     soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        #     wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
        #     avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
        #     i_shots, q_shots = wigtom.collect_shots()
        #     i_shots_x_pi_flip_col.append(i_shots)
        #     q_shots_x_pi_flip_col.append(q_shots)
        #     avgi_x_pi_flip_col.append(avgi[0])
        #     avgq_x_pi_flip_col.append(avgq[0])     
        
        for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
            self.cfg.expt.cavdr_gain_temp = i
            self.cfg.expt.cavdr_phase_temp = j
            self.cfg.expt.pi_flip = False
            self.cfg.expt.tomography_pulsetype = 'pi2_y'
            print('Gain = ', i, 'Phase = ', j)
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
            i_shots, q_shots = wigtom.collect_shots()
            i_shots_y_col.append(i_shots)
            q_shots_y_col.append(q_shots)
            avgi_y_col.append(avgi[0])
            avgq_y_col.append(avgq[0])   
        
        # # pi_ge before post-selection measurement
        # for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
        #     self.cfg.expt.cavdr_gain_temp = i
        #     self.cfg.expt.cavdr_phase_temp = j
        #     self.cfg.expt.pi_flip = True
        #     self.cfg.expt.tomography_pulsetype = 'pi2_y'
        #     print('Gain = ', i, 'Phase = ', j)
        #     soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        #     wigtom=QubitCavityTomographyTestProgram(soc, self.cfg)
        #     avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
        #     i_shots, q_shots = wigtom.collect_shots()
        #     i_shots_y_pi_flip_col.append(i_shots)
        #     q_shots_y_pi_flip_col.append(q_shots)
        #     avgi_y_pi_flip_col.append(avgi[0])
        #     avgq_y_pi_flip_col.append(avgq[0])  

        # Calibrate qubit probability

        hist_data = self.qubit_iq_calib(path=self.path, config_file=self.config_file)

        i_g = hist_data['ig']
        q_g = hist_data['qg']
        i_e = hist_data['ie']
        q_e = hist_data['qe']
        i_f = hist_data['if']
        q_f = hist_data['qf']

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)
    
        data_dict = {
            'cavdr_gains': cavdr_gains, 'cavdr_phases': cavdr_phases, 'i_g': i_g, 'q_g': q_g, 'i_e': i_e, 'q_e': q_e, 'i_f':i_f, 'q_f':q_f, 
            'avgi': avgi_col, 'avgq': avgq_col, 'i_shots': i_shots_col, 'q_shots': q_shots_col,
            'avgi_pi_flip': avgi_pi_flip_col, 'avgq_pi_flip': avgq_pi_flip_col, 'i_shots_pi_flip': i_shots_pi_flip_col, 'q_shots_pi_flip': q_shots_pi_flip_col,
            'avgi_x': avgi_x_col, 'avgq_x': avgq_x_col, 'i_shots_x': i_shots_x_col, 'q_shots_x': q_shots_x_col,
            'avgi_x_pi_flip': avgi_x_pi_flip_col, 'avgq_x_pi_flip': avgq_x_pi_flip_col, 'i_shots_x_pi_flip': i_shots_x_pi_flip_col, 'q_shots_x_pi_flip': q_shots_x_pi_flip_col,
            'avgi_y': avgi_y_col, 'avgq_y': avgq_y_col, 'i_shots_y': i_shots_y_col, 'q_shots_y': q_shots_y_col,
            'avgi_y_pi_flip': avgi_y_pi_flip_col, 'avgq_y_pi_flip': avgq_y_pi_flip_col, 'i_shots_y_pi_flip': i_shots_y_pi_flip_col, 'q_shots_y_pi_flip': q_shots_y_pi_flip_col}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict, create_dataset=True)
        

        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        data['fiti']=dsfit.fitlor(data["fpts"],data['avgi'][0][0])
        data['fitq']=dsfit.fitlor(data["fpts"],data['avgq'][0][0])
        print(data['fiti'], data['fitq'])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data
        print(self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Qubit Spectroscopy",  ylabel="I")
        plt.plot(data["fpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["fpts"], dsfit.lorfunc(data["fiti"], data["fpts"]))
        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Q")
        plt.plot(data["fpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["fpts"], dsfit.lorfunc(data["fitq"], data["fpts"]))
            
        plt.tight_layout()
        plt.show()
        