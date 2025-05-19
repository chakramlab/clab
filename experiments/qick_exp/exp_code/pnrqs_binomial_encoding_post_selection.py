import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PNRQSBinomialEncodingPostSelectionProgram(AveragerProgram):
    def initialize(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # --- Readout parameters
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_ch_nyquist = cfg.device.soc.resonator.nyqist
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.readout_freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.adc_trig_offset = self.us2cycles(cfg.device.soc.readout.adc_trig_offset)
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # --- Sideband parameters

        self.sideband_ch = cfg.device.soc.sideband.ch
        self.sideband_nyquist = cfg.device.soc.sideband.nyqist

        # --- Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.qubit_ch = self.q_ch
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist
        
        self.qubit_freq_placeholder = self.freq2reg(cfg.expt.freq_placeholder)
        self.qubit_gf_freq = self.freq2reg(cfg.device.soc.qubit.f_gf, gen_ch = self.sideband_ch)
        self.qubit_gf_gain = cfg.device.soc.qubit.pulses.pi_gf.gain

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.q_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

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

        # self.chi_e = self.cfg.expt.chi_e
        # self.chi_ef = self.cfg.expt.chi_ef

        # print ("chi_e = ", self.chi_e, "chi_ef = ", self.chi_ef, "MHz")

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
        if self.cfg.expt.qubit_prep_pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_prep", sigma=self.us2cycles(self.cfg.expt.qubit_prep_length), length=self.us2cycles(self.cfg.expt.qubit_prep_length) * 4)

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
            print('Initializing transmon+cavity reset')
            self.cfg.device.soc.readout.reset_cavity_n =  1

            for ii in range(cfg.device.soc.readout.reset_cycles):

                # print('Resetting System,', 'Cycle', ii)

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

                    # print('Resetting cavity for n =', ii)

                    # setup and play f,n g,n+1 sideband pi pulse

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][ii]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][ii]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][ii]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][ii]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
                    # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
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

        # qubit_ge

        print("qubit_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.expt.qubit_prep_gain)
        print('Length:', self.cfg.expt.qubit_prep_length)
        print('Phase:', self.cfg.expt.qubit_prep_phase)

        if self.cfg.expt.qubit_prep_pulse_type == 'const':

            self.set_pulse_registers(
                ch=self.q_ch, 
                style="const", 
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(self.cfg.expt.qubit_prep_phase),
                gain=self.cfg.expt.qubit_prep_gain, 
                length=self.us2cycles(self.cfg.expt.qubit_prep_length))
        
        if self.cfg.expt.qubit_prep_pulse_type == 'gauss':
            print('playing gaussian qubit prep')
            self.set_pulse_registers(
                    ch=self.qubit_ch,
                    style="arb",
                    freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                    phase=self.deg2reg(self.cfg.expt.qubit_prep_phase),
                    gain=self.cfg.expt.qubit_prep_gain,
                    waveform="qubit_prep")
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # Encoding Operation

        print('Encoding Operation')

        # 1. pi_ef
        
        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.play_pief_pulse(phase = 0)
        self.sync_all()

        # 2. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
        
        self.play_pige_pulse()
        self.sync_all()

        # 3. pi_f0g1

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

        # 4. pi_ge (shift by chi_e/2)

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.play_pige_pulse(shift = self.chi_e/2)
        self.sync_all()

        # 5. pi_ef

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.play_pief_pulse(shift=self.chi_ef)
        self.sync_all()

        # 6. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.play_pige_pulse()
        self.sync_all()

        # 7. pi2_f1g2

        print("pi2_f1g2")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][1])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi2_fngnp1_gains[self.cfg.expt.mode][1])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi2_fngnp1_times[self.cfg.expt.mode][1])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][1])

        self.play_sb(
            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][1], 
            length=self.cfg.device.soc.sideband.pulses.pi2_fngnp1_times[self.cfg.expt.mode][1],
            gain=self.cfg.device.soc.sideband.pulses.pi2_fngnp1_gains[self.cfg.expt.mode][1],
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][1])

        self.sync_all()

        # 8. pi_ge (shift by chi_e)

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.play_pige_pulse(shift = self.chi_e)
        self.sync_all()

        # 9. pi_ef  (shift by 3*chi_ef/2)

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.play_pief_pulse(shift = 3*self.chi_ef/2.0)
        self.sync_all()

        # 10. pi_f2g3

        print("pi_f2g3")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][2])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][2])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][2])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][2])
        print('Pulse Type:', self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode])
        print('Ramp Type:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode])
        
        self.play_sb(
            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][2], 
            length=self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][2],
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][2],
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][2])

        self.sync_all()

        # 11. pi_ge (shift by 4*chi_e/3)

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.play_pige_pulse(shift = 4*self.chi_e/3)
        self.sync_all()

        # 12. pi_f0g1

        print("pi_f0g1")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][0])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][0])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][0])
        print('Pulse Type:', self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode])
        print('Ramp Type:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode])

        self.play_sb(
            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][0],
            length=self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][0],
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][0],
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][0])
        self.sync_all()

        # 13. pi_ef (shift by 3*chi_ef/2)

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.play_pief_pulse(shift = 3*self.chi_ef/2.0)
        self.sync_all()

        # 14. pi_f0g1 and 4pi_f3g4

        print("pi_f0g1 and 4pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_freqs[0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_gains[0])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_times[0])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_ramp_sigmas[0])

        self.play_sb(
            freq = self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_freqs[0],
            length=self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_times[0],
            gain=self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_gains[0],
            ramp_sigma=self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_ramp_sigmas[0],
            pulse_type = 'flat_top',
            ramp_type = 'bump'
        )
        self.sync_all()

        # 15. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.play_pige_pulse(self.chi_e/2)
        self.sync_all()

        # 16. pi_f3g4

        print("pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][3])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][3])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][3])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][3])
        
        self.play_sb(
            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][3],
            length=self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][3],
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][3],
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][3])
        self.sync_all()

        # 17. pi_ef (shift by chi_ef)

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.play_pief_pulse(shift = self.chi_ef)
        self.sync_all()

        # 18. pi_f1g2 and 2pi_f3g4

        print("pi_f1g2 and 2pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_freqs[0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_gains[0])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_times[0])
        print('Ramp Sigma:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_ramp_sigmas[0])

        self.play_sb(
            freq = self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_freqs[0],
            length=self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_times[0],
            gain=self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_gains[0],
            ramp_sigma=self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_ramp_sigmas[0],
            pulse_type = 'flat_top',
            ramp_type = 'bump')

        self.sync_all()

        # Post-selection measurement of transmon
        
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
        
        # PNQRS 
        
        self.sigma_ge_resolved = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)

        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge_resolved", sigma=self.sigma_ge_resolved, length=self.sigma_ge_resolved * 4)
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.qubit_freq_placeholder,
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge_resolved")
        
        if self.qubit_pulsetype == 'const':
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.qubit_freq_placeholder,
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge_resolved)
            
        self.pulse(ch=self.q_ch)
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

        chi_e = cfg.device.soc.storage.chi_e[cfg.expt.mode]
        chi_f = cfg.device.soc.storage.chi_f[cfg.expt.mode]

        # System Reset

        if cfg.expt.reset:

            self.cfg.device.soc.readout.reset_cavity_n = 4

            for jj in range(cfg.device.soc.readout.reset_cycles):

                print('Reset cycle:', jj)

                for kk in range(cfg.device.soc.readout.reset_cavity_n, 0, -1):
                    
                    print('Resetting cavity for N =', kk)

                    # Cavity Reset for |gN>

                    for ii in range(kk-1, -1, -1):

                        print('Pulse for gn=', ii+1)
                        
                        if self.cfg.expt.chi_correction:
                            # print('chi_ge_cor', chi_e * ii)
                            # print('chi_ef_cor', (chi_f - chi_e) * ii)
                            chi_ge_cor = chi_e * ii
                            chi_ef_cor = (chi_f - chi_e) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f * (ii)
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                    # Cavity Reset for |eN>

                    # pi_ef

                    self.play_pief_pulse(shift = (kk)*chi_ef_cor)
                    self.sync_all()

                    # f0g1 with N photons in storage cavity to readout mode

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f * (kk)
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                    # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    
                    self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                    self.sync_all()

                    for ii in range(kk-1, -1, -1):
                        print('Pulse for en=', ii+1)
                        if self.cfg.expt.chi_correction:
                            # print('chi_ge_cor', chi_e * ii)
                            # print('chi_ef_cor', (chi_f - chi_e) * ii)
                            chi_ge_cor = chi_e * ii
                            chi_ef_cor = (chi_f - chi_e) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f * (ii)
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                        print('chi_f', chi_f)
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                        self.sync_all()
                
                self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg

        shots_i0 = self.di_buf[0].reshape((cfg.expt.reps, cfg.expt.n_meas)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        shots_q0 = self.dq_buf[0].reshape((cfg.expt.reps, cfg.expt.n_meas)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        return shots_i0, shots_q0
    
class PNRQSBinomialEncodingPostSelectionExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        
        avgi_col = []
        avgq_col = []
        i_shots_col = []
        q_shots_col = []

        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        for i in tqdm(fpts, disable = not progress):
            self.cfg.expt.freq_placeholder = i
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            qspec=PNRQSBinomialEncodingPostSelectionProgram(soc, self.cfg)
            avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False) 
            
            i_shots, q_shots = qspec.collect_shots()
            i_shots_col.append(i_shots)
            q_shots_col.append(q_shots)
            avgi_col.append(avgi[0])
            avgq_col.append(avgq[0])      
        
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

        data_dict = {'xpts':fpts, 'i_g': i_g, 'q_g': q_g, 'i_e': i_e, 'q_e': q_e, 'i_f':i_f, 'q_f':q_f, 'avgi': avgi_col, 'avgq': avgq_col, 'i_shots': i_shots_col, 'q_shots': q_shots_col}

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
        