import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PhotonNumberResolvedQSpecNOONStateStabilizationCoolingTestProgram(AveragerProgram):
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
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # --- Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.qubit_ch=cfg.device.soc.qubit.ch
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist

        self.qubit_freq_placeholder = cfg.expt.freq_placeholder

        # --- Sideband parameters
        self.sideband_ch = cfg.device.soc.sideband.ch

        # --- Initialize pulses ---

        self.cavdr_ch=cfg.device.soc.storage.ch

        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)
        self.declare_gen(ch=self.sideband_ch, nqz=cfg.device.soc.sideband.nyqist)
        self.declare_gen(ch=self.cavdr_ch, nqz=2)
        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)


        # convert frequency to DAC frequency
        self.freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        
        #initializing pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
        if self.cfg.expt.qubit_prep_pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_prep", sigma=self.us2cycles(self.cfg.expt.qubit_prep_length), length=self.us2cycles(self.cfg.expt.qubit_prep_length) * 4)
        # self.add_gauss(ch=self.cavdr_ch, name="cavdr", sigma=self.us2cycles(self.cfg.expt.cavdr_length), length=self.us2cycles(self.cfg.expt.cavdr_length)* 4)
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(self.cfg.expt.sb_ramp_sigma), length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 4)
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2)
        self.add_bump_func(ch=self.sideband_ch, name="sb_flat_top_bump", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2, k=2, flat_top_fraction=0.0)
        # if self.cfg.expt.sb_pulse_type == 'flat_top':
        #     for ii in range(self.cfg.expt.n):
        #         self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_pi"+str(ii), sigma=self.us2cycles(self.cfg.expt.sb_ramp_sigma), length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_fh", sigma=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_fh.sigma), length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_fh.sigma) * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ge_resolved", sigma=self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch), length=self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch) * 4)
        # print('new version')
        self.synci(200)  # give processor some time to configure pulses
        self.synci(200)  # give processor some time to configure pulses

    def play_pifh_pulse(self, phase=0, shift=0):

        if self.cfg.device.soc.qubit.pulses.pi_fh.pulse_type == 'const':
            
            self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh + shift), 
                phase=0,
                gain=self.cfg.device.soc.qubit.pulses.pi_fh.gain, 
                length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_fh.sigma))
        
        if self.cfg.device.soc.qubit.pulses.pi_fh.pulse_type == 'gauss':
    
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh + shift),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_fh.gain,
                waveform="qubit_fh")
        
        self.pulse(ch=self.qubit_ch)

    def play_pi2fh_pulse(self, phase=0, shift=0):

        if self.cfg.device.soc.qubit.pulses.pi_fh.pulse_type == 'const':
            
            self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh + shift), 
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_fh.gain, 
                length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_fh.sigma))
        
        if self.cfg.device.soc.qubit.pulses.pi_fh.pulse_type == 'gauss':
    
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_fh.gain,
                waveform="qubit_fh")
        
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

    def play_pige_pulse(self, phase=0, shift=0):

        self.pulse_type_ge = self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type

        if self.pulse_type_ge == 'const':

            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge+shift), 
                    phase=self.deg2reg(0+phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
                    length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.q_ch))
            
        if self.pulse_type_ge == 'gauss':

            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge+shift),
                phase=self.deg2reg(0+phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.q_ch)

    def play_pief_pulse(self, phase=0, shift=0):

        self.pulse_type_ef = self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type

        if self.pulse_type_ef == 'const':

            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef+shift), 
                    phase=self.deg2reg(0+phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                    length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.q_ch))
            
        if self.pulse_type_ef == 'gauss':

            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef+shift),
                phase=self.deg2reg(0+phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        self.pulse(ch=self.q_ch)

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=0.01, phase=0, shift=0):        

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
                print('Sideband flat top sin squared')
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

    def play_cavity_drive(self, gain = 0, phase = 0):
        
        if self.cfg.expt.cavdr_pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.cavdr_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.expt.cavdr_freq, gen_ch=self.cavdr_ch),
                    phase=self.deg2reg(phase, gen_ch=self.cavdr_ch),
                    gain=gain, 
                    length= self.us2cycles(self.cfg.expt.cavdr_length))
            
        if self.cfg.expt.cavdr_pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.cavdr_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.expt.cavdr_freq, gen_ch=self.cavdr_ch),
                phase=self.deg2reg(phase,gen_ch=self.cavdr_ch),
                gain=gain,
                waveform="cavdr")
        
        self.pulse(ch=self.cavdr_ch)

    def body(self):
        cfg=AttrDict(self.cfg)
    

        print('updated code')
        self.play_pige_pulse()
        self.sync_all()

        self.play_pief_pulse()
        self.sync_all()

        self.play_pi2fh_pulse(phase=self.cfg.expt.stabilize_phase)
        self.sync_all()

        print("pi_h0e1")
        print('Freq.:', self.cfg.expt.h0e1_freq)
        print('Gain:', self.cfg.expt.h0e1_gain)
        print('Length:', self.cfg.expt.h0e1_length)
        
        sb_freq = self.cfg.device.soc.sideband.hnenp1_freqs[self.cfg.expt.mode1][0]
        sb_sigma = self.cfg.device.soc.sideband.pulses.hnenp1pi_times[self.cfg.expt.mode1][0]
        sb_gain = self.cfg.device.soc.sideband.pulses.hnenp1pi_gains[self.cfg.expt.mode1][0]
        sb_pulse_type = self.cfg.device.soc.sideband.pulses.hnenp1pi_pulse_types[self.cfg.expt.mode1]
        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_sigmas[self.cfg.expt.mode1][0]
        sb_ramp_type = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_types[self.cfg.expt.mode1]
        print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
        self.sync_all()

        self.play_pifh_pulse()
        self.sync_all()

        sb_freq = self.cfg.device.soc.sideband.hnenp1_freqs[self.cfg.expt.mode2][0]
        sb_sigma = self.cfg.device.soc.sideband.pulses.hnenp1pi_times[self.cfg.expt.mode2][0]
        sb_gain = self.cfg.device.soc.sideband.pulses.hnenp1pi_gains[self.cfg.expt.mode2][0]
        sb_pulse_type = self.cfg.device.soc.sideband.pulses.hnenp1pi_pulse_types[self.cfg.expt.mode2]
        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_sigmas[self.cfg.expt.mode2][0]
        sb_ramp_type = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_types[self.cfg.expt.mode2]
        print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
        self.sync_all()

        self.play_pief_pulse()
        self.sync_all()

        # f0g1 with 1 photons in storage cavity to readout mode

        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + (cfg.device.soc.storage.chi_f[cfg.expt.mode1] + cfg.device.soc.storage.chi_f[cfg.expt.mode2]) / 2
        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
        print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
        
        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
        self.sync_all()


        # self.play_pige_pulse(shift=(cfg.device.soc.storage.chi_e[cfg.expt.mode1] + cfg.device.soc.storage.chi_e[cfg.expt.mode2]) / 2)
        # self.sync_all()
        
        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(self.qubit_freq_placeholder),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge_resolved")
        
        if self.qubit_pulsetype == 'const':
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(self.qubit_freq_placeholder),
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch))
            
        self.pulse(ch=self.q_ch)

        # sb_freq = self.cfg.expt.h0e1_freq + self.cfg.expt.h0e1_detuning
        # sb_sigma = cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma * 4  # Play for as long as the resolved pi-pulse
        # sb_gain = self.cfg.expt.h0e1_gain
        # sb_pulse_type = self.cfg.expt.h0e1_pulse_type
        # sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_sigmas[self.cfg.expt.mode][0]
        # sb_ramp_type = self.cfg.device.soc.sideband.pulses.hnenp1pi_ramp_types[self.cfg.expt.mode]
        # print('Playing sideband h0e1 pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
        # self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)

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
                     adc_trig_offset=self.us2cycles(self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg.device.soc.readout.relax_delay))  # sync all channels

        # System Reset

        if self.cfg.expt.chi_correction:
            chi_e_mode1 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode1]
            chi_f_mode1 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode1]
            chi_ef_mode1 = chi_f_mode1 - chi_e_mode1
            chi_e_mode2 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode2]
            chi_f_mode2 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode2]
            chi_ef_mode2 = chi_f_mode2 - chi_e_mode2
        else:
            chi_e_mode1 = 0
            chi_f_mode1 = 0
            chi_ef_mode1 = 0
            chi_e_mode2 = 0
            chi_f_mode2 = 0
            chi_ef_mode2 = 0

        if cfg.expt.reset:

            # Mode 1

            self.cfg.device.soc.readout.reset_cavity_n = self.cfg.expt.n

            for jj in range(cfg.device.soc.readout.reset_cycles):

                print('Reset cycle:', jj)
                
                # f0g1 with in storage cavity to readout mode

                sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                
                self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                self.sync_all()

                # Reset |h0>

                self.play_pifh_pulse()
                self.sync_all()

                # Transmon Reset

                # f0g1 with in storage cavity to readout mode

                sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                
                self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                self.sync_all()

                for kk in range(cfg.device.soc.readout.reset_cavity_n, 0, -1):
                    
                    print('Resetting cavity for N =', kk)

                    # Cavity Reset for |gN>

                    for ii in range(kk-1, -1, -1):

                        print('Pulse for gn=', ii+1)
                        
                        if self.cfg.expt.chi_correction:
                            # print('chi_ge_cor', chi_e * ii)
                            # print('chi_ef_cor', (chi_f - chi_e) * ii)
                            chi_ge_cor = chi_e_mode1 * ii
                            chi_ef_cor = (chi_f_mode1 - chi_e_mode1) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode1 * (ii)
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

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode1 * (kk)
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
                            chi_ge_cor = chi_e_mode1 * ii
                            chi_ef_cor = (chi_f_mode1 - chi_e_mode1) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode1 * (ii)
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                        print('chi_f', chi_f_mode1)
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                        self.sync_all()
                
                self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

            # Mode 2

            self.cfg.device.soc.readout.reset_cavity_n = self.cfg.expt.n

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
                            chi_ge_cor = chi_e_mode2 * ii
                            chi_ef_cor = (chi_f_mode2 - chi_e_mode2) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode2 * (ii)
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

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode2 * (kk)
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
                            chi_ge_cor = chi_e_mode2 * ii
                            chi_ef_cor = (chi_f_mode2 - chi_e_mode2) * ii
                        else:
                            chi_ge_cor = 0
                            chi_ef_cor = 0

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                        # Transmon Reset

                        # f0g1 with N photons in storage cavity to readout mode

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0] + chi_f_mode2 * (ii)
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                        print('chi_f', chi_f_mode2)
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                        self.sync_all()

                self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

class PhotonNumberResolvedQSpecNOONStateStabilizationCoolingTestExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        avgi_col = []
        avgq_col = []

        for i in tqdm(fpts, disable = not progress):
            self.cfg.expt.freq_placeholder = i
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            qspec=PhotonNumberResolvedQSpecNOONStateStabilizationCoolingTestProgram(soc, self.cfg)
            avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False) 

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])       
        
        data={'fpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col}
        
        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col, 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        # if data_path and filename:
        #     file_path = data_path + get_next_filename(data_path, filename, '.h5')
        #     with SlabFile(file_path, 'a') as f:
        #         f.append_line('freq', x_pts)
        #         f.append_line('avgi', avgi[0][0])
        #         f.append_line('avgq', avgq[0][0])
        #     print("File saved at", file_path)
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        

        return data

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
        