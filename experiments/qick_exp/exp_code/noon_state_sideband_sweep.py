import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict

class NOONStateSidebandSweepProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        try:
            self.sideband_ch = cfg.device.soc.sideband.ch
            # print('Sideband channel found')
        except:
            self.sideband_ch = self.qubit_ch



        
        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        # self.r_gain = self.sreg(self.qubit_ch, "gain")   # get gain register for qubit_ch    
        
        self.f_res=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])            # conver f_res to dac register value
        self.readout_length= self.us2cycles(self.cfg.device.soc.readout.length)
        


        self.sigma_test = self.us2cycles(self.cfg.expt.length_placeholder)
        # print('Sigma test (us):', self.cfg.expt.length_placeholder)
        # print('Sigma test (cycles)', self.sigma_test)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)


        for ch in [0]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)
        
        # qubit ge and ef pulse parameters

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'

        # print('Pulse type_ge: ' + self.pulse_type_ge)
        # print('Pulse type_ef: ' + self.pulse_type_ef)

        if self.pulse_type_ge == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        
        if self.pulse_type_ef == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
    
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        
        self.sync_all(self.us2cycles(0.2))
    
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

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=0.01, phase=0, shift=0):
        
        # why not add this inside the if statement?
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(ramp_sigma), length=self.us2cycles(ramp_sigma) * 4)
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(ramp_sigma) * 2)

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

    #same
    def body(self):

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

        print('Chi e mode 1:', chi_e_mode1)
        print('Chi f mode 1:', chi_f_mode1)
        print('Chi ef mode 1:', chi_ef_mode1)
        print('Chi e mode 2:', chi_e_mode2)
        print('Chi f mode 2:', chi_f_mode2)
        print('Chi ef mode 2:', chi_ef_mode2)

        cfg=AttrDict(self.cfg)
        self.sigma_test = self.cfg.expt.length_placeholder

        self.play_pige_pulse()
        self.sync_all()

        self.play_pief_pulse()
        self.sync_all()

        sb_mode1_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][0]
        sb_mode1_sigma = self.sigma_test
        sb_mode1_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][0]
        sb_mode1_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
        sb_mode1_ramp_sigma = self.cfg.expt.ramp_sigma
        sb_mode1_ramp_type = self.cfg.expt.ramp_type
        sb_mode2_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][0]
        sb_mode2_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][0]
        sb_mode2_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][0]
        sb_mode2_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
        sb_mode2_ramp_sigma = self.cfg.expt.ramp_sigma
        sb_mode2_ramp_type = self.cfg.expt.ramp_type

        print('Mode 1 freq:', sb_mode1_freq)
        print('Mode 1 sigma:', sb_mode1_sigma)
        print('Mode 1 gain:', sb_mode1_gain)
        print('Mode 2 freq:', sb_mode2_freq)
        print('Mode 2 sigma:', sb_mode2_sigma)
        print('Mode 2 gain:', sb_mode2_gain)

        # Sideband on mode 1

        self.play_sb(freq=sb_mode1_freq, length=sb_mode1_sigma, gain=sb_mode1_gain, pulse_type=sb_mode1_pulse_type, ramp_type=sb_mode1_ramp_type, ramp_sigma=sb_mode1_ramp_sigma)
        self.sync_all()

        # pi_f0g1 on mode 2

        self.play_sb(freq=sb_mode2_freq, length=sb_mode2_sigma, gain=sb_mode2_gain, pulse_type = sb_mode2_pulse_type, ramp_type=sb_mode2_ramp_type, ramp_sigma=sb_mode2_ramp_sigma)
        self.sync_all()
        
        for i in range(1, self.cfg.expt.n):
            # pi_ge
            self.play_pige_pulse(shift=i*(chi_e_mode1 + chi_e_mode2)/2)
            self.sync_all()

            # pi_ef
            self.play_pief_pulse(shift=i*(chi_ef_mode1 + chi_ef_mode2)/2)
            self.sync_all()

            # pi_fngnp1 on mode 1
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][i]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            # pi_fngnp1 on mode 2
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][i]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type = sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

        if self.cfg.expt.measure_mode1 == True:
            
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            if cfg.expt.add_pi_ef:
                print('Adding pi_ef pulse')
                self.play_pief_pulse(shift=(self.cfg.expt.n-1)*(chi_e_mode1))
                self.sync_all()
        
        else:
            
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][self.cfg.expt.n-1]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][self.cfg.expt.n-1]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][self.cfg.expt.n-1]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][self.cfg.expt.n-1]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            if cfg.expt.add_pi_ef:
                print('Adding pi_ef pulse')
                self.play_pief_pulse(shift=(self.cfg.expt.n-1)*(chi_e_mode2))
                self.sync_all()


        self.sync_all(self.us2cycles(0.05))


        self.measure(pulse_ch=self.res_ch,
                     adcs=[0],
                     pins=[0],
                     adc_trig_offset=self.us2cycles(cfg.device.soc.readout.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels
        
        # System Reset

        if cfg.expt.reset:

            self.cfg.device.soc.readout.reset_cavity_n = cfg.expt.n + 1

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
                        print('chi_ge_cor', chi_e * ii)
                        print('chi_ef_cor', (chi_f - chi_e) * ii)
                        chi_ge_cor = chi_e * ii
                        chi_ef_cor = (chi_f - chi_e) * ii
                    else:
                        chi_ge_cor = 0
                        chi_ef_cor = 0

                    print('Resetting cavity for n =', ii)

                    # Mode 1 Reset

                    # setup and play f,n g,n+1 sideband pi pulse

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][ii]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][ii]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][ii]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode1]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][ii]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
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

                    # Mode 2 Reset
                    
                    # setup and play f,n g,n+1 sideband pi pulse

                    sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][ii]
                    sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][ii]
                    sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][ii]
                    sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode2]
                    sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][ii]
                    sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
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
            
            self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))
        
        
        
class NOONStateSidebandSweepExperiment(Experiment):
    """Length Rabi Experiment
       Experimental Config
       expt_cfg={
       "start": start length, 
       "step": length step, 
       "expts": number of different length experiments, 
       "reps": number of reps,
       "gain": gain to use for the pulse
       "length_placeholder": used for iterating over lengths, initial specified value does not matter
        } 
    """

    def __init__(self, path='', prefix='LengthRabi', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, data_path=None, filename=None, prob_calib=True):
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lenrabi = NOONStateSidebandSweepProgram(soc, self.cfg)
            self.prog=lenrabi
            avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            amp=np.abs(avgi[0][0]+1j*avgq[0][0]) # Calculating the magnitude
            phase=np.angle(avgi[0][0]+1j*avgq[0][0]) # Calculating the phase
            data["xpts"].append(lengths)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data = data

        avgq_col = np.array([data['avgq'][i][0][0] for i in np.arange(len(data['avgq']))])
        avgi_col = np.array([data['avgi'][i][0][0] for i in np.arange(len(data['avgi']))])
        avgi_col_rot, avgq_col_rot = self.iq_rot(avgi_col, avgq_col, theta=self.cfg.device.soc.readout.iq_rot_theta)

        if prob_calib:

            # Calibrate qubit probability

            iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
            i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
            data_dict = {'xpts': data['xpts'][0], 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}
        
        else:

            data_dict = {'xpts': data['xpts'][0], 'avgq':avgq_col, 'avgi':avgi_col}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
        pI = dsfit.fitdecaysin(data['xpts'][0],
                               np.array([data["avgi"][i][0][0] for i in range(len(data['avgi']))]),
                               fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'][0],
                               np.array([data["avgq"][i][0][0] for i in range(len(data['avgq']))]),
                               fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0][0])
        pQ = np.append(pQ, data['xpts'][0][0]) 
        data['fiti'] = pI
        data['fitq'] = pQ
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        print(self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Length Rabi",  ylabel="I")
        plt.plot(data["xpts"][0], [data["avgi"][i][0][0] for i in range(len(data['avgi']))],'o-')
        plt.subplot(212, xlabel="Time (us)", ylabel="Q")
        plt.plot(data["xpts"][0], [data["avgq"][i][0][0] for i in range(len(data['avgq']))],'o-')
        plt.tight_layout()
        plt.show()

