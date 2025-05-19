import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict

class fngnp1BangBangProgram(AveragerProgram):
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
        


        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)


        for ch in [0]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)
        
        # qubit ge and ef pulse parameters

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'

        if self.pulse_type_ge == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        
        if self.pulse_type_ef == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)


        # if self.cfg.expt.fngnp1_probepulse_type == 'flat_top':
        #     self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_probe", sigma=self.us2cycles(self.cfg.expt.sb_sigma), length=self.us2cycles(self.cfg.expt.sb_sigma) * 4)
        
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        
        # ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][0]
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(self.cfg.expt.sb_sigma), length=self.us2cycles(self.cfg.expt.sb_sigma) * 4)
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(self.cfg.expt.sb_sigma) * 2)
        self.add_bump_func(ch=self.sideband_ch, name="sb_flat_top_bump", length=self.us2cycles(self.cfg.expt.sb_sigma) * 2, k=2, flat_top_fraction=0.0)

        print('Ramp sigma (us):', self.cfg.expt.sb_sigma)
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

    # def play_pi_sb(self, n = 0, mode=0, phase=0, shift=0, gain_shift=0):
        
    #     for ii in range(self.cfg.expt.n+1):
    #         self.add_gauss(
    #             ch=self.sideband_ch, 
    #             name="sb_flat_top_gauss_pi"+str(ii), 
    #             sigma=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[mode][n]), 
    #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[mode][n]) * 4)
    #     for ii in range(self.cfg.expt.n+1):
    #         self.add_cosine(
    #             ch=self.sideband_ch, 
    #             name="sb_flat_top_cos_pi"+str(ii), 
    #             sigma=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[mode][n]), 
    #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[mode][n]) * 4)

    #     if self.cfg.expt.fngnp1_pipulse_type == 'const':
            
    #         # print('Sideband const')
    #         self.set_pulse_registers(
    #             ch=self.sideband_ch,
    #             style="const",
    #             freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][n] + shift),  # freq set by update
    #             phase=self.deg2reg(phase),
    #             gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][n]+gain_shift,
    #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][n]))
            
    #     if self.cfg.expt.fngnp1_pipulse_type == 'flat_top':
            
    #         print('Sideband flat top')
    #         self.set_pulse_registers(
    #             ch=self.sideband_ch,
    #             style="flat_top",
    #             freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][n] +shift),
    #             phase=self.deg2reg(phase),
    #             gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][n]+gain_shift,
    #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][n]),
    #             waveform="sb_flat_top_cos_pi"+str(n))
        
    #     print('length', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][n])
    #     print('gain', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][n])
    #     print('freq', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][n])
    #     self.pulse(ch=self.sideband_ch)

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=0.01, phase=0, shift=0, gain_shift=0):

        if pulse_type == 'const':
            
            # print('Sideband const')
            self.set_pulse_registers(
                    ch=self.sideband_ch, 
                    style="const", 
                    freq=self.freq2reg(freq+shift), 
                    phase=self.deg2reg(phase),
                    gain=gain+gain_shift, 
                    length=self.us2cycles(length))
        
        elif pulse_type == 'flat_top':
            
            if ramp_type == 'sin_squared':
                print('Sideband flat top sin squared')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain+gain_shift,
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
                # print('Sideband flat top gaussian')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain+gain_shift,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_gaussian")
        
        self.pulse(ch=self.sideband_ch)


    def body(self):

        cfg=AttrDict(self.cfg)

        # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]

        # Put n photons into cavity 

        for i in np.arange(cfg.expt.n):
            
            if self.cfg.expt.chi_correction:
                chi_ge_cor = chi_e * i
                chi_ef_cor = (chi_f - chi_e) * i
            else:
                chi_ge_cor = 0
                chi_ef_cor = 0

            # pi_ge
            self.play_pige_pulse(phase=0, shift=chi_ge_cor)
            self.sync_all()

            # pi_ef

            self.play_pief_pulse(phase=0, shift=chi_ef_cor)
            self.sync_all()

            # pi_fngnp1 

            freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][i]
            length = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][i]
            gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][i]
            pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
            ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
            ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][i]
            print('Freq: ', freq, 'Length: ', length, 'Gain: ', gain, 'Pulse type: ', pulse_type, 'Ramp type: ', ramp_type, 'Ramp sigma: ', ramp_sigma)
            self.play_sb(
                freq=freq,
                length=length,
                gain=gain,
                pulse_type=pulse_type,
                ramp_type=ramp_type,
                ramp_sigma=ramp_sigma)
            self.sync_all()

        if self.cfg.expt.chi_correction:
            chi_ge_cor = chi_e * self.cfg.expt.n
            chi_ef_cor = (chi_f - chi_e) * self.cfg.expt.n
        else:
            chi_ge_cor = 0
            chi_ef_cor = 0

        # pi_ge
            
        self.play_pige_pulse(phase=0, shift=chi_ge_cor) 
        self.sync_all()

        # pi_ef 

        self.play_pief_pulse(phase=0, shift=chi_ef_cor)
        self.sync_all()

        # repeatedly play fngnp1 pulse

        

        if self.cfg.expt.relative_phase_bool:
            # print('Relative phase')
            # counter = 0
            # phase = 0
            # for ii in np.arange(self.cfg.expt.n_placeholder):
            #     if counter%(2*self.cfg.expt.relative_phase_num) < self.cfg.expt.relative_phase_num:
            #         self.set_pulse_registers(
            #             ch=self.sideband_ch,
            #             style="flat_top",
            #             freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.freq_offset_placeholder),
            #             phase=self.deg2reg(0),
            #             gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.gain_offset_placeholder,
            #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]),
            #             waveform="sb_flat_top_sin_squared")
            
            #         self.pulse(ch=self.sideband_ch)
            #         self.sync_all()
            #     else:
            #         self.set_pulse_registers(
            #             ch=self.sideband_ch,
            #             style="flat_top",
            #             freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.freq_offset_placeholder),
            #             phase=self.deg2reg(self.cfg.expt.relative_phase),
            #             gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.gain_offset_placeholder,
            #             length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]),
            #             waveform="sb_flat_top_sin_squared")
            #         self.pulse(ch=self.sideband_ch)
            #         self.sync_all()
            #     counter +=1
            print('Relative phase; advance')
            phase = 0
            for ii in np.arange(self.cfg.expt.n_placeholder):
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.freq_offset_placeholder),
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.gain_offset_placeholder,
                    length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]),
                    waveform="sb_flat_top_bump")
        
                self.pulse(ch=self.sideband_ch)
                self.sync_all()
                phase += self.cfg.expt.relative_phase


        else:
            print('No relative phase')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="flat_top",
                freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.freq_offset_placeholder),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n] + self.cfg.expt.gain_offset_placeholder,
                length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]),
                waveform="sb_flat_top_bump")

            print('Sideband pi pulse initiated:', 'Freq:', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n], 
                'Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n], 
                'Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n],
                'Freq. offset:', self.cfg.expt.freq_offset_placeholder, 'Gain offset:', self.cfg.expt.gain_offset_placeholder)
            print('waiting 0 ns between pulses')
            for ii in np.arange(self.cfg.expt.n_placeholder):

                self.pulse(ch=self.sideband_ch)
                self.sync_all()
                # self.sync_all(self.us2cycles(0.0023))



        if self.cfg.expt.add_pi_ef:
            print('Adding pi_ef pulse')
            self.play_pief_pulse(phase=0, shift=chi_ef_cor)
            self.sync_all()
        
        # Get rid of |f> (was in |e> before the pi_ef above) population 

        # SWAP to readout mode

        # pi_f0g1 to readout mode

        if self.cfg.expt.reset_f: 

            sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
            # sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pi_lengths[0]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
            
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

        # Move to |g01> of a different mode

        # freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.swap_mode][0]
        # length = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.swap_mode][0]
        # gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.swap_mode][0]
        # pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.swap_mode]
        # ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.swap_mode]
        # ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.swap_mode][0]
        # print('Freq: ', freq, 'Length: ', length, 'Gain: ', gain, 'Pulse type: ', pulse_type, 'Ramp type: ', ramp_type, 'Ramp sigma: ', ramp_sigma)
        # self.play_sb(
        #     freq=freq, 
        #     length=length, 
        #     gain=gain, 
        #     pulse_type=pulse_type, 
        #     ramp_type=ramp_type, 
        #     ramp_sigma=ramp_sigma,
        #     shift=self.cfg.expt.freq_offset_placeholder,
        #     gain_shift= self.cfg.expt.gain_offset_placeholder)
        # self.sync_all()


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
        
        self.sync_all()

        # System Reset

        if cfg.expt.reset:

            self.cfg.device.soc.readout.reset_cavity_n = cfg.expt.n+1
            
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
                    
                    # Primary Mode 

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
                
                    # # SWAP Mode 

                    # # setup and play f,n g,n+1 sideband pi pulse

                    # sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.swap_mode][ii]
                    # sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.swap_mode][ii]
                    # sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.swap_mode][ii]
                    # sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.swap_mode]
                    # sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.swap_mode][ii]
                    # sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.swap_mode]
                    # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    # self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
                    # self.sync_all()

                    # # Transmon Reset

                    # # f0g1 to readout mode

                    # sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                    # sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                    # sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                    # sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                    # sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                    # sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                    # # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    
                    # self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                    # self.sync_all()

                    # # pi_ef

                    # self.play_pief_pulse(shift=chi_ef_cor)
                    # self.sync_all()

                    # # f0g1 to readout mode

                    # sb_freq = self.cfg.device.soc.sideband.fngnp1_readout_freqs[0]
                    # sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_reset_lengths[0]
                    # sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1_readout_gains[0]
                    # sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_pulse_types[0]
                    # sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_types[0]
                    # sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1_readout_ramp_sigmas[0]
                    # # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                    
                    # self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                    # self.sync_all()

            self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

        
        
        
class fngnp1BangBangExperiment(Experiment):
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

    def acquire(self, progress=False, data_path=None, filename=None):
        n_pts = self.cfg.expt.n_pts
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        avgi_col = []  # 2D list with I data of element [i, j] corresponding to [dataset with freq_offset and gain_offset, element in n_pts], 
                       # needs to be reshaped outside into a 3D array where the first two dimensions are [i, j]=[freq_offset, gain_offset]
        avgq_col = []
        i_prob_col = []
        q_prob_col = []

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)

        for freq_offset in tqdm(self.cfg.expt.freq_offsets, total=len(self.cfg.expt.freq_offsets)):
            for gain_offset in self.cfg.expt.gain_offsets:

                avgi_col_temp = []
                avgq_col_temp = []
                for ii in tqdm(n_pts, disable=not progress):
                    self.cfg.expt.n_placeholder = int(ii)  # Required because h5py can't save int32 numpy values for the config
                    self.cfg.expt.freq_offset_placeholder = freq_offset
                    self.cfg.expt.gain_offset_placeholder = gain_offset
                    print('number of sideband pulses:', ii)
                
                    soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())

                    lenrabi = fngnp1BangBangProgram(soc, self.cfg)
                                        
                    avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                    avgi_col_temp.append(avgi[0][0])
                    avgq_col_temp.append(avgq[0][0])

                    

                avgi_col.append(avgi_col_temp)
                avgq_col.append(avgq_col_temp)
                avgi_col_temp = np.array(avgi_col_temp)
                avgq_col_temp = np.array(avgq_col_temp)
                i_prob, q_prob = self.get_qubit_prob(avgi_col_temp, avgq_col_temp, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
                i_prob_col.append(i_prob)
                q_prob_col.append(q_prob)
        
        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)
        i_prob = np.array(i_prob_col)
        q_prob = np.array(q_prob_col)

        data_dict = {'n_pts': n_pts, 'freq_offsets': self.cfg.expt.freq_offsets, 'gain_offsets': self.cfg.expt.gain_offsets, 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict, create_dataset=True)

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


