import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class f0g1SidebandRamseyPhaseSweepProgram(AveragerProgram):
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

        
        self.f_res=self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.soc.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.pisigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        # print(self.sigma)


        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)


        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

        # qubit ge and ef pulse parameters

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ge2 = cfg.device.soc.qubit.pulses.pi2_ge.pulse_type
        except: self.pulse_type_ge2 = 'const'

        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'
        try: self.pulse_type_ef2 = cfg.device.soc.qubit.pulses.pi2_ef.pulse_type
        except: self.pulse_type_ef2 = 'const'

        print('Pulse type_ge: ' + self.pulse_type_ge)
        print('Pulse type_ef: ' + self.pulse_type_ef)
        print('Pulse type_ge2: ' + self.pulse_type_ge2)
        print('Pulse type_ef2: ' + self.pulse_type_ef2)

        if self.pulse_type_ge == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)

        if self.pulse_type_ge2 == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        
        if self.pulse_type_ef == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        if self.pulse_type_ef2 == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef2", sigma=self.sigma_ef2, length=self.sigma_ef2 * 4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(self.cfg.expt.ramp_sigma) * 2)
        self.add_bump_func(ch=self.sideband_ch, name="sb_flat_top_bump", length=self.us2cycles(self.cfg.expt.ramp_sigma) * 2, k=2, flat_top_fraction=0.0)
        self.add_bump_func_freq_modulation(ch=self.sideband_ch, name='sb_flat_top_bump_freq_mod', ramp_length=self.us2cycles(self.cfg.expt.ramp_sigma), 
                                           flat_top_length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]), k=2, 
                                           freq = self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][self.cfg.expt.n])

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

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=1, phase=0, shift=0, stark_shift_idle_correction=False):
        
        if stark_shift_idle_correction: 
            
            if pulse_type == 'flat_top':

                if ramp_type == 'bump':
                    print('Sideband flat top bump with freq. modulation')
                    print('Freq. modulation (MHz):', self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][0])
                    self.set_pulse_registers(
                        ch=self.sideband_ch,
                        style="arb",
                        freq=self.freq2reg(freq + shift - self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][0]),
                        phase=self.deg2reg(phase),
                        gain=gain,
                        waveform="sb_flat_top_bump_freq_mod")
                
        else: 
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
        
        # self.mathi(self.s_rp, self.s_freq, self.s_freq2, "+", 0)
        self.pulse(ch=self.sideband_ch)

    def body(self):
        cfg=AttrDict(self.cfg)

        # Phase reset

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        # Put n photons into cavity 

        chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
        chi_ef = chi_f - chi_e

        for i in np.arange(cfg.expt.n):

            # setup and play qubit ge pi pulse

            self.play_pige_pulse(shift=chi_e * i)
            self.sync_all()

            # setup and play qubit ef pi pulse

            self.play_pief_pulse(shift=chi_e * i)
            self.sync_all()

            # setup and play f,n g,n+1 sideband pi pulse


            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][i]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
            # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
            self.sync_all()


        # setup and play pi/2 ge qubit pulse

        if self.pulse_type_ge2 == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(cfg.device.soc.qubit.f_ge + chi_e * cfg.expt.n), 
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.sigma_ge2)
            
        if self.pulse_type_ge2 == 'gauss':

            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge + chi_e * cfg.expt.n),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_ge2")
        
        # play ge pi/2 pulse
        self.pulse(ch=self.qubit_ch)

        self.sync_all()

        # setup and play pi ef qubit pulse

        if self.pulse_type_ef == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(cfg.device.soc.qubit.f_ef + chi_ef * cfg.expt.n), 
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                    length=self.sigma_ef)
            
        if self.pulse_type_ef == 'gauss':

            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ef + chi_ef * cfg.expt.n),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        # play ef pi pulse
        self.pulse(ch=self.qubit_ch)

        self.sync_all()

        # setup and play f0g1 sideband pi pulse

        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n]
        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]
        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n]
        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][self.cfg.expt.n]
        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
        print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))

        if self.cfg.device.soc.sideband.drive_frame_stark_shift_correction:
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma, stark_shift_idle_correction=True, phase=self.cfg.expt.phase_temp)
        else:
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma, phase=self.cfg.expt.phase_temp)
        
        self.sync_all()

        # setup and play f0g1 sideband pi pulse

        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n]
        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][self.cfg.expt.n]
        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n]
        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][self.cfg.expt.n]
        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
        print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))

        if self.cfg.device.soc.sideband.drive_frame_stark_shift_correction:
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma, stark_shift_idle_correction=True, phase=-self.cfg.expt.phase_temp)
        else:
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma, phase=-self.cfg.expt.phase_temp)
        self.sync_all()

        # Setup ef pi pulse

        if self.pulse_type_ef == 'const':

            self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(cfg.device.soc.qubit.f_ef + chi_ef * cfg.expt.n), 
                phase=0,
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                length=self.sigma_ef)
        
        if self.pulse_type_ef == 'gauss':

            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ef + chi_ef * cfg.expt.n),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        # Play ef pi pulse

        self.pulse(ch=self.qubit_ch)
        self.sync_all()

        # setup and play pi/2 ge qubit pulse

        if self.pulse_type_ge2 == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(cfg.device.soc.qubit.f_ge + chi_e * cfg.expt.n), 
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.sigma_ge2)
            
        if self.pulse_type_ge2 == 'gauss':

            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge + chi_e * cfg.expt.n),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_ge2")
        
        # play ge pi/2 pulse
        self.pulse(ch=self.qubit_ch)

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
                      
        # System Reset

        if cfg.expt.reset:

            self.cfg.device.soc.readout.reset_cavity_n = self.cfg.expt.n + 1

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
            
            self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

class f0g1SidebandRamseyPhaseSweepExperiment(Experiment):
    """T1 Experiment
       Experimental Config
        expt =  {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200}
    """

    def __init__(self, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)
    def acquire(self, progress=False, debug=False, data_path=None, filename=None):

        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        
        avgi_col = []
        avgq_col = []

        for phase in tqdm(xpts, disable = not progress):

            self.cfg.expt.phase_temp = phase

            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            t1 = f0g1SidebandRamseyPhaseSweepProgram(soc, self.cfg)
            avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0]) 
                
        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(np.array(avgi_col), np.array(avgq_col), iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts': xpts, 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path, filename, arrays=data_dict)

        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        pI = dsfit.fitexp(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitexp(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0])
        pQ = np.append(pQ, data['xpts'][0]) 
        data['fiti'] = pI
        data['fitq'] = pQ
        print("T1:", data['fiti'][3], data['fitq'][3])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data
        print(self.fname)
        # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="T1",  ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Wait Time (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fitq"], data["xpts"]))
            
        plt.tight_layout()
        plt.show()
