import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PNRQSSidebandOptimalControlDynamicalPhaseHardwareProgram(AveragerProgram):
    def initialize(self):

        # --- Config ---

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
        
        self.qubit_gf_freq = self.freq2reg(cfg.device.soc.qubit.f_gf, gen_ch = self.sideband_ch)
        self.qubit_gf_gain = cfg.device.soc.qubit.pulses.pi_gf.gain
        self.qubit_resolved_ch = cfg.device.soc.qubit.pulses.pi_ge_resolved.ch

        # --- Initialize pulses

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.q_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.qubit_resolved_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

        self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        self.chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
        self.chi_ef = self.chi_f - self.chi_e

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge_resolved = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
        self.add_gauss(ch=self.qubit_resolved_ch, name="qubit_ge_resolved", sigma=self.sigma_ge_resolved, length=self.sigma_ge_resolved * 4)
        
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(self.cfg.expt.sb_ramp_sigma), length=self.us2cycles(self.cfg.expt.sb_ramp_sigma * 4))
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2)
        self.add_bump_func(ch=self.sideband_ch, name="sb_flat_top_bump", length=self.us2cycles(self.cfg.expt.sb_ramp_sigma) * 2, k=2, flat_top_fraction=0.0)
        self.add_bump_func_freq_modulation(
            ch=self.sideband_ch, name='sb_flat_top_bump_freq_mod', 
            ramp_length=self.us2cycles(self.cfg.expt.sb_ramp_sigma), 
            flat_top_length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1twopi_times[self.cfg.expt.mode][self.cfg.expt.n]), 
            k=2, 
            freq = self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][self.cfg.expt.n])
        
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
    
    def play_ge_pulse(self, gain=0, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=gain, 
                    length=self.sigma_ge)
            
        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':

            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=gain,
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
                waveform="qubit_ge2")
        
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

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='bump', ramp_sigma=0.01, phase=0, shift=0, stark_shift_idle_correction=False):

        if stark_shift_idle_correction: 
            
            if pulse_type == 'flat_top':

                if ramp_type == 'bump':
                    print('Sideband flat top bump with freq. modulation')
                    print('Freq. modulation (MHz):', self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][self.cfg.expt.n])
                    self.set_pulse_registers(
                        ch=self.sideband_ch,
                        style="arb",
                        freq=self.freq2reg(freq + shift - self.cfg.device.soc.sideband.fngnp1_stark_shifts[self.cfg.expt.mode][self.cfg.expt.n]),
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
                    # print('Sideband flat top bump')
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
                # print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)
        
        # Put n photons into cavity 

        chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
        chi_ef = chi_f - chi_e
        
        for i in np.arange(self.cfg.expt.prep_n):

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
            print('Loading photon: playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type,ramp_sigma=sb_ramp_sigma)
            self.sync_all()

        qubit_thetas = cfg.expt.qubit_thetas
        qubit_phis = cfg.expt.qubit_phis
        sb_phis = cfg.expt.sb_phis
        dynamical_phase = self.cfg.expt.dynamical_phase_correction
        print('Dynamical phase hardware correction (degree):', dynamical_phase)

        for ii in range(len(qubit_thetas)):

            self.play_pief_pulse()
            self.sync_all()

            pi_gain = self.cfg.device.soc.qubit.pulses.pi_ge.gain
            theta_gain = int(qubit_thetas[ii] / 180 * pi_gain)
            print('theta_gain:', theta_gain)
            
            self.play_ge_pulse(gain=theta_gain, phase=qubit_phis[ii] + dynamical_phase * ii)
            self.sync_all()

            self.play_pief_pulse()
            self.sync_all()
        
            self.play_sb(
                freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n],
                gain= self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n], 
                pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode],
                ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode],
                ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][self.cfg.expt.n],
                phase = sb_phis[ii] + dynamical_phase * ii,    
                stark_shift_idle_correction = True)
            self.sync_all()

            # Length is initialized by the custom freq. modulated pulse

            print('Qubit theta (degree):', qubit_thetas[ii])
            print('Qubit phi (degree):', qubit_phis[ii])
            print('Sideband phi (degree):', sb_phis[ii])
            print('Sideband length (us):', self.cfg.device.soc.sideband.pulses.fngnp1twopi_times[self.cfg.expt.mode][self.cfg.expt.n])
            print('Sideband gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][self.cfg.expt.n])
            print('Sideband frequency (MHz):', self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][self.cfg.expt.n])
            print('Sideband ramp sigma (us):', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][self.cfg.expt.n])
            print('Sideband ramp type:', self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode])
        
        # Reset |e>
        print('Resetting |e> state')
        if self.cfg.expt.reset_f:

            # self.play_pief_pulse()
            # self.sync_all()

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
    
        # PNRQS 
        
        channel = self.qubit_resolved_ch
        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':

            self.set_pulse_registers(
                ch=channel,
                style="arb",
                freq=self.freq2reg(cfg.expt.freq_temp),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge_resolved")
        
        if self.qubit_pulsetype == 'const':
            self.set_pulse_registers(
                    ch=channel, 
                    style="const", 
                    freq=self.freq2reg(cfg.expt.freq_temp),
                    phase=self.deg2reg(0),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge_resolved)
            
        self.pulse(ch=channel)

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
        chi_ef = chi_f - chi_e

        if cfg.expt.reset:

            self.cfg.device.soc.readout.reset_cavity_n = cfg.expt.n + 1

            for jj in range(cfg.device.soc.readout.reset_cycles):

                # print('Reset cycle:', jj)

                for kk in range(cfg.device.soc.readout.reset_cavity_n, 0, -1):
                    
                    # print('Resetting cavity for N =', kk)

                    # Cavity Reset for |gN>

                    for ii in range(kk-1, -1, -1):

                        # print('Pulse for gn=', ii+1)

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

                    self.play_pief_pulse(shift = (kk)*chi_ef)
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
                        # print('Pulse for en=', ii+1)

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
                        # print('chi_f', chi_f)
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma))
                        
                        self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
                        self.sync_all()
                
                self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))
    
class PNRQSSidebandOptimalControlDynamicalPhaseHardwareExperiment(Experiment):
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

        xpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        for i in tqdm(xpts, disable = not progress):
            self.cfg.expt.freq_temp = i
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            qspec=PNRQSSidebandOptimalControlDynamicalPhaseHardwareProgram(soc, self.cfg)
            avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False) 
            
            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])      
        
        # Calibrate qubit probability

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts':xpts, 'avgi':avgi_col, 'avgq':avgq_col, 'avgi_prob': i_prob, 'avgq_prob': q_prob}

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
        