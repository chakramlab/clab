import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict

class ECDCharacteristicFunctionProgram(AveragerProgram):
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
        self.sigma_pi2_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch) 
        self.sigma_ge_resolved = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'

        if self.pulse_type_ge == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_pi2_ge", sigma=self.sigma_pi2_ge, length=self.sigma_pi2_ge * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ge_resolved", sigma=self.sigma_ge_resolved, length=self.sigma_ge_resolved * 4)
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

        # print('Ramp sigma (us):', self.cfg.expt.sb_sigma)

        # Cavity drive parameters
        self.cavdr_ch=cfg.device.soc.storage.ch
        self.cavdr_reg_page =self.ch_page(self.cavdr_ch)     # get register page for cavdr_ch
        self.cavdr_freq_reg = self.sreg(self.cavdr_ch, "freq")   # get frequency register for cavdr_ch
        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        self.declare_gen(ch=self.cavdr_ch, nqz=self.cavdr_ch_nyquist)
        
        self.add_gauss(ch=self.cavdr_ch, name="cavdr", sigma=self.us2cycles(self.cfg.expt.cavity_drive_length), length=self.us2cycles(self.cfg.expt.cavity_drive_length)* 4)

        self.sync_all(self.us2cycles(0.2))
    
    def play_cavity_drive(self, gain=0, phase=0):
                
        if self.cfg.expt.cavity_drive_pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.cavdr_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.storage.freqs[self.cfg.expt.mode]+self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]/2, gen_ch=self.cavdr_ch),
                    phase=self.deg2reg(phase, gen_ch=self.cavdr_ch),
                    gain=gain, 
                    length= self.us2cycles(self.cfg.expt.cavity_drive_length))
            
        if self.cfg.expt.cavity_drive_pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.cavdr_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.storage.freqs[self.cfg.expt.mode]+self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]/2, gen_ch=self.cavdr_ch),
                phase=self.deg2reg(phase, gen_ch=self.cavdr_ch),
                gain=gain,
                waveform="cavdr")
        
        self.pulse(ch=self.cavdr_ch)

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
    
    def play_pi2ge_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.sigma_pi2_ge)
            
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_pi2_ge")
        
        self.pulse(ch=self.qubit_ch)
    
    def play_pige_resolved_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi_ge_resolved.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge)
        
        if self.cfg.device.soc.qubit.pulses.pi_ge_resolved.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge_resolved")
        
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
                # print('Sideband flat top sin squared')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain+gain_shift,
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

        # System Reset

        if cfg.device.soc.readout.reset_cavity_beginning:
            print('Resetting cavity in beginning:')
            print('With ' + str(cfg.device.soc.readout.reset_cavity_beginning_reset_cycles) + ' reset cycles')
            print('For modes:', [ii+1 for ii in cfg.device.soc.readout.reset_cavity_beginning_modes])
            print('Up to n =', cfg.device.soc.readout.reset_cavity_beginning_n)

            for jj in range(cfg.device.soc.readout.reset_cavity_beginning_reset_cycles):
                
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

                # Cavity Reset

                for kk in self.cfg.device.soc.readout.reset_cavity_beginning_modes:
                    # print('Resetting cavity mode', kk+1)

                    for ii in range(self.cfg.device.soc.readout.reset_cavity_beginning_n-1, -1, -1):
                        
                        self.chi_e = self.cfg.device.soc.storage.chi_e[kk]
                        self.chi_f = self.cfg.device.soc.storage.chi_f[kk]
                        self.chi_ef = self.chi_f - self.chi_e
                        chi_ge_cor = self.chi_e * ii
                        chi_ef_cor = self.chi_ef * ii
            

                        # print('Resetting cavity for n =', ii)

                        # setup and play f,n g,n+1 sideband pi pulse

                        sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[kk][ii]
                        sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[kk][ii]
                        sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[kk][ii]
                        sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[kk]
                        sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[kk][ii]
                        sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[kk]
                        # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))
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

            self.sync_all(self.us2cycles(self.cfg.device.soc.readout.reset_cavity_beginning_relax_delay))

        # Put n photons into cavity

        self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        self.chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
        self.chi_ef = self.chi_f - self.chi_e
        
        print('Initializing cavity to |' + str(cfg.expt.n) + '>')
        for i in np.arange(cfg.expt.n):
            
            self.play_pige_pulse(shift = self.chi_e*i)
            self.sync_all()

            self.play_pief_pulse(shift = self.chi_ef*i)
            self.sync_all()

            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][i]
            sb_pulse_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_pulse_types[self.cfg.expt.mode]
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode]
            # print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

        self.play_pi2ge_pulse(phase=-90)
        self.sync_all()

        # ECD Sequence

        print('Cavity drive freq. (MHz):', self.cfg.device.soc.storage.freqs[self.cfg.expt.mode]+self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]/2)
        self.play_cavity_drive(gain = self.cfg.expt.cavity_drive_gains_temp, phase = 0)  # Drive at average of the cavity freq. when in |g> or |e>
        self.sync_all(self.us2cycles(self.cfg.expt.wait_time))

        chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]

        if self.cfg.expt.cavity_drive_pulse_type == 'const':
            pulse_length = self.cfg.expt.cavity_drive_length
            print('Cavity drive: const, pulse length (us):', pulse_length)
        elif self.cfg.expt.cavity_drive_pulse_type == 'gauss':
            pulse_length = self.cfg.expt.cavity_drive_length * 4 
            print('Cavity drive: gauss, pulse length (us):', pulse_length)
        else:
            print('No pulse length specified')

        r_factor = np.cos(np.pi*chi_e*(self.cfg.expt.wait_time))
        return_gain = int(r_factor * self.cfg.expt.cavity_drive_gains_temp)
        print('r_factor (pulse 2):', r_factor)
        print('Return gain (pulse 2):', return_gain)

        self.play_cavity_drive(gain = return_gain, phase = 180)
        self.sync_all()

        self.play_pige_pulse()  # Flip qubit to echo spurious terms
        self.sync_all()

        r_factor = np.cos(np.pi*chi_e*(self.cfg.expt.wait_time))
        return_gain = int(r_factor * self.cfg.expt.cavity_drive_gains_temp)
        print('r_factor (pulse 3):', r_factor)
        print('Return gain (pulse 3):', return_gain)

        self.play_cavity_drive(gain = return_gain, phase = 180)
        self.sync_all(self.us2cycles(self.cfg.expt.wait_time))

        r_factor = np.cos(np.pi*chi_e*(2*self.cfg.expt.wait_time))
        return_gain = int(r_factor * self.cfg.expt.cavity_drive_gains_temp)
        print('r_factor (pulse 4):', r_factor)
        print('Return gain (pulse 4):', return_gain)

        self.play_cavity_drive(gain = return_gain, phase = 0)
        self.sync_all()
        
        # Qubit tomography 

        self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        self.chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
        self.chi_ef = self.chi_f - self.chi_e
        
        if self.cfg.expt.shift_qubit:
            gain = self.cfg.expt.cavity_drive_gains_temp
            n = (gain * self.cfg.expt.beta_gain_factor / 2)**2
        else:
            n=0

        print('Photon n:', n)

        if self.cfg.expt.tomography_pulse_type_temp == 'pi2_x':

            print('Playing pi2_x pulse')

            self.play_pi2ge_pulse(phase=0, shift=n*self.chi_e)
            self.sync_all()  
    
        elif self.cfg.expt.tomography_pulse_type_temp == 'pi2_y':
            
            print('Playing pi2_y pulse')

            self.play_pi2ge_pulse(phase=90, shift=n*self.chi_e)
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
        
        self.sync_all()

        # System Reset

        if cfg.expt.reset:

            self.chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
            self.chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]
            self.chi_ef = self.chi_f - self.chi_e

            self.cfg.device.soc.readout.reset_cavity_n = cfg.expt.reset_n+1
            
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
                        # print('chi_ge_cor', self.chi_e * ii)
                        # print('chi_ef_cor', (self.chi_f - self.chi_e) * ii)
                        chi_ge_cor = self.chi_e * ii
                        chi_ef_cor = (self.chi_f - self.chi_e) * ii
                    else:
                        chi_ge_cor = 0
                        chi_ef_cor = 0

                    # print('Resetting cavity for n =', ii)
                    
                    # Primary Mode 

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

        
        
        
class ECDCharacteristicFunctionExperiment(Experiment):
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

        avgi_col = []
        avgq_col = []
        avgi_pi2_x_col = []
        avgq_pi2_x_col = []
        avgi_pi2_y_col = []
        avgq_pi2_y_col = []

        if self.cfg.expt.sigma_z:
            for ii in tqdm(self.cfg.expt.cavity_drive_gains, disable=not progress):
                
                self.cfg.expt.cavity_drive_gains_temp = ii 
                self.cfg.expt.tomography_pulse_type_temp = 'I' 
                soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
                lenrabi = ECDCharacteristicFunctionProgram(soc, self.cfg)       
                avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi_col.append(avgi[0][0])
                avgq_col.append(avgq[0][0])
        else:
            print('Not measuring sigma_z.')

        for ii in tqdm(self.cfg.expt.cavity_drive_gains, disable=not progress):

            self.cfg.expt.cavity_drive_gains_temp = ii 
            self.cfg.expt.tomography_pulse_type_temp = 'pi2_x'  
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            lenrabi = ECDCharacteristicFunctionProgram(soc, self.cfg)      
            avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi_pi2_x_col.append(avgi[0][0])
            avgq_pi2_x_col.append(avgq[0][0])
        
        for ii in tqdm(self.cfg.expt.cavity_drive_gains, disable=not progress):
            
            self.cfg.expt.cavity_drive_gains_temp = ii 
            self.cfg.expt.tomography_pulse_type_temp = 'pi2_y' 
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            lenrabi = ECDCharacteristicFunctionProgram(soc, self.cfg)       
            avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi_pi2_y_col.append(avgi[0][0])
            avgq_pi2_y_col.append(avgq[0][0])

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)
        avgi_pi2_x_col = np.array(avgi_pi2_x_col)
        avgq_pi2_x_col = np.array(avgq_pi2_x_col)
        avgi_pi2_y_col = np.array(avgi_pi2_y_col)
        avgq_pi2_y_col = np.array(avgq_pi2_y_col)

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
        i_pi2_x_prob, q_pi2_x_prob = self.get_qubit_prob(avgi_pi2_x_col, avgq_pi2_x_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
        i_pi2_y_prob, q_pi2_y_prob = self.get_qubit_prob(avgi_pi2_y_col, avgq_pi2_y_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'cavity_drive_gains': self.cfg.expt.cavity_drive_gains, 
                     'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']],
                     'avgq':avgq_col, 'avgi':avgi_col, 'avgi_prob': i_prob, 'avgq_prob': q_prob,
                     'avgq_pi2_x':avgq_pi2_x_col, 'avgi_pi2_x':avgi_pi2_x_col, 'avgi_pi2_x_prob': i_pi2_x_prob, 'avgq_pi2_x_prob': q_pi2_x_prob,
                     'avgq_pi2_y':avgq_pi2_y_col, 'avgi_pi2_y':avgi_pi2_y_col, 'avgi_pi2_y_prob': i_pi2_y_prob, 'avgq_pi2_y_prob': q_pi2_y_prob}
                     
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict, create_dataset=True)

        return data_dict

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


