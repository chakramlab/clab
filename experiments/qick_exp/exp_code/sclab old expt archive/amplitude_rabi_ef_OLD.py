import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class AmplitudeRabiEFProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
        
        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_gain = self.sreg(self.qubit_ch, "gain")   # get gain register for qubit_ch
        self.r_gain2 = 4
        
        self.f_res=self.freq2reg(cfg.device.readout.frequency)           # conver f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test)
        self.sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        # print(self.sigma_test)
        # initialize gain
        self.safe_regwi(self.q_rp, self.r_gain2, self.cfg.expt.start)

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="qubit_pi", sigma=self.sigma, length=self.sigma * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_test, length=self.sigma_test * 4)
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)
        
        self.sideband_ch = cfg.device.soc.sideband.ch
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)
        self.add_gauss(ch=self.qubit_ch, name="pi_ef", sigma=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma), length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma) * 4)
           
        self.sync_all(self.us2cycles(0.2))
    
    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=1, phase=0, shift=0):
        
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
        
        # self.mathi(self.s_rp, self.s_freq, self.s_freq2, "+", 0)
        self.pulse(ch=self.sideband_ch)

    def body(self):
        cfg=AttrDict(self.cfg)
        if cfg.expt.pi_qubit:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.qubit.pulses.pi_ge.gain,
                waveform="qubit_pi")
            self.pulse(ch=self.qubit_ch)
            self.sync_all()

        # setup and play ef probe pulse
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(self.cfg.device.qubit.f_ef),
            phase=0,
            gain=0,  # gain set by update
            waveform="qubit_ef")
        self.mathi(self.q_rp, self.r_gain, self.r_gain2, '+', 0)
        self.pulse(ch=self.qubit_ch)
        self.sync_all()

        if cfg.expt.ge_pi_after:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.qubit.pulses.pi_ge.gain,
                waveform="qubit_pi")
            self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=self.us2cycles(cfg.device.soc.readout.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels
    
        # Transmon Reset

        if cfg.expt.reset:
            
            for ii in range(cfg.device.soc.readout.reset_cycles):
                print('Resetting System,', 'Cycle', ii)

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

                if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':

                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="const",
                        freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
                        phase=self.deg2reg(0),
                        gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                        length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
                
                if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':

                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="arb",
                        freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
                        phase=self.deg2reg(0),
                        gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                        waveform="pi_ef")

                self.pulse(ch=self.qubit_ch)
                self.sync_all()

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
            
            self.sync_all(self.us2cycles(cfg.device.soc.readout.relax_delay))

    def update(self):
        self.mathi(self.q_rp, self.r_gain2, self.r_gain2, '+', self.cfg.expt.step) # update frequency list index
                      
                      
class AmplitudeRabiEFExperiment(Experiment):
    """Amplitude Rabi EF Experiment
       Experimental Config
        expt = {"start":0, "step": 150, "expts":200, "reps": 10, "rounds": 200, "sigma_test": 0.025, "pi_qubit": True}
        }
    """

    def __init__(self, path='', prefix='AmplitudeRabiEF', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        amprabi_ef=AmplitudeRabiEFProgram(soc, self.cfg)
        x_pts, avgi, avgq = amprabi_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,
                                               load_pulses=True,progress=progress)
        
        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
        pI = dsfit.fitdecaysin(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0])
        pQ = np.append(pQ, data['xpts'][0]) 
        data['fiti'] = pI
        data['fitq'] = pQ

        print(pI)
        gain_pi = 1 / (2 * pI[1])
        gain_half_pi = 1 / (4 * pI[1])

        print("pi gain:", gain_pi)
        print("half pi gain:", gain_half_pi)
        # ax.axvline(1*gain_pi)
        print("phase of sinusoid:", pI[2])

        print(pQ)
        gain_pi = 1 / (2 * pQ[1])
        gain_half_pi = 1 / (4 * pQ[1])

        print("pi gain:", gain_pi)
        print("half pi gain:", gain_half_pi)
        # ax.axvline(1*gain_pi)
        print("phase of sinusoid:", pQ[2])

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        print (self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Amplitude Rabi EF",  ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Gain", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq"], data["xpts"]))
            
        plt.tight_layout()
        plt.show()