import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class RamseyEchoCPMGProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch

        self.q_rp = self.ch_page(self.qubit_ch)  # get register page for qubit_ch
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase = self.sreg(self.qubit_ch, "phase")
        self.safe_regwi(self.q_rp, self.r_wait, cfg.expt.start)
        self.safe_regwi(self.q_rp, self.r_phase2, 0)

        self.f_res = self.freq2reg(cfg.device.readout.frequency)  # convert f_res to dac register value
        self.readout_length = self.us2cycles(cfg.device.readout.readout_length)
        # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
        # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

        self.sigma = self.us2cycles(cfg.device.qubit.pulses.pi2_ge.sigma)
        self.pisigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # add qubit and readout pulses to respective channels
            
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))
    
    def play_pige(self, phase = 0, shift = 0):

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
                waveform="qubit_ge2")
        
        self.pulse(ch=self.qubit_ch)

    def body(self):
        cfg = AttrDict(self.cfg)

        self.play_piby2ge()
        self.sync_all()

        for ii in np.arange(cfg.expt.n_spin_echoes):
            tau = self.us2cycles(cfg.expt.tau_placeholder/cfg.expt.n_spin_echoes/2)
            self.sync(self.q_rp, tau)
            self.play_pige(phase=90)
            self.sync(self.q_rp, tau)

        self.play_piby2ge(phase=cfg.expt.phase_placeholder)
        self.sync_all()

        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.readout.trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels


class RamseyEchoCPMGExperiment(Experiment):
    """Ramsey Echo Experiment
       Experimental Config
        expt = {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200, "phase_step": deg2reg(360/50), "echo_times": 1,
        "cp: False, "cpmg": True}
         }
    """

    def __init__(self, path='', prefix='RamseyEcho', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, data_path=None, filename=None):
        
        x_pts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        avgi_col = []
        avgq_col = []
        for i in tqdm(np.arange(len(x_pts)), disable = not progress):
            self.cfg.expt.tau_placeholder = x_pts[i]
            self.cfg.expt.phase_placeholder = i*self.cfg.expt.phase_step
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            ramsey_echo = RamseyEchoCPMGProgram(soc, self.cfg)
            avgi, avgq = ramsey_echo.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                                    progress=progress, debug=debug)
            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])  

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col, 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)

        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        pI = dsfit.fitdecaysin(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)

        data['fiti'] = pI
        data['fitq'] = pQ

        data['T2echo'] = (1 + self.cfg.expt.echo_times) * pI[3]
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print(self.fname)
            # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Ramsey Echo", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Delay between pulses (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq"], data["xpts"]))

        plt.tight_layout()
        plt.show()
