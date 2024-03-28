import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class RamseyFHProgram(RAveragerProgram):

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch
        
        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_wait = 3# self.sreg(self.qubit_ch, "time")
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))

        self.r_phase2 = 4
        self.r_phase = self.sreg(self.qubit_ch, "phase")
        #self.r_phase = 0
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rp, self.r_phase2, 0)
        
        self.f_res=self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.soc.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.pisigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        self.pigain = cfg.device.soc.qubit.pulses.pi_ge.gain
        self.piby2sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma)
        self.piby2gain = cfg.device.soc.qubit.pulses.pi2_ge.gain

        self.pisigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.pigain_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.gain, gen_ch=self.qubit_ch)
        self.pi2sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ef.sigma)
        self.pi2gain_ef = cfg.device.soc.qubit.pulses.pi2_ef.gain
        
        self.pi2sigma_fh = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_fh.sigma)
        self.pi2gain_fh = cfg.device.soc.qubit.pulses.pi2_fh.gain
        
        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)

        # add qubit and readout pulses to respective channels
        try: pulse_type = cfg.device.soc.qubit.pulses.pi2_ge.pulse_type
        except: pulse_type = 'const'

        print ("pulse type = ",pulse_type)
            
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        
        self.sync_all(self.us2cycles(0.2))



    def body(self):
        cfg = AttrDict(self.cfg)

        # Setup and play pi_ge qubit pulse

        self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge), 
                phase=0,
                gain=self.pigain, 
                length=self.pisigma)
        
        self.pulse(ch=self.qubit_ch) 
        self.sync_all()

        # Setup and play pi_ef qubit pulse

        self.set_pulse_registers(
            ch=self.qubit_ch, 
            style="const", 
            freq=self.freq2reg(cfg.device.soc.qubit.f_ef), 
            phase=0,
            gain=self.pigain_ef, 
            length=self.pisigma_ef)
        
        self.pulse(ch=self.qubit_ch) 
        self.sync_all()

        # Setup and play pi/2_fh qubit pulse

        self.set_pulse_registers(
            ch=self.qubit_ch, 
            style="const", 
            freq=self.freq2reg(cfg.device.soc.qubit.f_fh), 
            phase=0,
            gain=self.pi2gain_fh, 
            length=self.pi2sigma_fh)
        
        self.pulse(ch=self.qubit_ch) 
        self.sync_all()

        # Wait time and advance phase of qubit pulse

        self.sync(self.q_rp, self.r_wait)
        # self.safe_regwi(self.q_rp, self.r_phase, 0)
        self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)

        # Play pi_fh/2 qubit pulse

        self.pulse(ch=self.qubit_ch)
        self.sync_all()

        # Setup and play pi_ef qubit pulse

        self.set_pulse_registers(
            ch=self.qubit_ch, 
            style="const", 
            freq=self.freq2reg(cfg.device.soc.qubit.f_ef), 
            phase=0,
            gain=self.pigain_ef, 
            length=self.pisigma_ef)
        
        self.pulse(ch=self.qubit_ch) 
        self.sync_all()

        # Setup and play pi_ge qubit pulse

        self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge), 
                phase=0,
                gain=self.pigain, 
                length=self.pisigma)
        
        self.pulse(ch=self.qubit_ch) 
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # Measure
        
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg.expt.step))  # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+',
                   self.deg2reg(self.cfg.expt.phase_step, gen_ch=self.qubit_ch))  # advance the phase of the LO for the second π/2 pulse


class RamseyFHExperiment(Experiment):
    """Ramsey Experiment
       Experimental Config
        expt = {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200, "phase_step": deg2reg(360/50)}
         }
    """

    def __init__(self, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        ramsey = RamseyFHProgram(soc, self.cfg)
        print(self.im[self.cfg.aliases.soc], 'test0')
        xpts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)
        

        avgi_rot, avgq_rot = self.iq_rot(avgi[0][0], avgq[0][0], self.cfg.device.soc.readout.iq_rot_theta)
        data_dict = {'xpts':xpts, 'avgi':avgi[0][0], 'avgq':avgq[0][0], 'avgi_rot':avgi_rot, 'avgq_rot':avgq_rot}

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
        corr_freq = (self.cfg.device.qubit.f_ge) - data['fiti'][1]
        data['corr_freq'] = corr_freq

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print(self.fname)

            # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Ramsey", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Delay (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq"], data["xpts"]))

        plt.tight_layout()
        plt.show()



