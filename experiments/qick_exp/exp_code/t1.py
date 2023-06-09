import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class T1Program(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch
        
        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
        self.f_res=self.freq2reg(cfg.device.soc.resonator.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.soc.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.pisigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.resonator.freq, gen_ch=self.res_ch)


        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.pisigma, length=self.pisigma * 4)
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=cfg.device.soc.qubit.pulses.pi_ge.gain,
            waveform="qubit")
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.pulse(ch=self.qubit_ch)  # play pi pulse
        self.sync_all()
        self.sync(self.q_rp, self.r_wait) # sets internal time offset to value stored in register self.r_wait in page self.q_rp
        self.sync_all() # no need after sync
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels
    
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step))  # update wait time
                      
                      
class T1Experiment(Experiment):
    """T1 Experiment
       Experimental Config
        expt =  {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200}
    """

    def __init__(self, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)
    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        t1 = T1Program(soc, self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)
        
        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        data_dict = {'xpts':data['xpts'], 'avgi':data['avgi'][0][0], 'avgq':data['avgq'][0][0]}

        if data_path and filename:
            self.save_data(data_path, filename, arrays=data_dict)

        return data

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
