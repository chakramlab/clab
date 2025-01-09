import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class RamseyNPhotonsProgram(RAveragerProgram):

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
        
        self.piby2sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma)
        self.piby2gain = cfg.device.soc.qubit.pulses.pi2_ge.gain
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

        # add qubit and readout pulses to respective channels
        try: pulse_type = cfg.device.soc.qubit.pulses.pi2_ge.pulse_type
        except: pulse_type = 'const'

        print ("pulse type = ",pulse_type)

        if pulse_type == 'const':

            self.set_pulse_registers(ch=self.qubit_ch, style="const", 
                                freq=self.freq2reg(cfg.device.soc.qubit.f_ge), phase=0,
                                    gain=self.piby2gain, length=self.piby2sigma)

        elif pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.piby2sigma, length= self.piby2sigma* 4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(0),
                gain= self.piby2gain,
                waveform="qubit")
            print('Pulse type set to gauss')
            
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




        self.safe_regwi(self.q_rp, self.r_phase, 0)
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all()
        self.sync(self.q_rp, self.r_wait)
        self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
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

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg.expt.step))  # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+',
                   self.deg2reg(self.cfg.expt.phase_step, gen_ch=self.qubit_ch))  # advance the phase of the LO for the second π/2 pulse


class RamseyNPhotonsExperiment(Experiment):
    """Ramsey Experiment
       Experimental Config
        expt = {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200, "phase_step": deg2reg(360/50)}
         }
    """

    def __init__(self, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        tpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        for i in tqdm(zip(), disable = not progress):
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            ramsey = RamseyNPhotonsProgram(soc, self.cfg)

            self.cfg.expt.waittime_placeholder = i
            self.cfg.expt.phase_placeholder = j
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



