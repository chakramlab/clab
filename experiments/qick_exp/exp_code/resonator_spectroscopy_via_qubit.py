import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class ResonatorSpectroscopyViaQubitProgram(RAveragerProgram):
    def initialize(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # --- Readout parameters
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_ch_nyquist = cfg.device.soc.resonator.nyqist
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # --- Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist


        self.q_length = self.us2cycles(cfg.expt.length)
        self.q_freq_step = self.freq2reg(cfg.expt.step)
        self.q_gain = cfg.expt.gain
        
        # --- Cavity drive parameters
        self.cavdr_ch=cfg.device.soc.storage.ch
        self.cavdr_reg_page =self.ch_page(self.cavdr_ch)     # get register page for cavdr_ch
        self.cavdr_freq_reg = self.sreg(self.cavdr_ch, "freq")   # get frequency register for cavdr_ch
        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        self.cavdr_freq_start = self.freq2reg(cfg.expt.start, gen_ch = self.cavdr_ch)
        self.cavdr_freq_step = self.freq2reg(cfg.expt.step)
        self.cavdr_length = self.us2cycles(cfg.expt.length)
        self.cavdr_gain = cfg.expt.gain
        try:
            self.cavdr_pulse_type = cfg.expt.pulse_type
        except:
            self.cavdr_pulse_type = "const"

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)
        self.declare_gen(ch=self.cavdr_ch, nqz=2)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.res_freq, gen_ch=self.res_ch)


        # convert frequency to DAC frequency
        self.freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        
        #initializing pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)
        
        if self.cavdr_pulse_type == "const":
        
            self.set_pulse_registers(
                ch=self.cavdr_ch,
                style="const",
                freq=self.cavdr_freq_start,
                phase=self.deg2reg(0, gen_ch=self.cavdr_ch), # 0 degrees
                gain=self.cavdr_gain,
                length=self.cavdr_length)
            
        else:

            self.sigma_cavdr = self.cavdr_length//4
            self.add_gauss(ch=self.cavdr_ch, name="cavdr", sigma=self.sigma_cavdr, length=self.sigma_cavdr * 4)

            self.set_pulse_registers(
                ch=self.cavdr_ch,
                style="arb",
                freq=self.cavdr_freq_start,
                phase=self.deg2reg(0, gen_ch=self.cavdr_ch), # 0 degrees
                gain=self.cavdr_gain,
                waveform="cavdr")
        
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)

        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':

            self.add_gauss(ch=self.q_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge")
            
        if self.qubit_pulsetype == 'const':
            print('Qubit pulse type set to const')
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(cfg.device.soc.qubit.f_ge), 
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge)
        
        self.synci(200)  # give processor some time to configure pulses
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg=AttrDict(self.cfg)

        self.pulse(ch=self.cavdr_ch)
        self.sync_all()
        
        self.pulse(ch=self.q_ch)
        self.sync_all()

        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

    def update(self):
        self.mathi(self.cavdr_reg_page, self.cavdr_freq_reg, self.cavdr_freq_reg, '+', self.cavdr_freq_step) # update frequency list index

class ResonatorSpectroscopyViaQubitExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        qspec=ResonatorSpectroscopyViaQubitProgram(soc, self.cfg)
        x_pts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'fpts':x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data
        
        avgi_rot, avgq_rot = self.iq_rot(avgi[0][0], avgq[0][0], theta=self.cfg.device.soc.readout.iq_rot_theta)

        data_dict = {'xpts':x_pts, 'avgi':avgi[0][0], 'avgq':avgq[0][0], 'avgi_rot': avgi_rot, 'avgq_rot': avgq_rot}

        # if data_path and filename:
        #     file_path = data_path + get_next_filename(data_path, filename, '.h5')
        #     with SlabFile(file_path, 'a') as f:
        #         f.append_line('freq', x_pts)
        #         f.append_line('avgi', avgi[0][0])
        #         f.append_line('avgq', avgq[0][0])
        #     print("File saved at", file_path)
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        

        return data

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
        