import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PNRQSGenLawEberlyProgram(AveragerProgram):
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
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist
        
        self.qubit_freq_placeholder = cfg.expt.freq_placeholder
        self.qubit_gf_freq = self.freq2reg(cfg.device.soc.qubit.f_gf, gen_ch = self.q_ch)
        self.qubit_gf_gain = cfg.device.soc.qubit.pulses.pi_gf.gain

        # --- Sideband parameters

        self.sideband_ch = cfg.device.soc.sideband.ch
        self.sideband_nyquist = cfg.device.soc.sideband.nyqist

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

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
    
        self.synci(200)  # give processor some time to configure pulses
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg=AttrDict(self.cfg)
        
        # for i in np.arange(len(cfg.expt.sideband_freqs)):

        #     qubit_phase = self.deg2reg(cfg.expt.qubit_phases[i])
        #     qubit_length = self.us2cycles(cfg.expt.qubit_lengths[i])

        #     sideband_freq = self.freq2reg(cfg.expt.sideband_freqs[i], gen_ch = self.sideband_ch)
        #     sideband_phase = self.deg2reg(cfg.expt.sideband_phases[i])
        #     sideband_length = self.us2cycles(cfg.expt.sideband_lengths[i])

        #     # Qubit_gf pulse

        #     self.set_pulse_registers(
        #                     ch=self.q_ch, 
        #                     style="const", 
        #                     freq=self.qubit_gf_freq, 
        #                     phase=qubit_phase,
        #                     gain=self.qubit_gf_gain, 
        #                     length=qubit_length)
        
        #     self.pulse(ch=self.q_ch)
        #     self.sync_all()
        
        #     # Sideband pulse

        #     self.set_pulse_registers(
        #             ch=self.sideband_ch,
        #             style="const",
        #             freq=sideband_freq, 
        #             phase=sideband_phase,
        #             gain=cfg.expt.sideband_gain,
        #             length=sideband_length)
            
        #     self.pulse(ch=self.sideband_ch)
        #     self.sync_all()
        
        # PNQRS 
        
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)

        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.qubitdr_freq_start,
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge")
        
        if self.qubit_pulsetype == 'const':
            print('Qubit pulse type set to const')
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.qubit_freq_placeholder,
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge)
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

class PNRQSGenLawEberlyExperiment(Experiment):
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

        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        for i in tqdm(fpts, disable = not progress):
            self.cfg.expt.freq_placeholder = i
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            qspec=PNRQSGenLawEberlyProgram(soc, self.cfg)
            avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False, debug=debug) 

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])       
        
        data={'fpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col}
        
        self.data=data
        
        # avgi_rot, avgq_rot = self.iq_rot(avgi_col, avgq_col, theta=self.cfg.device.soc.readout.iq_rot_theta)

        data_dict = {'xpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col}

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
        