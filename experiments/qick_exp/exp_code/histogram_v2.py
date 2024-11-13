import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class HistogramProgram(RAveragerProgram):
    def initialize(self):
        self.cfg.expt.expts=1
        self.cfg.expt.start=0
        self.cfg.expt.step=0
        self.cfg.update(self.cfg.expt)
        self.cfg = AttrDict(self.cfg)

        cfg = self.cfg

        # self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        # self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch

        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        # self.safe_regwi(self.q_rp, self.q_gain, cfg.expt.start)

        self.f_res = self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length = self.us2cycles(self.cfg.expt.readout_length, gen_ch=self.res_ch)
        # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
        # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)

        # add qubit and readout pulses to respective channels

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
            
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(0),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
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

    def body(self):
        cfg = self.cfg
        print(self.cfg.expt.state_temp)
        if self.cfg.expt.state_temp == 'e':
            print('playing ge pulse')
            self.play_pige_pulse(phase=0)
            self.sync_all()
        if self.cfg.expt.state_temp == 'f':
            self.play_pige_pulse(phase=0)
            self.sync_all()
            self.play_pief_pulse(phase=0)
            self.sync_all()

        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=self.us2cycles(cfg.expt.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels

    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg

        # print(self.di_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)))
        # print(cfg.device.readout.length)
        shots_i0 = self.di_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(
            cfg.device.soc.readout.length)
        shots_q0 = self.dq_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(
            cfg.device.soc.readout.length)
        return shots_i0, shots_q0


class HistogramExperiment(Experiment):
    """Histogram Experiment
       Experimental Config
        expt =  {"reps": 10000}
    """

    def __init__(self, path='', prefix='Histogram', config_file=None, progress=None, datapath=None, filename=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

        if datapath is None:
            self.datapath = None
        else:
            self.datapath = datapath

        if filename is None:
            self.filename = None
        else: 
            self.filename = filename

    def acquire(self, progress=False, debug=False):
        i_g = []
        q_g = []
        i_e = []
        q_e = []
        i_f = []
        q_f = []
        data_dict = {}
        if self.cfg.expt.f_state:
            states = ['g', 'e', 'f']
        else:
            states = ['g', 'e']
        for state in states:
            self.cfg.expt.state_temp = state
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            histpro = HistogramProgram(soc, self.cfg)
            x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                                progress=progress)
            i, q = histpro.collect_shots()
            if state=='g':
                i_g.append(i[0])
                q_g.append(q[0])
                data_dict['ig'] = list(i_g[0])
                data_dict['qg'] = list(q_g[0])
            if state=='e':
                i_e.append(i[0])
                q_e.append(q[0])
                data_dict['ie'] = list(i_e[0])
                data_dict['qe'] = list(q_e[0])
            if state=='f':
                i_f.append(i[0])
                q_f.append(q[0])
                data_dict['if'] = list(i_f[0])
                data_dict['qf'] = list(q_f[0])

        # data_dict = {'ig': i_g[0], 'qg': q_g[0], 'ie': i_e[0], 'qe': q_e[0], 'if': i_f[0], 'qf': q_f[0]}

        if self.datapath and self.filename:
            self.save_data(data_path=self.datapath, filename=self.filename, arrays=data_dict)
        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        i_g = data['i0'][0]
        q_g = data['q0'][0]
        i_e = data['i0'][1]
        q_e = data['q0'][1]

        fid, threshold, angle = self.hist(data=[i_g, q_g, i_e, q_e], plot=False, ran=300)
        data['fid'] = fid
        data['angle'] = angle
        data['threshold'] = threshold

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print(self.fname)

        i_g = data['i0'][0]
        q_g = data['q0'][0]
        i_e = data['i0'][1]
        q_e = data['q0'][1]

        fid, threshold, angle = self.hist(data=[i_g, q_g, i_e, q_e], plot=True, ran=200)

        plt.tight_layout()
        plt.show()
        print("fidelity:", fid)
        print("angle:", -angle*180/np.pi)
        print("threshold:", threshold)
        data['fid'] = fid
        data['angle'] = -angle*180/np.pi
        data['threshold'] = threshold

    def hist(self, data=None, plot=True, ran=1.0):
        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]

        numbins = 200

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0].scatter(ig, qg, label='g', color='b', marker='*')
            axs[0].scatter(ie, qe, label='e', color='r', marker='*')
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xe, ye, color='k', marker='o')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        # print(xg, xe)

        xlims = [xg - ran, xg + ran]
        ylims = [yg - ran, yg + ran]

        if plot == True:
            axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
            axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xe, ye, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title('Rotated')
            axs[1].axis('equal')

            """X and Y ranges for histogram"""

            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
        #        axs[2].set_xlabel('I(a.u.)')

        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        # axs[2].set_title(f"Fidelity = {fid*100:.2f}%")

        return fid, threshold, theta, ig_new, qg_new, ie_new, qe_new
