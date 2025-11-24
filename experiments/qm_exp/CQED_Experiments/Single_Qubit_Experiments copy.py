"""
A single qubit experiment is a readout resonator coupled to a qubit.

We use the General_QM_Exps and OPXexp() classes in order to do experiments
on this system
To Do:

* Create a new folder for the calibration json file
* start_qm --> start_qm_octave    -- Done 
* Create a more general run experiment function with the following attributes:
    * A dictionary that controls our inputs to experiments and outputs - Done
    * A manifest of all experiments we can run
* Implement rabi experiment. 
* Move dbm function out, might just be worth inputting in volts - Done
* Come up with a scheme for calibration

*** Ask Quantum Machines people the following ***

* Efficiency of the mixers that we use
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm.qua import *
import os
import time
import matplotlib.pyplot as plt
from qualang_tools.units import unit
import config

from OPXexperiments import OPXexp
from General_QM_Experiments import General_QM_Exps

from oldConfig import config as initialConfiguration 
import json


opx_ip = "192.168.88.252"
opx_port = 9510
octave_ip = "192.168.88.254"
octave_port = 80
con = "con1"
octave = "octave1"


class Single_Qubit_Experiments(General_QM_Exps, OPXexp):
    """
    Starting with a preexisting configuration file with various experiments, we change our qubit frequencies in order to actually

    
    """
    
    def __init__(self, qubitDrive, resonatorDrive, initConfig = initialConfiguration):
        """OPX configuration"""

        #Below we open the experiments config file where we define our experimental parameters

        experiment_json = open("experiments.json")
        self.experiment_config = json.load(experiment_json)
        experiment_json.close()

        #This creates the elements that we are using and names them qubit and resonator
        #To do: add support for second qubit
        self.elements = {
            'qubit' : qubitDrive,
            'resonator': resonatorDrive
        }

        self.OPX_config = initConfig
        super().__init__(self.elements, initialConfiguration)


        ###Adding relevant qubit and resonator operations ###
        qubitOperations = {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'saturation_multi': 'saturation_pulse_multi',
                'gaussian': 'gaussian_pulse',
                'test':'test_pulse',
                'pi': 'pi_pulse',
                'marker': 'marker_pulse',
            }
        resonatorOperations = {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
                'test':'test_readout_pulse',
            }

        self.changeElement('qubit', 'operations', qubitOperations)
        self.changeElement('resonator', 'operations', resonatorOperations)

        """Setup complete, now we can run octave programs"""
    
    
    """
    Creates our quantum machine using an octave config and other values.
    Launch just before you begin a job; but after you have altered all of
    your OPX_config parameters

    """
    # TODO: change_qm_octave

    def start_QM_octave(self, calibration = False, LO = 6e9, IF = 50e9):
        self.octave_config = QmOctaveConfig()
        # set up calibration database for octave (calibration_db.json)
        self.octave_config.set_calibration_db(os.getcwd())
        self.octave_config.add_device_info(octave, octave_ip, octave_port)
        ###Create Octave mapping, see introduction.py for more info ###
        self.octave_config.set_opx_octave_mapping([(con, octave)])

        ###Creating quantum machine ####
        self.qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=self.octave_config)
        self.qm = self.qmm.open_qm(self.OPX_config)
        
        ###If we want to calibrate elements beforehand we can specify###
        if calibration:
            for el in self.elements:
                print("-" * 37 + f" Calibrates {el}")
                self.qm.octave.calibrate_element(el, [(LO, IF)])  # can provide many pairs of LO & IFs.
                self.qm = self.qmm.open_qm(self.OPX_config)

        ###Assuming we are using the internal octave clock###=
        self.qm.octave.set_clock(octave, clock_mode=ClockMode.Internal)
        ###Setting the LO Source we're using (the internal octave source) ###
        for el in self.elements:
            self.qm.octave.set_lo_source(el, OctaveLOSource.Internal)  # Use the internal synthetizer to generate the LO.
            self.qm.octave.set_rf_output_gain(el, 0)  # can set the gain from -10dB to 20dB
            self.qm.octave.set_rf_output_mode(el, RFOutputMode.trig_normal)  # set the behaviour of the RF switch to be 'on'.
                                                                             # set the behaviour of the RF switch to be on only when triggered
                                                                             # 'RFOutputMode' can be : on, off, trig_normal or trig_inverse
        
        ###Sets the downconversion from the octave to RF1in and assigns it to the resonator element###
        self.qm.octave.set_qua_element_octave_rf_in_port('resonator', octave, 1)
        ###Sets the downconverter and its LO Source to the internal downconverter in the octave ###
        self.qm.octave.set_downconversion(
            'resonator', lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
        )

        
        return None

    """
    This function allows you to run any experiment and get a dictionary of results.
    Takes: 
        *experiment_name: type = string, description: name of the experiment we 
         seek to run, constraints: must be within the experiments.json and 
         implemented_experiments (see OPXexperiments.py and experiments.json)
        *calibration: type: bool, description: Choose whether or not you 
         would like to calibrate the octave before running experiment
        *gain: type = float, description: value from -10db to 20 dB that defines
         the gain from rf output, constraints: ensure that any output is within
         the limits of the octave and the qubit you are running
    Returns:
        *results: type = dictionary, contents =  {
            "I": np.array(I), #values of real part of signal received
            "Q": np.array(Q), #values of imaginary part of signal received
            "program and parameters": program_and_parameters, #A dictionary of the qua program used 
                                                                and certain parameters
                                                                like the frequency range 
                                                                or time range that's used
            "experiment parameters": self.experiment_config[experiment_name],
            "OPX parameters": self.OPX_config
        }

    """
    def RUN_experiment(self, experiment_name, calibration = False):
        program_and_parameters = self.implemented_experiments[experiment_name](self.experiment_config[experiment_name])
        prog = program_and_parameters["prog"]

        current_LO_freq = self.OPX_config["elements"]["resonator"]["mixInputs"]["lo_frequency"]


        u = unit()

        
        ###Starts the quantum machine with the parameters we have passed###
        self.start_QM_octave(calibration, LO = current_LO_freq, IF = 50)
        
        
        ### Run experiment and get results 
        job = self.qm.execute(prog)
        res = job.result_handles
        res.wait_for_all_values()

        ### Convert I and Q signal to numpy arrays in Volts
        I = u.raw2volts(res.get("I").fetch_all())
        Q = u.raw2volts(res.get("I").fetch_all())

        ### Result dictionary that can be transformed into an h5py file
        results = {
            "I": np.array(I),
            "Q": np.array(Q),
            "program and parameters": program_and_parameters,
            "experiment parameters": self.experiment_config[experiment_name],
            "OPX parameters": self.OPX_config
        }

        return results
   
    """
    This runs the resonator spectroscopy function from OPX experiemnts.
    We provide an input power in dBM, which is how much power we are inputting into the 
    fridge.
    We have frequency values fmin, fmax, and df, which define a range of frequencies from
    fmin to fmax with df intervals to plot. These are all floats.
    We can optionally include an impedance value for our wiring or LO/IF 
    frequencies that change the constant pulse that we send into the fridge.
    """
    def RUN_resspec(self, input_power_dBM, fmin = 20e6, fmax = 330e6, df = 0.1e6, impedance = 50, LO = None, IF = None, calibration = False):
        
        ### Allows us to change the IF and LO frequencies if we want ###
        if(LO is not None):
            self.changeLOfrequency('resonator', LO)
        if (IF is not None):
            self.changeIFfrequency('resonator', IF)
        
        #Converts a given value for power in dbm to amplitude in Volts
        #Optional impedance
        def powerToAmplitude(dbm, Z = 50):
            impFactor = np.sqrt(Z/1000)
            powerFactor = 10 ** (dbm/20)
            Vrms = impFactor * powerFactor
            amplitudeV = Vrms * np.sqrt(2)
            return amplitudeV 
            
        """
        We want to control how much power we are putting into the fridge. The
        following code allows us to change the const_wf waveforms in order to
        change the power we are putting into the fridge, ie, changing the 
        value of the sample voltage we are using.
        """
        input_amplitude_volts = powerToAmplitude(input_power_dBM, Z=impedance)
        self.addChangeWaveform("readout_wf", self.createWaveform('constant', input_amplitude_volts))
        

        """
        Now we run the resonator spectroscopy function with our prefered IF'
        frequency values and using our current LO frequency. We get a range of frequencies,
        I, and Q which we return as a dictionary.
        """
        prog, IFfreqs = self.resspec(fmin, fmax, df)

        current_LO_freq = self.OPX_config["elements"]["resonator"]["mixInputs"]["lo_frequency"]


        u = unit()

        frequency_range = np.add(np.array(IFfreqs), current_LO_freq)
        
        ###Starts the quantum machine with the parameters we have passed###
        self.start_QM_octave(calibration, LO = current_LO_freq, IF = np.abs(np.mean(frequency_range)-current_LO_freq))
        
        ###Set the RF output gain to 0 ###
        self.qm.octave.set_rf_output_gain('resonator', 0)
        ### At this point you can connect it to an oscilloscope and see a single pulse
        ### On RF2, but just remember that it will end on an error because the job will never
        ### end prematurely (no results are being collected) 
        

        job = self.qm.execute(prog)
        res = job.result_handles
        res.wait_for_all_values()

        I = u.raw2volts(res.get("I").fetch_all()) # changed from fetch() to fetch_all() => Rob
        Q = u.raw2volts(res.get("I").fetch_all()) # changed from fetch() to fetch_all() => Rob

        results = {
            "I": np.array(I),
            "Q": np.array(Q),
            "frequency range": frequency_range
        }

        return results
    """
    Runs Qubit spectroscopy experiment: sweeps qubit and resonator frequencies by saturating 
    qubit and measuring the resonator. Should see 2 peaks, one for the ground and 
    excited state of the qubit and one for the ground state.

    input powers in dBm for resonator and qubit can be provided, default to 0.4 V(amplitude) for saturation
    and 0.2 V (amplitude) for readout pulse

    fmin is minimum IF freq, fmax is max IF freq, df is the steps we sweep from fmax to fmin

    Impedance is value for wiring, given in Ohms
    """
    def RUN_qubitspec(self, input_power_resonator_dBm = None , input_power_qubit_dBm = None, fmin = 20e6, fmax = 330e6, df = 0.1e6, impedance = 50, calibration = False):
         ### Allows us to change the IF and LO frequencies if we want ###
        def powerToAmplitude(dbm, Z = 50):
            impFactor = np.sqrt(Z/1000)
            powerFactor = 10 ** (dbm/20)
            Vrms = impFactor * powerFactor
            amplitudeV = Vrms * np.sqrt(2)
            return amplitudeV 
        
        ### Allows us to change the resonator pulse power ###
        if (input_power_resonator_dBm is not None):
            resonator_input_amplitude_volts = powerToAmplitude(input_power_resonator_dBm, Z=impedance)
            self.addChangeWaveform("readout_wf", self.createWaveform('constant', resonator_input_amplitude_volts))
        

        ### Allows us to change the saturation pulse power if needed ###
        if (input_power_qubit_dBm is not None):
            qubit_input_amplitude_volts = powerToAmplitude(input_power_qubit_dBm, Z=impedance)
            self.addChangeWaveform("saturation_wf", self.createWaveform('constant', qubit_input_amplitude_volts))
        
        prog, IFfreqs = self.qubitspec(fmin, fmax, df)
        print(IFfreqs)

        current_LO_freq_resonator = self.OPX_config["elements"]["resonator"]["mixInputs"]["lo_frequency"]
        current_LO_freq_qubit = self.OPX_config["elements"]["qubit"]["mixInputs"]["lo_frequency"]
        print(current_LO_freq_resonator, 'LO_res')
        print(current_LO_freq_qubit, 'LO_qubit')

        u = unit()

        frequency_range = np.add(np.array(IFfreqs), current_LO_freq_qubit)
        
        ###Starts the quantum machine with the parameters we have passed###
        ###Since we have 2 different elements, we can't use calibration from start_QM_octave ###
        """To-do: Write new start_QM_octave function that calibrates for all elements"""
        self.start_QM_octave(calibration = False)
        

        """Basic calibration step, needs to be replaced with a more robust method"""
        if calibration:
            self.qm.octave.calibrate_element('qubit', [(current_LO_freq_qubit, np.abs(np.mean(frequency_range)-current_LO_freq_qubit))])  # can provide many pairs of LO & IFs.
            self.qm.octave.calibrate_element('resonator', [(current_LO_freq_resonator, 50e9)])
            self.qm = self.qmm.open_qm(self.OPX_config)
            self.start_QM_octave(calibration = False)
        ### At this point you can connect it to an oscilloscope and see a single pulse
        ### On RF1 and RF2, but just remember that it will end on an error because the job will never
        ### end prematurely (no results are being collected) 
        

        job = self.qm.execute(prog)
        res = job.result_handles
        res.wait_for_all_values()

        I = u.raw2volts(res.get("I").fetch())
        Q = u.raw2volts(res.get("I").fetch())

        results = {
            "I": np.array(I),
            "Q": np.array(Q),
            "frequency range": frequency_range
        }

        return results


nanoTest = Single_Qubit_Experiments(3.0e9, 3.24e9)
# # # nanoTest.start_QM(calibration = True)
# nanoTest.RUN_resspec(0, calibration= False, fmin=100e6, fmax=110e6, df=0.01e6)
nanoTest.RUN_experiment("qubit spectroscopy")    

