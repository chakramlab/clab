"""
A single qubit experiment is a readout resonator coupled to a qubit.

We use the General_QM_Exps and OPXexp() classes in order to do experiments
on this system
Done: 
    * start_qm --> start_qm_octave    -- Done 
    * Create a more general run experiment function with the following attributes:
        * A dictionary that controls our inputs to experiments and outputs - Done
        * A manifest of all experiments we can run - Done (note, did it in a simpler way with the eval finction)
    * Implement rabi experiment. - Done
    * Move dbm function out, might just be worth inputting in volts - Done (note, removed volt functionality)
    * Implement h5py file for output - Done
    * Come up with a scheme for calibration  - Done
To Do:
    * Create a new folder for the calibration json file

    * Fix gaussian waveform
    * Implement experiments:
        *Ramsey
        *T1
    * Slab h5py implementation
    * move all material properties of OPX, ie, LO frequency, IF frequency into OPX config, ie, keep things outside of init. 

*** Ask Quantum Machines people the following ***

* Efficiency of the mixers that we use
"""

import json
import os
import time

import config
import h5py
import matplotlib.pyplot as plt
from General_QM_Experiments import General_QM_Exps
from oldConfig import config as initialConfiguration
from OPXexperiments import OPXexp
from qm import QuantumMachinesManager
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm.qua import *
from qualang_tools.units import unit

# opx_ip = "192.168.88.252"
opx_ip = "192.168.0.169"
opx_port = 10252 #9510
octave_ip = "192.168.0.169"
octave_port = 11254 #80
con = "con1"
octave = "octave1"
with open("experiments.json", "r") as f:
    experiment_config = json.load(f)

class Single_Qubit_Experiments(General_QM_Exps, OPXexp):
    """
    Starting with a preexisting configuration file with various experiments, we change our qubit frequencies in order to actually

    
    """
    
    def __init__(self, initConfig = initialConfiguration):
        """OPX configuration"""

        #Below we open the experiments config file where we define our experimental parameters

        # experiment_json = open("experiments.json")
        # self.experiment_config = json.load(experiment_json)
        # # experiment_json.close()

        # #This creates the elements that we are using and names them qubit and resonator
        # #To do: add support for second qubit
        # self.elements = {
        # }

        # self.OPX_config = initialConfiguration
        # super().__init__(self.elements, initConfig)


        # ###Adding relevant qubit and resonator operations ###
        # qubitOperations = {
        #         'CW': 'CW',
        #         'saturation': 'saturation_pulse',
        #         'saturation_multi': 'saturation_pulse_multi',
        #         'gaussian': 'gaussian_pulse',
        #         'test':'test_pulse',
        #         'pi': 'pi_pulse',
        #         'marker': 'marker_pulse',
        #     }
        # resonatorOperations = {
        #         'CW': 'CW',
        #         'saturation': 'saturation_pulse',
        #         'long_readout': 'long_readout_pulse',
        #         'readout': 'readout_pulse',
        #         'test':'test_readout_pulse',
        #     }

        # self.changeElement('qubit', 'operations', qubitOperations)
        # self.changeElement('resonator', 'operations', resonatorOperations)
        super().__init__({}, initConfig)
        self.elements = list(self.OPX_config['elements'].keys())
        """Setup complete, now we can run octave programs"""
        
    
    """
    Creates our quantum machine using an octave config and other values.
    Launch just before you begin a job; but after you have altered all of
    your OPX_config parameters

    """
 

    def start_QM_octave(self, calibrationFile = os.getcwd(), calibration = False, paramFile = "calibration_params.json"):
        
        self.octave_config = QmOctaveConfig()
        # set up calibration database for octave (calibration_db.json)
        self.octave_config.set_calibration_db(calibrationFile)
        self.octave_config.add_device_info(octave, octave_ip, octave_port)
        ###Create Octave mapping, see introduction.py for more info ###
        self.octave_config.set_opx_octave_mapping([(con, octave)])

        ###Creating quantum machine ####
        self.qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=self.octave_config, log_level="DEBUG" )
        self.qm = self.qmm.open_qm(self.OPX_config)
        
        ###If we want to calibrate elements beforehand we can specify###
        # if calibration:
        #     for el in elements:
        #         print("-" * 37 + f" Calibrates {el}")
        #         self.qm.octave.calibrate_element(el, [(LO, IF)])  # can provide many pairs of LO & IFs.
        #         self.qm = self.qmm.open_qm(self.OPX_config)

        ###Assuming we are using the internal octave clock###=
        # self.qm.octave.set_clock(octave, clock_mode=ClockMode.Internal) # removing this line because it seems to make it search for a nonexistent external clock
        ###Setting the LO Source we're using (the internal octave source) ###
        for el in self.elements:
            if('mixInputs' in list(self.OPX_config['elements'][el].keys())):
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

        if(calibration):
            with open(paramFile, 'r') as f:
                param_config = json.load(f)
            paramList = []
            for element in param_config:
                LOFrequencies = param_config[element]["LOFrequencies"]
                IFFrequencies = param_config[element]["IFFrequencies"]
                
                """HARDWARE LIMITED TEMPORARY MEASURE 1: We only can support 1 LO frequency and 1 IF frequency at this moment
                    due to an update that needs to be done on the OPX. In the meanwhile, the below function
                    is used; implementation for multiple freqs is present but is inefficient
                    in system at moment. We only use the middle value of paramList to account for this
                """
                for i in range(len(LOFrequencies)):
                    paramList= [(float(LOFrequencies[i]), float(IFFrequencies[i]))]
                    self.qm.octave.calibrate_element(element, paramList, close_open_quantum_machines=True)
                    self.qm = self.qmm.open_qm(self.OPX_config)
                
                # paramList = [paramList[int(len(paramList)/2)]]
                # self.qm.octave.calibrate_element(element, paramList, close_open_quantum_machines=True)
                

       
        return None
    """
    We have a calibration dictionary called calibration_params.json that we store the parameters for the qubit and resonator calibration.

    It takes the following form:
        {
            "qubit": {
                "LOFrequencies": [
                    5000000000.0
                ],
                "IFFrequencies": []
            },
            "resonator": {
                "LOFrequencies": [],
                "IFFrequencies": []
            }
        }
        The class below allows us to edit the "LOFrequencies" and "IFFrequencies" dictionary
        
        We take an element which must be either "qubit" or "resonator"

        We then pass in a paramFile which is a string that specifies the name and location of the file

        We pass in LOFrequencies which is a list of LOFrequencies we wish to test for
        
        We pass in IFFrequencies which is a list of IFFrequencies we wish to test for

        Note that IFFrequencies and LOFrequencies must be the same length: in the actual calibration step
        we will make one to one in order maps from LOFrequencies to IFFrequencies that will lead to pairs of (LO, IF):
            LOFreq = [LO1, LO2, LO3, ...]
            IFFreq = {IF1, IF2, IF3, ...}

            ListOfLOIFs = [(LO1, IF1), (LO2, IF2), (LO3, IF3)]
    """
    def editCalibrationParams(self,  element, paramFile = "calibration_params.json", LOFrequencies = None, IFFrequencies = None):
        if(not (element == "qubit" or element == "resonator")):
            raise Exception("Element not valid, please choose qubit or resonator")

        with open(paramFile, "r") as f:
            param_config = json.load(f)
        
        if(LOFrequencies is not None):
            if(len(IFFrequencies)!=len(LOFrequencies)):
                raise Exception("IFFrequency length is not equivalent to LOFrequency length")

            param_config[element]["LOFrequencies"] = LOFrequencies
        
        if(IFFrequencies is not None):
            if(len(IFFrequencies)!=len(LOFrequencies)):
                raise Exception("IFFrequency length is not equivalent to LOFrequency length")

            param_config[element]["IFFrequencies"] = IFFrequencies
        
        if(len(param_config[element]["LOFrequencies"])!=len(param_config[element]["IFFrequencies"])):
            raise Exception("IFFrequency length is not equivalent to LOFrequency length")

        with open(paramFile, "w") as f:
            json.dump(param_config, f)
         
        return param_config
    """
    HARDWARE: Connect RF1 to RF1in and RF2 to RF2in

    Allows you to calibrate individual elements
    
    From the paramFile (which is set to a default but can be changed) we extract LOFrequencies and IFFrequencies.
    From here we will make one to one in order maps from LOFrequencies to IFFrequencies that will lead to pairs of (LO, IF):
            LOFreq = [LO1, LO2, LO3, ...]
            IFFreq = [IF1, IF2, IF3, ...]

            paramList = [(LO1, IF1), (LO2, IF2), (LO3, IF3)]
    
    We can then pass this to the calibrate_element function for our element (which should be a string that 
    is either "qubit" or "resonator"). 

    Calibrating requires you to have already started the octave with the start_QM_octave function. 
    """
    def calibrate(self, element, paramFile = "calibration_params.json"):


        with open(paramFile, 'r') as f:
            param_config = json.load(f)
        
        paramList = []

        LOFrequencies = param_config[element]["LOFrequencies"]
        IFFrequencies = param_config[element]["IFFrequencies"]

        for i in range(len(LOFrequencies)):
            paramList.append((float(LOFrequencies[i]), float(IFFrequencies[i])))

        self.qm.octave.calibrate_element(element, paramList, save_to_db=True,close_open_quantum_machines=True)
        self.qm = self.start_QM_octave(calibration=False)

        return paramList

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
    def RUN_experiment(self, experiment_name, config, elementsAndGains = None):
        
        
        #program_and_parameters = self.implemented_experiments[experiment_name](self.experiment_config[experiment_name])
        fcommand = f"self.{experiment_name}({config[experiment_name]})"
        program_and_parameters = eval(fcommand)
        prog = program_and_parameters["prog"]

        current_LO_freq = self.OPX_config["elements"]["resonator"]["mixInputs"]["lo_frequency"]


        u = unit()

        ###Starts the quantum machine with the parameters we have passed###
        self.start_QM_octave()        
        
        ###Change the gain of elements
        if(elementsAndGains is not None):
            for element, gain in elementsAndGains.items():
                self.qm.octave.set_rf_output_gain(element, gain)

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
            "program_name": experiment_name,
            "experiment parameters" : config,
            "OPX parameters": self.OPX_config
        }

        ### Adds all of the program parameters, ex: frequencies, times, etc, to the results dictionary ###
        for key in program_and_parameters:
            if key != "prog":
                results[key] = (str)(program_and_parameters[key])

        ### Final results dictionary ###
        return results
  
    """
    Allows you to save a results dictionary to an existing or a new h5py file.

    results(dict): dictionary of results, should be in the style of RUN_experiment's results dictionary being returned
    resultName(string): name of the dataset. Note, to find individual attributes of a dataset need to do dataset[resultName+"_"+attributeName]
    fileName(string): name of the h5py file, will create if it doesn't exist.
    path (string): path to h5py file directory.
    """
    def saveResults(self, results, resultName, fileName, path = ""):
        with h5py.File(path+fileName, "a") as f:
            for key in results:
                if(not isinstance(results[key], dict)):
                    f.create_dataset(resultName+"_"+key, data = results[key])
                else:
                    f.create_dataset(resultName+"_"+key, data = str(results[key]))

   
   


# nanoTest = Single_Qubit_Experiments()
# nanoTest.start_QM_octave(calibration=False)
# # # # results = nanoTest.editCalibrationParams("qubit", LOFrequencies=[5e9], IFFrequencies=[5e6]) 
# nanoTest.start_QM_octave(calibration=True)

# # nanoTest.RUN_experiment("qubitspec", experiment_config)
# # # nanoTest.saveResults(results, "test", "test")   

