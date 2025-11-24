"""
A single qubit experiment is a readout resonator coupled to a qubit.

We use the General_QM_Exps and OPXexp() classes in order to do experiments
on this system
To Do:

* Create a new folder for the calibration json file
* start_qm --> start_qm_octave
* Create a more general run experiment function with the following attributes:
    * A dictionary that controls our inputs to experiments and outputs
    * A manifest of all experiments we can run
* Implement rabi experiment. 
* Move dbm function out, might just be worth inputting in volts
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
# just to compare the different config dictionaries that I keep seeing
# from octaveintroduction import config as O_intro_config
from config import config as base_config


opx_ip = "192.168.88.252"
opx_port = 9510
octave_ip = "192.168.88.254"
octave_port = 80
con = "con1"
octave = "octave1"


class Single_Qubit_Experiments(General_QM_Exps, OPXexp):
    """
    Starting with a preexisting configuration file with various experiments, we change our qubit frequencies in
    order to actually

    
    """
    
    def __init__(self, qubitDrive, resonatorDrive, initConfig = initialConfiguration):
        """OPX configuration"""

        #This creates the elements that we are using and names them qubit and resonator
        #To do: add support for second qubit
        self.elements = {
            'qubit' : qubitDrive,
            'resonator': resonatorDrive
        }

        self.OPX_config = initConfig
        super().__init__(self.elements, initialConfiguration)
        # Long story short, the above line assigns all values in all dictionaries
        # (including named placeholders to be defined later) for each element
        # EXCEPT the operations dictionary, which is passed as an empty dictionary

        '''
        Looking at methods called in
        self.OPX_config = initConfig
        super().__init__(self.elements, initialConfiguration)

        From General_QM_Exps:
        def __init__(self, elements, OPXconfig = OPX_config):
            self.OPX_config = OPXconfig
            self.elementLimit = 5 #This is the number of elements we can handle. Since we are only using one OPX and Octave
                                #this value is 5
            if(len(elements) > self.elementLimit ):
            raise Exception("Error: Too many elements, remember there are only 5 elements we can add to this system")
            for element, driveFreq in elements.items():
                self.addChangeElement(element, self.createElement(driveFreq, {}))

        def addChangeElement(self, elementName, element):
            self.OPX_config["elements"][elementName] = element
            return None
        
        def createElement(self, elementDriveFreq, operations, IF = 50e6, flight = 224, smearing = 50, delay = 136, buffer = 0):
            """Using the fact that the bandwidth of the OPX is 350 MHz, we should ensure that all of our
                of our pulses are within this limit. Moreover we need to make sure that each operation
                has a pulse
            """
            try:
                for operation in operations:
                    self.OPX_config["pulses"][operation]
            except KeyError:
                print("Error: Specified operation not in pulses, please add pulse before adding qubit")
            def LOIF(drive_freq, IF):
                
                if(IF > 350e6):
                    raise Exception("Error: OPX bandwidth is 350 MHZ, please choose a different IF value")
                LO = drive_freq - IF

                return {"IF" : IF, "LO" : LO}

            index = len(self.OPX_config["elements"])+1 # This index gives us the element's number in the elements dictionary
            mixer = f"octave_{octave}_{index}"       # This allows us to specify a mixer and octave/OPX ports systematically
            

            """
            The next block of code allows us to define the I and 
            Q ports in the mixInputs block for the mixers in the 
            OPX config dictionary

            Element #/Index : 1   2   3   4   5  
                    I/Q Port: I Q I Q I Q I Q I Q
                    con $   : 1 2 3 4 5 6 7 8 9 10
            
            """
            IPort = 2*index - 1 
            I = (con, IPort)
            QPort = 2*index
            Q = (con, QPort)
            
            lo_and_if = LOIF(elementDriveFreq, IF)

            LO = lo_and_if["LO"] #Find an adequate LO frequency for the drive frequency
            
            #"""We need to adjust mixer values to account for IF and LO freqs"""

            switch = 2*index - 1 #This is the trigger port, note that we are defining element 1 to be triggered
                                #on port 1, element 2 on port 3, etc. Makes wiring on the octave look cleaner.

            elementDict = {
                "mixInputs": {
                    "I": I,
                    "Q": Q,
                    "lo_frequency": LO,
                    "mixer": mixer,  
                },
                "intermediate_frequency": IF,
                "operations": operations,
                "digitalInputs": {
                    "switch": {
                        "port": (con, switch),
                        "delay": delay,
                        "buffer": buffer,
                    },
                },
                "outputs": {
                    "out1": (con, 1),
                    "out2": (con, 2),
                },
                "time_of_flight": flight,
                "smearing": smearing,
            }

            self.OPX_config["mixers"][mixer][0]["intermediate_frequency"] = lo_and_if["IF"]
            self.OPX_config["mixers"][mixer][0]["lo_frequency"] = lo_and_if["LO"]

            return elementDict

        '''

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
        '''
        Listing functions and methods called in .changeElement:

        def changeElement(self, elementName, property, value):
            self.OPX_config["elements"][elementName][property] = value
            return None
        '''

        ###Setting up octave and octave configuration ###
        # octave_config = QmOctaveConfig()
        # octave_config.set_calibration_db(os.getcwd())
        # octave_config.add_device_info(octave, octave_ip, octave_port)
        # ###Create Octave mapping, see introduction.py for more info ###
        # octave_config.set_opx_octave_mapping([(con, octave)])

        # ###Creating quantum machine ####
        # self.qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)
        # self.qm = self.qmm.open_qm(self.OPX_config)

        # ###Assuming we are using the internal octave clock###=
        # self.qm.octave.set_clock(octave, clock_mode=ClockMode.Internal)
        # ###Setting the LO Source we're using (the internal octave source) ###
        # for el in self.elements:
        #     self.qm.octave.set_lo_source(el, OctaveLOSource.Internal)  # Use the internal synthetizer to generate the LO.
        #     self.qm.octave.set_rf_output_gain(el, 0)  # can set the gain from -10dB to 20dB
        #     self.qm.octave.set_rf_output_mode(el, RFOutputMode.trig_normal)  # set the behaviour of the RF switch to be 'on'.
        #                                                                      # set the behaviour of the RF switch to be on only when triggered
        #                                                                      # 'RFOutputMode' can be : on, off, trig_normal or trig_inverse
        
        # ###Sets the downconversion from the octave to RF1in and assigns it to the resonator element###
        # self.qm.octave.set_qua_element_octave_rf_in_port('resonator', octave, 1)
        # ###Sets the downconverter and its LO Source to the internal downconverter in the octave ###
        # self.qm.octave.set_downconversion(
        #     'resonator', lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
        # )


        """Setup complete, now we can run octave programs"""
    
    
    """
    Creates our quantum machine using an octave config and other values.
    Launch just before you begin a job; but after you have altered all of
    your OPX_config parameters

    """
    # TODO: change_qm_octave

    def start_QM(self, calibration = False, LO = 6e9, IF = 50e9):
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
            self.qm.octave.set_rf_output_gain(el, -5)  # can set the gain from -10dB to 20dB
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
        self.start_QM(calibration, LO = current_LO_freq, IF = np.abs(np.mean(frequency_range)-current_LO_freq))
        
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
        ###Since we have 2 different elements, we can't use calibration from start_QM ###
        """To-do: Write new start_QM function that calibrates for all elements"""
        self.start_QM(calibration = False)
        

        """Basic calibration step, needs to be replaced with a more robust method"""
        if calibration:
            self.qm.octave.calibrate_element('qubit', [(current_LO_freq_qubit, np.abs(np.mean(frequency_range)-current_LO_freq_qubit))])  # can provide many pairs of LO & IFs.
            self.qm.octave.calibrate_element('resonator', [(current_LO_freq_resonator, 50e9)])
            self.qm = self.qmm.open_qm(self.OPX_config)
            self.start_QM(calibration = False)
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

    def rabi1d(self, *args):
        dt = 12//4
        T_min = 4
        T_max = 3000 // 4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 35e3
        reset_time = 5 * T1
        Tpi=130
        avgs =3000
        a=1.0
        with program() as prog1:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            t = declare(int)  # array of time delays
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 29.4e6)
            update_frequency("qubit", 400e6)
            update_frequency("qubit_multi", 392e6)

            # update_frequency("qubit_multi", 448e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time // 4), "qubit")

                    align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration=t)
                    play("saturation_multi" , "qubit_multi", duration=t)
                    align("qubit","qubit_multi", "resonator")
                    measure("readout" , 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

                """Play a ge pi pulse and then readout"""
                wait(int(reset_time // 4), "qubit")
                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=Tpi//4)
                play("saturation_multi", "qubit_multi", duration=Tpi//4)
                align('qubit','resonator')
            #    reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

                """Just readout without playing anything"""
                wait(int(reset_time // 4), "qubit")
                align('qubit','resonator')
             #   reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return prog1, times

    


nanoTest = Single_Qubit_Experiments(3.0e9, 3.24e9)
# # # nanoTest.start_QM(calibration = True)
nanoTest.RUN_resspec(0, calibration= True, fmin=100e6, fmax=110e6, df=0.01e6)
print('Complete...')

# print('Experiment configuration:')
# # print(nanoTest.OPX_config.keys())
# for el in nanoTest.OPX_config.keys():
#     if el == 'elements':
#         print(nanoTest.OPX_config[el].keys())
#     else:
#         pass
"""
From here down, new stuff for comfig_handler class
"""

# print('Keys from oldConfig.py config:\n', initialConfiguration.keys())

# print('Keys from octaveintroduction.py config:\n', O_intro_config.keys())
# Running the above and getting O_intro_config throws a connectivity error.
# Turns out that importing from octaveintroduction changes the IP and ports.

# print('Keys from config.py:\n', base_config.keys())

"""
Assume we have a device in the fridge (qubit and responator),
and we know the drive frequencies for each (from simulations).

What information do we need to set up the experiment?
    1) Connectivity between the hardware:
        i) opx_ip
        ii) opx_port
        iii) octave_ip
        iv) octave_port
        v) names of the aforementioned
        - All of the above look to be global variables declared BEFORE the config dictionary
    2) Frequencies
        i) Qubit:
            a) drive frequency
            b) IF, hardcoded in createElement() method in General_QM_Experiments
            - LO defined as a function of drive frequency and IF in createElement() method
        ii) Readout Resonator:
            a) drive frequency
            b) IF, hardcoded in createElement() method in General_QM_Experiments
            - LO defined as a function of drive frequency and IF in createElement() method
    3) Controllers
        -its name, type, and I/O configuration (the OPX)
    4) Our elements for the PARTICULAR experiment
        i) qubits
            - potentially more than one
        ii) readout resonator/cavity
    5) Specific element operations

Return to the configuration handler later.  (Not a complete waste of time.
Now you understand the set up for experiments better)

"""

# class octave_config_handler(dict):

#     def __init__(self):
#         pass

"""
PRIORITY: Focus on the following single qubit experiments

    rabi
    ramsey
    T1
    ef spectroscopy
    
"""


