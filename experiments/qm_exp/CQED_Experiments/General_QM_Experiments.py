import config

"""
The following are the ip address, opx port, octave port, name
of OPX controller, and the name of the octave.
"""
qop_ip = "172.0.0.1"
cluster_name = "Cluster_1"
opx_port = None

octave_port = 11250  # Must be 11xxx, where xxx are the last three digits of the Octave IP address
con = "con1"
octave = "octave1"



"""
CQED creates a quantum machine with the OPX and Octave that 
allows you to run experiments on a specified number of qubits and
resonators.

Each qubit/resonator passed in has the following properties:
* A drive frequency
* A name

So, we submit a dictionary with the following properties:

elements = {
    "name of qubit 1" : drive frequency as a double, ex: 7.65456e9
    "name of resonator 1" : drive frequency as a double, ex: ...
    etc
}

Now, this orients the octave and OPX towards creating these elements,
and we can add waveforms/pulses/operations that allow us to do more complex
operations.

Notably, all elements are mixInputs, ie, we are always using IQ upconverting mixing 
to create pulses we can use. 

Generally, it is best practice to create elements via initialization, add
waveforms (both digital and analog) and integration weights that will be used, 
create pulses, and add these pulses as operations to associated elements. An 
example is commented at the bottom of this file. 

Important to note that we do not change the correction matrix in the mixers. We do,
however, change the LO and IF frequencies
"""
OPX_config = config.config

# TODO: change name to qm_opx_config_handler
class General_QM_Exps():
    """
    The initialization only adds elements, pulses and other functionality need to be manually added
    """
    def __init__(self, elements, OPXconfig = OPX_config):
        self.OPX_config = OPXconfig
        self.elementLimit = 5 #This is the number of elements we can handle. Since we are only using one OPX and Octave
                              #this value is 5
        if(len(elements) > self.elementLimit):
            raise Exception("Error: Too many elements, remember there are only 5 elements we can add to this system")
        
        for element, driveFreq in elements.items():
            self.addChangeElement(element, self.createElement(driveFreq, {}))
        super().__init__()

    """
    This allows us to create a waveform with type = "arbitrary" or type = "constant"
    sample = an amplitude constant if constant type, and is a list with length of pulse duration and 
    forms an envelope for the wave. Note that the amplitude/envelope values are in units of volts and are
    not peak to peak, ie, specifies the amplitude of the wave
    """
    def createWaveform(self, type, sample):
        
        if(type == "constant"):
            waveform = {
                "type": type,
                "sample": sample
            }
            return waveform
        elif(type == "arbitrary"):
            waveform = {
                "type": type,
                "samples": sample
            }
        else:
            raise Exception("type not arbitrary or constant")
    """
    This creates a digital waveform with samples, which is a list of tuples:
    [(1, 40), (0, 50)]. The example I have given runs a high trigger for 40 seconds
    and then a low triger for 50 nanoseconds

    Note that to keep always high, you can pass:
    [(1, 0)], the 0 means duration of the pulse

    And for always low, you can pass:
    [(0, 0)]
    """
    def createDigitalWaveform(self, samples):
        return {"samples": samples}

    """
    This function returns an element dictionary (see https://docs.quantum-machines.co/0.1/assets/qua_config.html for what is included in an element)
    for a qubit.

    The first thing passed in is a qubit, this is just the drive frequency of the qubit, and is given as a float in hz

    We then specify a list of operations, this is a dictionary like the following:
            {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #         "readout": "readout_pulse",
        #     },

    The name of the operation can be whatever you like, however, you must make sure that
    the pulse has been added. Otherwise there will be an error.

    Optional params (set to default values):
    We pass in a desired Intermediate frequency IF, this can also just be a float.
    Note that IF has to be within 350 MHz

    we can pass in a time of flight (a float) flight which allows us to account for
    digital readout windows for demodulation and measurement. More details can be found here:
    https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/demod/#timing-of-the-measurement-operation

    Finally, we can pass in a smearing parameter, delay, and buffer parameters (float)  

    Important note: you need to still add the element to the config file after creating it, this is done
    so in the initialization, however, needs to be done so explicitly after initialization
    using the add element function
    """
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
    """
    creates the integration weights in the config file. Importantly, notice
    how you need to specify cosWeights and sineWeights, these should be in the following:

    [(weightVal, length), (weightVal, length), ..., (weightVal, length)]

    where eavh weightVal is between -1.0 and 1.0 and length is given in ns. All lengths should be 
    given in ns. Sum of all lengths should be the length of the pulse given.
    """
    def createIntegrationWeights(self, cosWeights, sineWeights):
        return {
            "cosine": cosWeights,
            "sine": sineWeights
        }
    """
    This creates the pulses that we are actually going to use.

    First, we define an operation, either "control" or "measurement"
    Control operations are for driving a qubit or cavity mode, and measurement operations
    are for readout

    Then, we define a length in ns, this is the length of the pulse.

    We assign a waveform for the I and Q outputs (not I and Q tilde, distinction given
    here: https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Introduction/qua_overview/?h=play#mixed-inputs-element
    as waveformI and waveformQ. These are strings and should be associated with the key value of 
    a waveform in the waveforms sections. 

    integrationWeights is a key string for a value in the integration_weights dictionary in
    the config. It should NOT be provided for control operations but MUST be 
    provided for measurement operations. Integration weights is an important step in
    demodulation: https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Introduction/config/?h=integra#integration-weights

    digitalMarker is optional but specifies a key string for a value in the 
    digital_waveforms dictionary. Note, while it is not NECESSARY to have this value,
    it is highly recommended to do so. Most applications require some form of triggering
    that the digital waveforms does. 

    This function returns a pulse dictionary that then needs to be manually
    added using the addPulse function to the config file.

    To be implemented: waveformI, waveformQ, integrationWeights, and digital_Marker should
    all have the same total length and be equal to the length value. THIS IS NOT CHECKED FOR
    IN THE CODE. You must ensure all of your waveforms are sampled to the same length.
    
    """
    def createPulse(self, operation, length, waveformI, waveformQ, integrationWeights = None, digital_Marker = None):
        """
        The following checks make sure that you are not adding elements that are out of 
        spec with the way that the octave works. Look at the comment that explains
        this method to understand what is in spec. 
        """
        if(operation == "control" and integrationWeights is not None):
            raise Exception("Error: you have included integration weights for a control pulse, you can only include these for a measurement pulse")
        try:
            self.OPX_config["waveforms"][waveformI]
            self.OPX_config["waveforms"][waveformQ]
        except KeyError:
            print("Error: I or Q waveforms do not exist in the config file")
        if digital_Marker is not None:
            try:
                self.OPX_config["digital_waveforms"][digital_Marker]
            except KeyError:
                print("Error: Digital marker is not in digital waveforms of the config")
        
        if integrationWeights is not None:
            falseName = None
            try:
                for name, integrationWeight in integrationWeights.items:
                    self.OPX_config["integration_weights"][integrationWeight]
                    falseName = name
            except KeyError:
                print(f"Error: the integration weight with name {falseName} is not in the config")
        
        """we now actually define the pulse """
        pulseDict = {}
        if(operation == "control"):
            pulseDict = {
                "operation": operation,
                "length": length,
                "waveforms": {
                    "I": waveformI,
                    "Q": waveformQ,
                },
            }
            if digital_Marker is not None:
                pulseDict["digital_marker"] = digital_Marker

        elif(operation == "measurement"):
            if(integrationWeights is None):
                raise Exception("Error: no integration weights given for measurement pulse")
            pulseDict = {
                "operation": operation,
                "length": length,
                "waveforms": {
                    "I": waveformI,
                    "Q": waveformQ,
                },
                "integration_weights": integrationWeights
            }
            if digital_Marker is not None:
                pulseDict["digital_marker"] = digital_Marker
        else:
            raise Exception("Error: operation is not measurement or control")
        
        return pulseDict
        

    """
    changes the given property of elementName in the config file
    with the value. note that LO frequency is changed separately
    """
    def changeElement(self, elementName, property, value):
        self.OPX_config["elements"][elementName][property] = value
        return None
    """
    Changes the sampling, ie the voltages, of a pulse
    """
    def changeSample(self, sample, pulseName):
        pulse = self.OPX_config['waveforms'][pulseName]
        if(pulse['type'] == 'constant'):
            if(not isinstance(sample, float)):
                raise TypeError(f"sample value provided not a float for {pulseName} waveform")
            self.OPX_config['waveforms'][pulseName]['sample'] = sample
            return self.OPX_config['waveforms'][pulseName]
        if(pulse['type'] == 'arbitrary'):
            if(not isinstance(sample, list)):
                raise TypeError(f"sample value provided not a list for {pulseName} waveform")
            self.OPX_config['waveforms'][pulseName]['samples'] = sample
            return self.OPX_config['waveforms'][pulseName]
    
    """
    replaces LO frequency of elementName by value
    """
    def changeLOfrequency(self, elementName, value):
        try:
            self.OPX_config["elements"][elementName]
        except KeyError:
            raise Exception("Element is not in elements dictionary")
        
        self.OPX_config["elements"][elementName]["mixInputs"]["lo_frequency"] = value
        
        mixer = self.OPX_config["elements"][elementName]["mixInputs"]["mixer"]
        self.OPX_config["mixers"][mixer][0]["lo_frequency"] = value

        return None
    """
    replaces IF frequency of elementName by value
    """
    def changeIFfrequency(self, elementName, value):
        try:
            self.OPX_config["elements"][elementName]
        except KeyError:
            raise Exception("Element is not in elements dictionary")


        self.OPX_config["elements"][elementName]["intermediate_frequency"] = value
        
        mixer = self.OPX_config["elements"][elementName]["mixInputs"]["mixer"]
        self.OPX_config["mixers"][mixer][0]["intermediate_frequency"] = value

        return None
    """
    adds or changes an element to the config file. ElementName is the name of the element (string)
    and element is the body, ie, a dictionary
    """
    def addChangeElement(self, elementName, element):
        self.OPX_config["elements"][elementName] = element
        return None
    """
    adds or changes a waveform to the config file. waveformName is the name of the waveform
    and the waveform is a dictionary in the style of the create waveform function.
    """
    def addChangeWaveform(self, waveformName, waveform):
        self.OPX_config["waveforms"][waveformName] = waveform
        return None
    """
    adds or changes a digital waveform to the config file, this allows for triggering from the 
    digital markers of the OPX. adds the waveform: 
    name: DWave
    , where name is the name of the waveform and DWave should be in the style of 
    create digital waveform.
    """
    def addChangeDigitalWaveform(self, name, DWave):
        self.OPX_config["digital_waveforms"][name] = DWave
        return None

    """
    Adds or changes a pulse to the pulse dictionary in the config file. name is a string that
    is the key of the pulse, and Pulse should be a dictionary in the fashion of the 
    createPulse() method
    """
    def addChangePulse(self, name, Pulse):
        self.OPX_config["pulses"][name] = Pulse
        return None
    """
    Adds or changes integration weights into the config file. name is the key value
    of the weights and weights should be in a dictionary in the style of 
    createIntegrationWeights function
    
    """
    def addChangeIntegrationWeights(self, name, weights):
        self.OPX_config["integration_weights"][name] = weights
        return None
    """
    Adds or changes an operation to a specified element. element is the key of the element,
    ie, its name, name is the name of the operation, and operation is a specified
    operation 
    """
    def addChangeOperationToElement(self, element, name, operation):
        try:
            self.OPX_config["pulses"][operation]
        except KeyError:
            print("Error: Specified operation not in pulses, please add pulse before adding operation")
        try:
            self.OPX_config["elements"][element]
        except KeyError:
            print("Error: Specified element not in elements, please add element before adding operation")
        self.OPX_config["elements"][element]["operations"][name] = operation
###test data below###
# elements = {
#     "qubit 1" : 4.68553e9,
#     "resonator" : 7.68553e9
# }
# system = General_QM_Exps(elements)
# system.addChangeDigitalWaveform("ON", system.createDigitalWaveform([(1, 0)]))
# system.addChangeWaveform("const_wf", system.createWaveform("constant", 0.1))
# system.addChangeWaveform("zero_wf", system.createWaveform("constant", 0.0))

# system.addChangeIntegrationWeights("cosine weights", system.createIntegrationWeights([(1.0, 1000)], [(0.0, 1000)]))
# system.addChangePulse("const", system.createPulse("control", 1000, "const_wf", "zero_wf", integrationWeights=None, digital_Marker="ON"))
# system.addChangeOperationToElement("qubit 1", "cw", "const")
# debug = True ##Just a point to put a breakpoint at

