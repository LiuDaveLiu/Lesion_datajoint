import datajoint as dj
import pipeline.lab as lab#, ccf
from pipeline.pipeline_tools import get_schema_name
import pipeline.experiment as experiment
import numpy as np

schema = dj.schema(get_schema_name('tracking'))
#[experiment]  # NOQA flake8
@schema
class VideoTrialNum(dj.Manual):
    definition = """
    ->experiment.Session
    video_trial_num:                 smallint 		# how many trials for recorded because usually the last couple trials are not recorded
    ---
    """

@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device:                    varchar(20)     # device type/function
    ---
    tracking_position:                  varchar(20)     # device position
    sampling_rate:                      decimal(8, 4)   # sampling rate (Hz)
    tracking_device_description:        varchar(100)    # device description
    """
    contents = [
       ('Camera 0', 'side_face', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
       ('Camera 1', 'front_face', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)')]


@schema
class Tracking(dj.Imported): 
    '''
    Video feature tracking.
    Position values in px; camera location is fixed & real-world position
    can be computed from px values.
    '''

    definition = """
    -> experiment.SessionTrial
    -> TrackingDevice
    ---
    tracking_samples:           int             # number of events (possibly frame number, relative to the start of the trial)
    """

    class NoseTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        nose_x:                 longblob        # nose x location (px)
        nose_y:                 longblob        # nose y location (px)
        nose_likelihood:        longblob        # nose location likelihood
        """

    class TongueTracking(dj.Part): # for tongue center
        definition = """
        -> Tracking
        ---
        tongue_mid_x:               longblob        # tongue x location (px)
        tongue_mid_y:               longblob        # tongue y location (px)
        tongue_mid_likelihood:      longblob        # tongue location likelihood
        """

    class JawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        jaw_x:                  longblob        # jaw x location (px)
        jaw_y:                  longblob        # jaw y location (px)
        jaw_likelihood:         longblob        # jaw location likelihood
        """

    class LeftPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        left_paw_x:             longblob        # left paw x location (px)
        left_paw_y:             longblob        # left paw y location (px)
        left_paw_likelihood:    longblob        # left paw location likelihood
        """

    class RightPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        right_paw_x:            longblob        # right paw x location (px)
        right_paw_y:            longblob        # right_paw y location (px)
        right_paw_likelihood:   longblob        # right_paw location likelihood
        """
#--------NT added for more tracking points 11/19/2019 --------------
        
    class LeftTongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_left_x:             longblob        # left paw x location (px)
        tongue_left_y:             longblob        # left paw y location (px)
        tongue_left_likelihood:    longblob        # left paw location likelihood
        """
        
    class RightTongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_right_x:             longblob        # left paw x location (px)
        tongue_right_y:             longblob        # left paw y location (px)
        tongue_right_likelihood:    longblob        # left paw location likelihood
        """
        
    class RightLickPortTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        right_lickport_x:             longblob        # left paw x location (px)
        right_lickport_y:             longblob        # left paw y location (px)
        right_lickport_likelihood:    longblob        # left paw location likelihood
        """  
        
    class LeftLickPortTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        left_lickport_x:             longblob        # left paw x location (px)
        left_lickport_y:             longblob        # left paw y location (px)
        left_lickport_likelihood:    longblob        # left paw location likelihood
        """ 
        
    
#-------NT added to convert frames to time 11/19/2019 -------------   
@schema
class VideoFiducialsTrial(dj.Computed): # converting frames to time relative to go cue
    definition = """
    -> Tracking
    -> experiment.SessionTrial
    ---
    fiducial_time:                   longblob      # time for each frame relative to go cue in sec         
    """
    #key_source = lab.Subject * experiment.SessionTrial() * TrackingDevice() * (tracking.Tracking) #& 'tracking_device_id=1')

                
    def make(self, key): 
        subject_id = key['subject_id']
        trial = key['trial']
        #print(trial)
        session = key['session']
        # get timestamps relative to go cue
        time_go1 = (experiment.TrialEvent & {'trial_event_type':"go"} & {'subject_id': subject_id}).fetch()
        time_go = time_go1['trial_event_time']
                                  
        if ((experiment.TrialEvent & {'trial_event_type':"presample"} & {'subject_id': subject_id}).fetch()).size > 0:
            time_start1 = (experiment.TrialEvent & {'trial_event_type':"presample"} & {'subject_id': subject_id}).fetch()
            time_start = time_start1['trial_event_time']
        else:
            time_start1 = (experiment.TrialEvent & {'trial_event_type':"sample"} & {'subject_id': subject_id}).fetch()
            time_start = time_start1['trial_event_time']
            
        time_go_aligned = time_go #- time_start[-1]
        
        
        # get the frame time
        tracking_samples = (Tracking & {'subject_id': subject_id} & {'trial' : trial} & {'session' : session}).fetch('tracking_samples')
        #tracking_samples = tracking_data['tracking_samples']
        t = np.linspace(0,int(tracking_samples)-1,int(tracking_samples))
        
        camera = (TrackingDevice & {'subject_id': subject_id}).fetch()
        sampling_rate = camera['sampling_rate']
             
        dt = sampling_rate[1]        
        dt = float(dt)
        t1 = t/dt #-1/dt # frame to sec
        #for x in range(0,len(tracking_samples)+1):
            #temp_t = t1 - float(time_go_aligned[x])
        
        temp_t = t1 - float(time_go_aligned[trial])
        #print(temp_t)
        key['fiducial_time'] = temp_t
        
        # insert the key into self
        self.insert1(key)
        


