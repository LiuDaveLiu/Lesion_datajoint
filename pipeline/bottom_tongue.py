import datajoint as dj
dj.config["enable_python_native_blobs"] = True
import pipeline.lab as lab#, ccf
from pipeline.pipeline_tools import get_schema_name
import pipeline.experiment as experiment
import pipeline.tracking as tracking
import numpy as np
import math
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import statistics as st
import math
import scipy.signal
import pandas as pd




schema = dj.schema(get_schema_name('bottom_tongue'))

#[experiment]  # NOQA flake8

def movmean(interval, window_size): # calculate the moving average; the beginning and the end has some issues
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def cart2pol(x, y):
    theta = np.nan
    rho = []
    for i in enumerate(x):
        rho = (np.sqrt(x**2 + y**2))
        theta1 = np.arctan2(y, x)    
    for i in enumerate(theta1):
        theta2 = math.degrees(i[1])
        theta = np.append(theta,theta2)  
    theta = np.delete(theta,0,None)   
    return (theta, rho)

#%%

@schema
class Parameters(dj.Manual): # defining camera pixels
    definition = """
    parameter_name                     :  varchar(200)
    ---
    parameter_value                    :  blob
    parameter_description              :  varchar(4000) #5.7 / 152.2 or 5.7 / 80
    """
    contents = [('Earlycamera_bottomview_pixels_to_mm','0.037450722733245734','pixels to mm conversion factor'),
                   ('Latecamera_bottomview_pixels_to_mm','0.07125000000000001','pixels to mm conversion factor')]
@schema
class Camera_pixels(dj.Computed): # pick camera pixel settings based on session
    definition = """
    -> experiment.Session
    ---
    camera_pixels_value : float(8, 4)
    """
    def make(self, key):
        session_date = (experiment.Session & key).fetch('session_date')
        session_date = session_date[0]
        
        def to_integer(dt_time):
            return 10000*dt_time.year + 100*dt_time.month + dt_time.day
        
        session_date_int = to_integer(session_date)
        
        if session_date_int < 20190710:
            Camera_pixels_value = 5.7 / 152.2
        else:
            Camera_pixels_value = 5.7 / 80
        
        key['camera_pixels_value'] = Camera_pixels_value
        self.insert1(key)  
    

@schema
class VideoTongueTrial(dj.Computed): # units are mm, deg, and seconds
    definition = """
    -> tracking.Tracking
    -> experiment.SessionTrial
    ---
    lick_peak_x= null:longblob      # time for each frame relative to go cue in sec  
    lick_peak_y= null:longblob      # time for each frame relative to go cue in sec
    lick_amplitude= null:longblob  
    lick_vel_linear= null:longblob   
    lick_vel_angular= null:longblob
    lick_yaw= null:longblob
    lick_yaw_relative= null:longblob 
    lick_yaw_avg= null:longblob
    lick_yaw_avg_relative= null:longblob
    lick_horizoffset= null:longblob
    lick_horizoffset_relative= null:longblob
    lick_rt_electric= null:longblob
    lick_rt_video_onset= null:longblob
    lick_rt_video_peak= null:longblob
    """
    class Raw_data(dj.Part):
        definition = """ # raw excel file data from deeplabcup
        -> VideoTongueTrial
        ---
        x= null:longblob
        y= null:longblob
        t= null:longblob
        pks_idx= null:longblob
        trough_idx= null:longblob
        lp_x= null:longblob
        lp_y= null:longblob
        rp_x= null:longblob
        rp_y= null:longblob
        nose_x= null:longblob
        nose_y= null:longblob
        """
    #key_source = lab.Subject * experiment.SessionTrial() * TrackingDevice() * (tracking.Tracking) #& 'tracking_device_id=1')
    # lick_amplitude:tongue displacement in x,y at the peak of the lick, peak is defined at 75% from trough
    # lick_vel_linear:longblob     # median tongue linear velocity during the lick duration, from peak to trough
    # lick_vel_angular:longblob     # median tongue angular velocity during the lick duration, from peak to trough
    # lick_yaw:longblob     # tongue yaw at the peak of the lick
    # lick_yaw_relative:longblob     # tongue yaw at the peak of the lick, relative to the left lick port
    # lick_yaw_avg:longblob     # median tongue yaw  during the lick duration, from peak to trough
    # lick_yaw_avg_relative:longblob     # median tongue yaw  during the lick duration, from peak to trough, relative to the left lick port
    # lick_horizoffset:longblob    # tongue horizontal displacement at the peak of the lick, relative to midline. Positive values - right port, negative values - left port. Normalized to the distance between ports.
    # lick_horizoffset_relative:longblob    # tongue horizontal displacement at the peak of the lick, relative to the left lick port
    # lick_rt_electric:longblob     # rt based on electric lick port
    # lick_rt_video_onset:longblob     # rt based on video trough
    

            
    def make(self, key):
        print(key)
        #%%
        #key = {'subject_id': 433018, 'session': 12, 'trial': 388}
        tracking_fs = (tracking.TrackingDevice & key).fetch1('sampling_rate')
        MinPeakDistance = 0.1/(1/float(tracking_fs)) # minimum frame numbers
        MinPeakProminence = 2 # minimum pixels
#%%        
        class k:
            pass
        
        k.subject_id = experiment.Session.fetch("subject_id")
        k.session = experiment.Session.fetch("session")  
        
        # Define x origin
        
        class trackingpoints:
            pass        
        
        # jaw
        tracking_data_jaw = (tracking.Tracking.JawTracking & key).fetch()
        jaw = trackingpoints()
        jaw.x = tracking_data_jaw['jaw_x']
        jaw.y = tracking_data_jaw['jaw_y']
        
        # nose
        tracking_data_nose = (tracking.Tracking.NoseTracking & key).fetch()
        nose = trackingpoints()
        nose.x = tracking_data_nose['nose_x']-jaw.x
        nose.y = tracking_data_nose['nose_y']
        nose.p = tracking_data_nose['nose_likelihood']
        
        #x_origin = (nose.y - jaw.y)*0.5 # In Arseny's code, he multiplied a factor 0.5, reason unknown
        
        nose.y = nose.y - jaw.y
        
        # load other tracking points
        #Tongue tip
        tracking_data_tongue_tip = (tracking.Tracking.TongueTracking & key).fetch()
        tongue_tip = trackingpoints()
        tongue_tip.x = tracking_data_tongue_tip['tongue_mid_x']-jaw.x # removing jaw position
        tongue_tip.y = tracking_data_tongue_tip['tongue_mid_y']-jaw.y # removing jaw position
        tongue_tip.p = tracking_data_tongue_tip['tongue_mid_likelihood']
        tongue_tip.t = (tracking.VideoFiducialsTrial & key).fetch('fiducial_time')
        
        # Tongue left
        tracking_data_tongue_left = (tracking.Tracking.LeftTongueTracking & key).fetch()
        tongue_left = trackingpoints()
        tongue_left.x = tracking_data_tongue_left['tongue_left_x']
        tongue_left.y = tracking_data_tongue_left['tongue_left_y']
        tongue_left.p = tracking_data_tongue_left['tongue_left_likelihood']
        
        # Tongue Right
        tracking_data_tongue_right = (tracking.Tracking.RightTongueTracking & key).fetch()
        tongue_right = trackingpoints()
        tongue_right.x = tracking_data_tongue_right['tongue_right_x']
        tongue_right.y = tracking_data_tongue_right['tongue_right_y']
        tongue_right.p = tracking_data_tongue_right['tongue_right_likelihood']
        
        
        # lickport right
        tracking_data_port_right = (tracking.Tracking.RightLickPortTracking & key).fetch()
        port_right = trackingpoints()
        port_right.x = tracking_data_port_right['right_lickport_x']-jaw.x
        port_right.y = tracking_data_port_right['right_lickport_y']-jaw.y
        port_right.p = tracking_data_port_right['right_lickport_likelihood']
        
        # left lickport
        tracking_data_port_left = (tracking.Tracking.LeftLickPortTracking & key).fetch()
        port_left = trackingpoints()
        port_left.x = tracking_data_port_left['left_lickport_x']-jaw.x
        port_left.y = tracking_data_port_left['left_lickport_y']-jaw.y
        port_left.p = tracking_data_port_left['left_lickport_likelihood']
        
        ## Number of trials, filenames etc.
        #session_date = (experiment.Session & key).fetch('session_date')
        numFrames = len(tongue_tip.x)
        trials = (experiment.SessionTrial & key).fetch('trial',order_by="trial") 
        time_go = (experiment.BehaviorTrial * experiment.TrialEvent & experiment.TrialEventType & key & {'trial_event_type = "go"'}).fetch('trial_event_time',order_by="trial")
        #session = (experiment.SessionTrial & key).fetch('session',order_by="session")
        print(trials)
        #print(time_go)
        #time_go = time_go[int(trials)]       
        
        p_threshold = 0.9
        #camera_bottomview_pixels_to_mm = (Camera_pixels & key).fetch('camera_pixels_value')
        numTrials = tongue_tip.x.shape
        tip = []
        left = []
        right = []
        counter = 0
        peak_to_trough = []
        peak_at_75 = []
        
        class insert_key:
            pass
        insert_key.subject_id = pd.DataFrame([])
        insert_key.session = pd.DataFrame([])
        insert_key.trial = pd.DataFrame([])        
        
        for x in range(numTrials[0]):
            tip_a = [i for (i,j) in enumerate(tongue_tip.p[x]) if j>p_threshold]
            tip.append(tip_a)
            left_a = [i for (i,j) in enumerate(tongue_left.p[x]) if j>p_threshold]
            left.append(left_a)
            right_a = [i for (i,j) in enumerate(tongue_right.p[x]) if j>p_threshold]
            right.append(right_a)
        
            idx_P = list(set().union(tip[x],left[x],right[x]))   # finding index that either tongue tip, tongue_right or tongue_left has p>threshold probability     
            idx_P.sort()
            #print(idx_P)
            
            non_idx_P = [i for i in range(len(tongue_tip.p[x])) if i not in idx_P]
            X_all = tongue_tip.x[x] *1
            Y_all = tongue_tip.y[x] *1
            t_all = tongue_tip.t[x] *1
            
            X_all.setflags(write=1)
            Y_all.setflags(write=1)
            t_all.setflags(write=1)
        
            X_all[non_idx_P] = -1
            Y_all[non_idx_P] = -1
        
            # Taking only frames in which the tongue was seen
            t = tongue_tip.t[x][idx_P]
            t = np.array(t)
            X = X_all[idx_P]
            Y = Y_all[idx_P]
            
            
            if len(X) < 1:
                #insert_key.subject_id[x] = k.subject_id
                #insert_key.session[x] = k.session
                #insert_key.trial[x] = trials[x]
                print('X is < 1!')
                continue
            X = movmean(X,5)
            Y = movmean(Y,5)
            
            if len(t) <20:
                print('len(t) <20')
                continue
            
            counter = counter+1
            
            # Extracting lick port positions
            idx_P4 = [i for (i,j) in enumerate(port_left.p[x]) if j>p_threshold] # left lickport
            L_Port_x = st.median(port_left.x[x][idx_P4])
            L_Port_y = st.median(port_left.y[x][idx_P4])
            idx_P5 = [i for (i,j) in enumerate(port_right.p[x]) if j>p_threshold] # Right lickport
            R_Port_x = st.median(port_right.x[x][idx_P5])
            R_Port_y = st.median(port_right.y[x][idx_P5])
            
            # Extracting nose positions
            idx_P6 = [i for (i,j) in enumerate(nose.p[x]) if j>p_threshold] # left lickport
            if len(idx_P6)<=0:
                nose_x1 = st.median(nose.x[x])
                nose_y1 = st.median(nose.y[x])
            else:
                nose_x1 = st.median(nose.x[x][idx_P6])
                nose_y1 = st.median(nose.y[x][idx_P6])
            
            
            
            # Extracting lick times based on electric lick port
            k_trial = trials[x]
            time_lick_Elect = ((experiment.BehaviorTrial * experiment.ActionEvent * experiment.Session &{'session':k.session[-1]} & {'trial':k_trial})).fetch('action_event_time')-time_go[x]
            #trial_instruction = (experiment.BehaviorTrial * experiment.Session).fetch('trial_instruction')
            #trial_outcome = (experiment.BehaviorTrial  * experiment.Session ).fetch('outcome')
            #early_lick = (experiment.BehaviorTrial  * experiment.Session).fetch('early_lick')
            #print(time_lick_Elect)
            
            # finding tongue angle, amplitude and vel
            X = np.array(X)
            Y = np.array(Y)
            [tongue_yaw, tongue_amplitude] = cart2pol(X,Y)
            tongue_vel_linear1 = np.diff(tongue_amplitude)
            tongue_vel_linear = [0]
            tongue_vel_linear.extend(tongue_vel_linear1)
            #np.transpose(tongue_vel_linear)
            tongue_vel_linear = movmean(tongue_vel_linear,5)
            
            tongue_vel_angular1 = np.diff(Y)
            tongue_vel_angular = [0]
            tongue_vel_angular.extend(tongue_vel_angular1)
            #np.transpose(tongue_vel_angular)
            tongue_vel_angular = movmean(tongue_vel_angular,5)
            
            # Finding peaks and troughs
            pks_idx,_ = scipy.signal.find_peaks(Y_all,distance = MinPeakDistance,prominence = MinPeakProminence)
            if len(Y) > 5000:
                #insert_key.subject_id[x] = k.subject_id
                #insert_key.session[x] = k.session
                #insert_key.trial[x] = trials[x]
                print('length is over 5000!')
                continue
                
            Y_trough = -1*(Y_all - max(Y_all))
            if sum(t_all[pks_idx]>0)>0:
                Y_trough[[i for (i,j) in enumerate(t_all) if j <=0]] =0
            
            trough_idx,_ = scipy.signal.find_peaks(Y_trough,distance = MinPeakDistance,prominence = MinPeakProminence)
            #print(trough_idx)
            #print(pks_idx)
            corresponding_trough_idx=[]
            for xx in range(len(pks_idx)): 
                temp_idx1 = np.argwhere(trough_idx < pks_idx[xx])
                
                #temp_idx = [i for (i,j) in enumerate(trough_idx) if j-pks_idx[xx] <0]
                #if temp_idx:
                    #temp_idx = temp_idx[-1]
                if temp_idx1.size<=0 and xx ==0:
                    trough_idx = np.insert(trough_idx,0,1)
                    corresponding_trough_idx = pd.DataFrame(np.array([0]))
                    #corresponding_trough_idx.insert(xx,1)
                else:
                    if isinstance(corresponding_trough_idx,list):
                        corresponding_trough_idx = pd.DataFrame([])                    
                    temp_idx = temp_idx1[-1]
                    corresponding_trough_idx = pd.DataFrame.append(corresponding_trough_idx,pd.DataFrame(np.array([trough_idx[temp_idx][0]])))
                    #corresponding_trough_idx.insert(xx,trough_idx[temp_idx])
                    
            trough_idx = corresponding_trough_idx.to_numpy() #take only troughs that match to peaks
            trough_idx = trough_idx.flatten()            
            
            #print(pks_idx)
            #print(trough_idx)
            
            a = np.array(pks_idx[1:]) - np.array(pks_idx[:-1])
            b = np.array(pks_idx[1:]) - np.array(trough_idx[1:])
            c = a-b
            d = np.where(c>0)[0]
            e = pks_idx[d+1]
            f = np.insert(e,0,pks_idx[0])
            pks_idx = np.array(f)
                        
            trough_idx1 = trough_idx[d+1]
            g = np.insert(trough_idx1,0,trough_idx[0])
            trough_idx = np.array(g)+1            
            add_idx = []
            
            for xxx in range(len(trough_idx)):
                h = np.where((Y_all[trough_idx[xxx]-1:pks_idx[xxx]+1])>0)[0]
                
                #d = np.array([i for (i,j) in enumerate(X_all[trough_idx[xxx]:pks_idx[xxx]]) if j > -0])-1
                if len(h)>0:
                    if h[0] ==0:
                        h = h[0]
                    else:
                        h = h[0]-1
                    add_idx.insert(xxx,h)
                else:
                    add_idx.insert(xxx,0)
            trough_idx = np.array(trough_idx) + np.array(add_idx)
            
                        
            #print(pks_idx)
            #print(trough_idx)
            #print('!')
                        
            pks_idx = [i for i, j in enumerate(np.isin(t,[t_all[y] for y in pks_idx])) if j]
            
            
            trough_idx = [i for i, j in enumerate(np.isin(t,[t_all[y] for y in trough_idx])) if j]
            
            #pks_idx = [j for (i,j) in enumerate(pks_idx) if j>=0] # remove potential -1 index; not present in original Arseny matlab code
            #trough_idx = [j for (i,j) in enumerate(trough_idx) if j>=0] # remove potential -1 index ; not present in original Arseny matlab code
            #print(pks_idx) 
            #print(trough_idx)
            
            bad_1 = np.intersect1d(pks_idx,trough_idx)
            if len(bad_1)>0:
            
                for i in range(len(bad_1)):
                    bad_2 = np.where(bad_1[i]==pks_idx)
                    if i == 0:
                        bad = bad_2
                    else:
                        bad = np.append(bad,bad_2)
                        
                for ii in reversed(bad):
                    pks_idx = np.delete(pks_idx, ii, 0)
                    
                    
                for i in range(len(bad_1)):
                    bad_3 = np.where(bad_1[i]==trough_idx)
                    if i == 0:
                        bad_4 = bad_3
                    else:
                        bad_4 = np.append(bad_4,bad_3)
                        
                for ii in reversed(bad_4):
                    trough_idx = np.delete(trough_idx, ii, 0) 
                
            
            if len(pks_idx)<=0:
                #insert_key.subject_id[x] = k.subject_id
                #insert_key.session[x] = k.session
                #insert_key.trial[x] = trials[x]
                continue       
#%%                
            # defining peak as 75% from trough to peak            
                
            try:
                if len(pks_idx) == len(trough_idx):
                    peak_to_trough = np.array([tongue_amplitude[y] for y in pks_idx]) - np.array([tongue_amplitude[y] for y in trough_idx])
                else:
                    corresponding_trough_idx1=[]
                    for xxx in range(len(pks_idx)):
                        temp_idx2 = np.argwhere(np.array(trough_idx) < pks_idx[xxx])               
                        if temp_idx2.size<=0 and xxx ==0:
                            trough_idx = np.insert(trough_idx,0,1)
                            corresponding_trough_idx1 = pd.DataFrame(np.array([0]))
                            #corresponding_trough_idx.insert(xx,1)
                        else:
                            if isinstance(corresponding_trough_idx1,list):
                                corresponding_trough_idx1 = pd.DataFrame([])                    
                            temp_idx3 = temp_idx2[-1]                        
                            trough_idx_array = np.array(trough_idx)
    
                            corresponding_trough_idx1 = pd.DataFrame.append(corresponding_trough_idx1,pd.DataFrame(np.array([trough_idx_array[temp_idx3][0]])))
                    trough_idx = corresponding_trough_idx1.to_numpy() #take only troughs that match to peaks
                    trough_idx = trough_idx.flatten()
                    
                    bad_5 = np.intersect1d(pks_idx,trough_idx)
                    if len(bad_5)>0:
                    
                        for i in range(len(bad_5)):
                            bad_6 = np.where(bad_5[i]==pks_idx)
                            if i == 0:
                                badd = bad_6
                            else:
                                badd = np.append(badd,bad_6)
                                
                        for ii in reversed(badd):
                            pks_idx = np.delete(pks_idx, ii, 0)
                            
                            
                        for i in range(len(bad_5)):
                            bad_7 = np.where(bad_5[i]==trough_idx)
                            if i == 0:
                                bad_8 = bad_7
                            else:
                                bad_8 = np.append(bad_8,bad_7)
                                
                        for ii in reversed(bad_8):
                            trough_idx = np.delete(trough_idx, ii, 0)
                            
                    peak_to_trough = peak_to_trough = np.array([tongue_amplitude[y] for y in pks_idx]) - np.array([tongue_amplitude[y] for y in trough_idx])
                    
                        
                        #corresponding_trough_idx.insert(xx,trough_idx[temp_idx])
                    
    #            for i in range(len(pks_idx)):
    #                if pks_idx[i]!=[]:
    #                    peak_to_trough1.insert(i,np.array(tongue_amplitude[i]))
    #                else:
    #                    peak_to_trough1.insert(i,[])           
                
            except:
                continue
            peak_at_75 = np.array([peak_to_trough[i]*0.75 for i in range(len(peak_to_trough))]) + np.array([tongue_amplitude[i] for i in trough_idx])   
            pks75_idx = []
            
            for i_p in range(len(pks_idx)):                
                current_idx = list(range(trough_idx[i_p],pks_idx[i_p]+1))
                temp_temp = abs(np.array([tongue_amplitude[i] for i in current_idx]) - np.array(peak_at_75[i_p]))
            
                temp_idx = np.argmin(temp_temp)
                temp_2 = trough_idx[i_p]
                pks75_idx.insert(i_p,trough_idx[i_p]+temp_2-1)
           
            # ensuring the trough is belonging to the same lick as the peak
            for i_p in range(len(pks_idx)):
                current_idx = list(range(trough_idx[i_p],pks_idx[i_p]+1))
                new = np.array([t[i] for i in current_idx])
                new1 = np.diff(np.diff(new))
                lickbout_start_idx1 = np.where(new1 > (1/294))[0]
                try:
                    lickbout_start_idx = lickbout_start_idx1[0]
                except:
                    lickbout_start_idx = 1
                trough_idx[i_p] = current_idx[0]+lickbout_start_idx -1
                
            ## angle, amplitude and reaction times
            offset = (L_Port_y+R_Port_y)/2 #relative to the mid-distance between the lickports
            
            #find lick angle at peak
            #----------------------------------------------------------
            [lick_yaw, lick_amplitude] = cart2pol(X[pks_idx],Y[pks_idx])
            ## % reversing the angle because  the video recorded is flipped - left side appears on the right???
            #print(trough_idx)
            #print(pks_idx)
            # relative to left lickport
            [lick_yaw_relative,notIMportant] = cart2pol(X[pks_idx]-offset,Y[pks_idx] )
            
            #%find the average lick angle and velocity during the entire outbound lick (not only at the peak of the lick)
            #%----------------------------------------------------------
            yaw_avg=np.empty(len(pks_idx))
            lick_vel_linear=np.empty(len(pks_idx))
            lick_vel_angular=np.empty(len(pks_idx))
            for ll in range(len(pks_idx)):
                idx = list(range(trough_idx[ll],pks_idx[ll]))
                [temp_theta_lick,notimportant] = cart2pol(X[idx],Y[idx])                
                yaw_avg[ll] = np.nanmedian(temp_theta_lick)                
                lick_vel_linear[ll]=np.nanmedian(tongue_vel_linear[idx])
                lick_vel_angular[ll]=np.nanmedian(tongue_vel_angular[idx])
                
            
            # relative to left lickport
            yaw_avg_relative = np.empty(len(pks_idx))
            for ll in range(len(pks_idx)):
                idx1 = list(range(trough_idx[ll],pks_idx[ll]))
                [temp_theta_lick,notimportant] = cart2pol(X[idx1]- offset,Y[idx1]) 
                yaw_avg_relative[ll] = np.nanmedian(temp_theta_lick) 
                
            #% horizontal displacement
                #%----------------------------------------------------------    
            #% relative to ML midline    
            lick_horizoffset = X[pks_idx] #-eq_midline(midline_slope, X(pks_idx)))
            
            #% relative to the middle of the lickports
            lick_horizoffset_relative = X[pks_idx]- offset
            trough_idx_interger = np.int8(trough_idx) # to make the index as interger
            idx_noearly_licks = np.where(t[trough_idx_interger]>0)[0]
            
            pks_idx = np.asarray(pks_idx)
            # Reaction time
            #%------------------------------------------------------------------
            
            RT_VideoPeak = t[pks_idx[idx_noearly_licks]]                  
            RT_VideoOnset = t[trough_idx_interger[idx_noearly_licks]]
            RT_Electric1 = np.where(time_lick_Elect>0)[0]
            RT_Electric = time_lick_Elect[RT_Electric1]
            RT_Electric = RT_Electric.astype(float)
            
            
            # parsed by licks
            lick_peak_y_1 = np.transpose([Y[pks_idx[idx_noearly_licks]]])#*camera_bottomview_pixels_to_mm
            lick_peak_x_1 = np.transpose([X[pks_idx[idx_noearly_licks]]])#*camera_bottomview_pixels_to_mm
            lick_amplitude_1 = np.transpose(lick_amplitude[idx_noearly_licks])#*camera_bottomview_pixels_to_mm
            lick_vel_linear_1 = np.transpose(lick_vel_linear[idx_noearly_licks])#*camera_bottomview_pixels_to_mm
            lick_vel_angular_1 = np.transpose(lick_vel_angular[idx_noearly_licks])*(-1)
            lick_yaw_1 = np.transpose(lick_yaw[idx_noearly_licks])*(-1)
            lick_yaw_relative_1 = np.transpose(lick_yaw_relative[idx_noearly_licks])*(-1)
            lick_yaw_avg_1 = np.transpose(yaw_avg[idx_noearly_licks])*(-1)
            lick_yaw_avg_relative_1 = np.transpose(yaw_avg_relative[idx_noearly_licks])*(-1)
            lick_horizoffset_1 = np.transpose(lick_horizoffset[idx_noearly_licks])*(-1)#*camera_bottomview_pixels_to_mm
            lick_horizoffset_relative_1 = np.transpose(lick_horizoffset_relative[idx_noearly_licks])*(-1)#*camera_bottomview_pixels_to_mm
            lick_rt_video_onset_1 = RT_VideoOnset
            lick_rt_electric_1 = RT_Electric
            lick_rt_video_peak_1 = np.transpose(RT_VideoPeak)
            #print(key)
            #%%
            self.insert1(
                    dict(key, lick_peak_y = lick_peak_y_1, lick_peak_x = lick_peak_x_1,lick_amplitude = lick_amplitude_1
                         ,lick_vel_linear = lick_vel_linear_1, lick_vel_angular = lick_vel_angular_1, lick_yaw = lick_yaw_1,
                         lick_yaw_relative = lick_yaw_relative_1,lick_yaw_avg = lick_yaw_avg_1, lick_yaw_avg_relative = lick_yaw_avg_relative_1,
                         lick_horizoffset = lick_horizoffset_1, lick_horizoffset_relative = lick_horizoffset_relative_1, 
                         lick_rt_video_onset = lick_rt_video_onset_1, lick_rt_electric = lick_rt_electric_1,lick_rt_video_peak = lick_rt_video_peak_1 )
                    )
            
            #key['x'] = X
            #key['y'] = Y
            #key['t'] = t
            #key['pks_idx'] = pks_idx
            #key['trough_idx'] = trough_idx
            #key['lp_x'] = L_Port_x          
            #key['lp_y'] = L_Port_y 
            #key['rp_x'] = R_Port_x          
            #key['rp_y'] = R_Port_y
            #key['nose_x'] = nose_x1
            #key['nose_y'] = nose_y1
            VideoTongueTrial.Raw_data.insert1(
                    dict(key, x = X, y = Y, t = t, pks_idx = pks_idx,
                         trough_idx = trough_idx, lp_x = L_Port_x, lp_y = L_Port_y,
                         rp_x = R_Port_x, rp_y = R_Port_y, nose_x = nose_x1, nose_y = nose_y1)
                    )
            
        
        
        
        
        
        


