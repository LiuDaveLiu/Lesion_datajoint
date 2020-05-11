import pandas as pd
import numpy as np
from datetime import datetime
#% connect to server
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment,tracking, bottom_tongue
#import ray

# from  MAP v2
import logging
log = logging.getLogger(__name__)
from collections import defaultdict
#import re
import pathlib
from datetime import date
import os
import scipy.io as spio
from collections import namedtuple
import math

#%%
def load_tracking(trkpath):
    log.debug('load_tracking() {}'.format(trkpath))
    '''
    load actual tracking data.

    example format:

    scorer,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000
    bodyparts,jaw,jaw,jaw, nose,nose,nose,tongue_mid,tongue_mid,tongue_mid,tongue_left,tongue_left,tongue_left,tongue_right,tongue_right,tongue_right,left_lickport,left_lickport,left_lickport,right_lickport,right_lickport,right_lickport
    coords,x,y,likelihood,x,y,likelihood,x,y,likelihood
    0,418.48327827453613,257.231650352478,1.0,426.47182297706604,263.82502603530884,1.796432684386673e-06,226.12365770339966,395.8081398010254,1.0

    results are of the form:

      {'feature': {'attr': [val, ...]}}

    where feature is e.g. 'nose', 'attr' is e.g. 'x'.

    the special 'feature'/'attr' pair "samples"/"ts" is used to store
    the first column/sample timestamp for each row in the input file.
    '''
    res = defaultdict(lambda: defaultdict(list))

    with open(trkpath, 'r') as f:
        f.readline()  # discard 1st line
        parts, fields = f.readline(), f.readline()
        parts = parts.rstrip().split(',')
        fields = fields.rstrip().split(',')

        for l in f:
            lv = l.rstrip().split(',')
            for i, v in enumerate(lv):
                v = float(v)
                if i == 0:
                    res['samples']['ts'].append(v)
                else:
                    res[parts[i]][fields[i]].append(v)

    return res       



#%% 
# if data exist in the google sheet, then upload data 
def populatebehavior(paralel = True,drop_last_session_for_mice_in_training = True):
    print('adding behavior experiments')
    if paralel:
        #ray.init()
        result_ids = []
        #%%
        IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
        df_surgery = pd.read_csv(dj.config['locations.metadata_behavior']+'Surgery.csv')
        for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects and removing last session     
            if subject_now in df_surgery['ID'].values and drop_last_session_for_mice_in_training == True and df_surgery['status'][df_surgery['ID']==subject_now].values[0] != 'sacrificed': # the last session is deleted only if the animal is still in training..
                print(df_surgery['status'][df_surgery['ID']==subject_now].values[0])
                if len((experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"').fetch('session')) > 0:
                    sessiontodel = np.max((experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"').fetch('session'))
                    session_todel = experiment.Session() & 'subject_id = "' + str(subject_id_now)+'"' & 'session = ' + str(sessiontodel)
                    dj.config['safemode'] = False
                    print('deleting last session of ' + subject_now)
                    session_todel.delete()
                    dj.config['safemode'] = True   
                    #%%
        for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects                       
            dict_now = dict()
            dict_now[subject_now] = subject_id_now
            result_ids.append(populatebehavior_core(dict_now))
            
        #ray.get(result_ids)
        #ray.shutdown()
    else:
        arguments = {'display_progress' : True}
        populatebehavior_core(arguments)
        
#@ray.remote
def populatebehavior_core(IDs = None):
    if IDs:
        print('subject started:')
        print(IDs.keys())
        print(IDs.values())
        
    rigpath_1 = 'E:/Projects/Ablation/datajoint/Behavior'
    
    #df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
    if IDs == None:
        IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}   

    for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects
        print('subject: ',subject_now)
    # =============================================================================
    #         if drop_last_session_for_mice_in_training:
    #             delete_last_session_before_upload = True
    #         else:
    #             delete_last_session_before_upload = False
    #         #df_wr = online_notebook.fetch_water_restriction_metadata(subject_now)
    # =============================================================================
        try:
            df_wr = pd.read_csv(dj.config['locations.metadata_behavior']+subject_now+'.csv')
        except:
            print(subject_now + ' has no metadata available')
            df_wr = pd.DataFrame()
        for df_wr_row in df_wr.iterrows():
            date_now = df_wr_row[1].Date.replace('-','')
            print('subject: ',subject_now,'  date: ',date_now)
            session_date = datetime(int(date_now[0:4]),int(date_now[4:6]),int(date_now[6:8]))
            if len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date > "'+str(session_date)+'"') != 0: # if it is not the last
                print('session already imported, skipping: ' + str(session_date))
                dotheupload = False
            elif len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"') != 0: # if it is the last
                dotheupload = False
            else: # reuploading new session that is not present on the server
                dotheupload = True
                
            # if dotheupload is True, meaning that there are new mat file hasn't been uploaded
            # => needs to find which mat file hasn't been uploaded
            
            if dotheupload:
                found = set()
                rigpath_2 = subject_now
                rigpath_3 = rigpath_1 + '/' + rigpath_2
                rigpath = pathlib.Path(rigpath_3)
                
                def buildrec(rigpath, root, f):
                    try:
                        fullpath = pathlib.Path(root, f)
                        subpath = fullpath.relative_to(rigpath)
                        fsplit = subpath.stem.split('_')
                        h2o = fsplit[0]
                        ymd = fsplit[-2:-1][0]
                        animal = IDs[h2o]
                        if ymd == date_now:
                            return {
                                    'subject_id': animal,
                                    'session_date': date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])),
                                    'rig_data_path': rigpath.as_posix(),
                                    'subpath': subpath.as_posix(),
                                    }
                    except:
                        pass
                for root, dirs, files in os.walk(rigpath):
                    for f in files:
                        r = buildrec(rigpath, root, f)
                        if r:
                            found.add(r['subpath'])
                            file = r
                
                # now start insert data
            
                path = pathlib.Path(file['rig_data_path'], file['subpath'])
                mat = spio.loadmat(path, squeeze_me=True)
                SessionData = mat['SessionData'].flatten()
                            
                # session record key
                skey = {}
                skey['subject_id'] = file['subject_id']
                skey['session_date'] = file['session_date']
                skey['username'] = 'NT'
                #skey['rig'] = key['rig']
            
                trial = namedtuple(  # simple structure to track per-trial vars
                        'trial', ('ttype', 'settings', 'state_times',
                                  'state_names', 'state_data', 'event_data',
                                  'event_times', 'trial_start'))
            
                # parse session datetime
                session_datetime_str = str('').join((str(SessionData['Info'][0]['SessionDate']),' ', str(SessionData['Info'][0]['SessionStartTime_UTC'])))
                session_datetime = datetime.strptime(session_datetime_str, '%d-%b-%Y %H:%M:%S')
            
                AllTrialTypes = SessionData['TrialTypes'][0]
                AllTrialSettings = SessionData['TrialSettings'][0]
                AllTrialStarts = SessionData['TrialStartTimestamp'][0]
                AllTrialStarts = AllTrialStarts - AllTrialStarts[0]
            
                RawData = SessionData['RawData'][0].flatten()
                AllStateNames = RawData['OriginalStateNamesByNumber'][0]
                AllStateData = RawData['OriginalStateData'][0]
                AllEventData = RawData['OriginalEventData'][0]
                AllStateTimestamps = RawData['OriginalStateTimestamps'][0]
                AllEventTimestamps = RawData['OriginalEventTimestamps'][0]
            
                trials = list(zip(AllTrialTypes, AllTrialSettings,
                                  AllStateTimestamps, AllStateNames, AllStateData,
                                  AllEventData, AllEventTimestamps, AllTrialStarts))
                
                if not trials:
                    log.warning('skipping date {d}, no valid files'.format(d=date))
                    return    
                #
                # Trial data seems valid; synthesize session id & add session record
                # XXX: note - later breaks can result in Sessions without valid trials
                #
            
                assert skey['session_date'] == session_datetime.date()
                
                skey['session_date'] = session_datetime.date()
                #skey['session_time'] = session_datetime.time()
            
                if len(experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"' & 'session_date = "'+str(file['session_date'])+'"') == 0:
                    if len(experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"') == 0:
                        skey['session'] = 1
                    else:
                        skey['session'] = len((experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"').fetch()['session']) + 1
            
                #
                # Actually load the per-trial data
                #
                log.info('BehaviorIngest.make(): trial parsing phase')

                # lists of various records for batch-insert
                rows = {k: list() for k in ('trial', 'behavior_trial', 'trial_note',
                                        'trial_event', 'corrected_trial_event',
                                        'action_event')} #, 'photostim',
                                    #'photostim_location', 'photostim_trial',
                                    #'photostim_trial_event')}

                i = 0  # trial numbering starts at 1
                for t in trials:
                    t = trial(*t)  # convert list of items to a 'trial' structure
                    i += 1  # increment trial counter

                    log.debug('BehaviorIngest.make(): parsing trial {i}'.format(i=i))

                    states = {k: (v+1) for v, k in enumerate(t.state_names)}
                    required_states = ('PreSamplePeriod', 'SamplePeriod',
                                       'DelayPeriod', 'ResponseCue', 'StopLicking',
                                       'TrialEnd')
                
                    missing = list(k for k in required_states if k not in states)
                    if len(missing) and missing =='PreSamplePeriod':
                        log.warning('skipping trial {i}; missing {m}'.format(i=i, m=missing))
                        continue

                    gui = t.settings['GUI'].flatten()
                    if len(experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"' & 'session_date = "'+str(file['session_date'])+'"') == 0:
                        if len(experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"') == 0:
                            skey['session'] = 1
                        else:
                            skey['session'] = len((experiment.Session() & 'subject_id = "'+str(file['subject_id'])+'"').fetch()['session']) + 1
                
                    #
                    # Top-level 'Trial' record
                    #
                    protocol_type = gui['ProtocolType'][0]
                    tkey = dict(skey)
                    has_presample = 1
                    try:
                        startindex = np.where(t.state_data == states['PreSamplePeriod'])[0]
                        has_presample = 1
                    except:
                        startindex = np.where(t.state_data == states['SamplePeriod'])[0]
                        has_presample = 0
                
                    # should be only end of 1st StopLicking;
                    # rest of data is irrelevant w/r/t separately ingested ephys
                    endindex = np.where(t.state_data == states['StopLicking'])[0]
                    log.debug('states\n' + str(states))
                    log.debug('state_data\n' + str(t.state_data))
                    log.debug('startindex\n' + str(startindex))
                    log.debug('endindex\n' + str(endindex))
                
                    if not(len(startindex) and len(endindex)):
                        log.warning('skipping {}: start/end mismatch: {}/{}'.format(i, str(startindex), str(endindex)))
                        continue
                    
                    try:
                        tkey['trial'] = i
                        tkey['trial_uid'] = i
                        tkey['trial_start_time'] = t.trial_start
                        tkey['trial_stop_time'] = t.trial_start + t.state_times[endindex][0]
                    except IndexError:
                        log.warning('skipping {}: IndexError: {}/{} -> {}'.format(i, str(startindex), str(endindex), str(t.state_times)))
                        continue
                    
                    log.debug('tkey' + str(tkey))
                    rows['trial'].append(tkey)
                
                    #
                    # Specific BehaviorTrial information for this trial
                    #                              
                    
                    bkey = dict(tkey)
                    bkey['task'] = 'audio delay'  # hard-coded here
                    bkey['task_protocol'] = 1     # hard-coded here
                
                    # determine trial instruction
                    trial_instruction = 'left'    # hard-coded here

                    if gui['Reversal'][0] == 1:
                        if t.ttype == 1:
                            trial_instruction = 'left'
                        elif t.ttype == 0:
                            trial_instruction = 'right'
                        elif t.ttype == 2:
                            trial_instruction = 'catch_right_autowater'
                        elif t.ttype == 3:
                            trial_instruction = 'catch_left_autowater'
                        elif t.ttype == 4:
                            trial_instruction = 'catch_right_noDelay'
                        elif t.ttype == 5:
                            trial_instruction = 'catch_left_noDelay'    
                    elif gui['Reversal'][0] == 2:
                        if t.ttype == 1:
                            trial_instruction = 'right'
                        elif t.ttype == 0:
                            trial_instruction = 'left'
                        elif t.ttype == 2:
                            trial_instruction = 'catch_left_autowater'
                        elif t.ttype == 3:
                            trial_instruction = 'catch_right_autowater'
                        elif t.ttype == 4:
                            trial_instruction = 'catch_left_noDelay'
                        elif t.ttype == 5:
                            trial_instruction = 'catch_right_noDelay'
                
                    bkey['trial_instruction'] = trial_instruction
                    # determine early lick
                    early_lick = 'no early'
                    
                    if (protocol_type >= 5 and 'EarlyLickDelay' in states and np.any(t.state_data == states['EarlyLickDelay'])):
                        early_lick = 'early'
                    if (protocol_type >= 5 and ('EarlyLickSample' in states and np.any(t.state_data == states['EarlyLickSample']))):
                        early_lick = 'early'
                        
                    bkey['early_lick'] = early_lick
                
                    # determine outcome
                    outcome = 'ignore'
                    if ('Reward' in states and np.any(t.state_data == states['Reward'])):
                        outcome = 'hit'
                    elif ('TimeOut' in states and np.any(t.state_data == states['TimeOut'])):
                        outcome = 'miss'
                    elif ('NoResponse' in states and np.any(t.state_data == states['NoResponse'])):
                        outcome = 'ignore'    
                    bkey['outcome'] = outcome
                    rows['behavior_trial'].append(bkey)
                    
                    #
                    # Add 'protocol' note
                    #
                    nkey = dict(tkey)
                    nkey['trial_note_type'] = 'protocol #'
                    nkey['trial_note'] = str(protocol_type)
                    rows['trial_note'].append(nkey)

                    #
                    # Add 'autolearn' note
                    #
                    nkey = dict(tkey)
                    nkey['trial_note_type'] = 'autolearn'
                    nkey['trial_note'] = str(gui['Autolearn'][0])
                    rows['trial_note'].append(nkey)
                    
                    #
                    # Add 'bitcode' note
                    #
                    if 'randomID' in gui.dtype.names:
                        nkey = dict(tkey)
                        nkey['trial_note_type'] = 'bitcode'
                        nkey['trial_note'] = str(gui['randomID'][0])
                        rows['trial_note'].append(nkey)
               
                
                    #
                    # Add presample event
                    #
                    sampleindex = np.where(t.state_data == states['SamplePeriod'])[0]
                    
                    if has_presample == 1:
                        log.debug('BehaviorIngest.make(): presample')
                        ekey = dict(tkey)                    
    
                        ekey['trial_event_id'] = len(rows['trial_event'])
                        ekey['trial_event_type'] = 'presample'
                        ekey['trial_event_time'] = t.state_times[startindex][0]
                        ekey['duration'] = (t.state_times[sampleindex[0]]- t.state_times[startindex])[0]
    
                        if math.isnan(ekey['duration']):
                            log.debug('BehaviorIngest.make(): fixing presample duration')
                            ekey['duration'] = 0.0  # FIXDUR: lookup from previous trial
    
                        rows['trial_event'].append(ekey)
                
                    #
                    # Add other 'sample' events
                    #
    
                    log.debug('BehaviorIngest.make(): sample events')
    
                    last_dur = None
    
                    for s in sampleindex:  # in protocol > 6 ~-> n>1
                        # todo: batch events
                        ekey = dict(tkey)
                        ekey['trial_event_id'] = len(rows['trial_event'])
                        ekey['trial_event_type'] = 'sample'
                        ekey['trial_event_time'] = t.state_times[s]
                        ekey['duration'] = gui['SamplePeriod'][0]
    
                        if math.isnan(ekey['duration']) and last_dur is None:
                            log.warning('... trial {} bad duration, no last_edur'.format(i, last_dur))
                            ekey['duration'] = 0.0  # FIXDUR: cross-trial check
                            rows['corrected_trial_event'].append(ekey)
    
                        elif math.isnan(ekey['duration']) and last_dur is not None:
                            log.warning('... trial {} duration using last_edur {}'.format(i, last_dur))
                            ekey['duration'] = last_dur
                            rows['corrected_trial_event'].append(ekey)
    
                        else:
                            last_dur = ekey['duration']  # only track 'good' values.
    
                        rows['trial_event'].append(ekey)
                
                    #
                    # Add 'delay' events
                    #
    
                    log.debug('BehaviorIngest.make(): delay events')
    
                    last_dur = None
                    delayindex = np.where(t.state_data == states['DelayPeriod'])[0]
    
                    for d in delayindex:  # protocol > 6 ~-> n>1
                        ekey = dict(tkey)
                        ekey['trial_event_id'] = len(rows['trial_event'])
                        ekey['trial_event_type'] = 'delay'
                        ekey['trial_event_time'] = t.state_times[d]
                        ekey['duration'] = gui['DelayPeriod'][0]
    
                        if math.isnan(ekey['duration']) and last_dur is None:
                            log.warning('... {} bad duration, no last_edur'.format(i, last_dur))
                            ekey['duration'] = 0.0  # FIXDUR: cross-trial check
                            rows['corrected_trial_event'].append(ekey)
    
                        elif math.isnan(ekey['duration']) and last_dur is not None:
                            log.warning('... {} duration using last_edur {}'.format(i, last_dur))
                            ekey['duration'] = last_dur
                            rows['corrected_trial_event'].append(ekey)
    
                        else:
                            last_dur = ekey['duration']  # only track 'good' values.
    
                        log.debug('delay event duration: {}'.format(ekey['duration']))
                        rows['trial_event'].append(ekey)
                         
                    #
                    # Add 'go' event
                    #
                    log.debug('BehaviorIngest.make(): go')
    
                    ekey = dict(tkey)
                    responseindex = np.where(t.state_data == states['ResponseCue'])[0]
    
                    ekey['trial_event_id'] = len(rows['trial_event'])
                    ekey['trial_event_type'] = 'go'
                    ekey['trial_event_time'] = t.state_times[responseindex][0]
                    ekey['duration'] = gui['AnswerPeriod'][0]
    
                    if math.isnan(ekey['duration']):
                        log.debug('BehaviorIngest.make(): fixing go duration')
                        ekey['duration'] = 0.0  # FIXDUR: lookup from previous trials
                        rows['corrected_trial_event'].append(ekey)
    
                    rows['trial_event'].append(ekey)
                
                    #
                    # Add 'trialEnd' events
                    #

                    log.debug('BehaviorIngest.make(): trialend events')

                    last_dur = None
                    trialendindex = np.where(t.state_data == states['TrialEnd'])[0]

                    ekey = dict(tkey)
                    ekey['trial_event_id'] = len(rows['trial_event'])
                    ekey['trial_event_type'] = 'trialend'
                    ekey['trial_event_time'] = t.state_times[trialendindex][0]
                    ekey['duration'] = 0.0
    
                    rows['trial_event'].append(ekey)
                    
                    #
                    # Add lick events
                    #
                       
                    lickleft = np.where(t.event_data == 69)[0]
                    log.debug('... lickleft: {r}'.format(r=str(lickleft)))
    
                    action_event_count = len(rows['action_event'])
                    if len(lickleft):
                        [rows['action_event'].append(
                                dict(tkey, action_event_id=action_event_count+idx,
                                     action_event_type='left lick',
                                     action_event_time=t.event_times[l]))
                        for idx, l in enumerate(lickleft)]
    
                    lickright = np.where(t.event_data == 71)[0]
                    log.debug('... lickright: {r}'.format(r=str(lickright)))
    
                    action_event_count = len(rows['action_event'])
                    if len(lickright):
                        [rows['action_event'].append(
                                dict(tkey, action_event_id=action_event_count+idx,
                                     action_event_type='right lick',
                                     action_event_time=t.event_times[r]))
                        for idx, r in enumerate(lickright)]
                    
                    # end of trial loop..    
        
                    # Session Insertion                     
                    log.info('BehaviorIngest.make(): adding session record')
                    skey['session_date'] = df_wr_row[1].Date 
                    skey['rig'] = 'Old Recording rig'
                    skey['username']  = 'NT'
                    experiment.Session().insert1(skey,skip_duplicates=True)

                # Behavior Insertion                

                log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
                experiment.SessionTrial().insert(
                        rows['trial'], ignore_extra_fields=True, allow_direct_insert=True)

                log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
                experiment.BehaviorTrial().insert(
                        rows['behavior_trial'], ignore_extra_fields=True,
                        allow_direct_insert=True)

                log.info('BehaviorIngest.make(): ... experiment.TrialNote')
                experiment.TrialNote().insert(
                        rows['trial_note'], ignore_extra_fields=True,
                        allow_direct_insert=True)

                log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
                experiment.TrialEvent().insert(
                        rows['trial_event'], ignore_extra_fields=True,
                        allow_direct_insert=True, skip_duplicates=True)
        
#        log.info('BehaviorIngest.make(): ... CorrectedTrialEvents')
#        BehaviorIngest().CorrectedTrialEvents().insert(
#            rows['corrected_trial_event'], ignore_extra_fields=True,
#            allow_direct_insert=True)

                log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
                experiment.ActionEvent().insert(
                        rows['action_event'], ignore_extra_fields=True,
                        allow_direct_insert=True)
                            
#%% for ingest tracking                
                if IDs:
                    print('subject started:')
                    print(IDs.keys())
                    print(IDs.values())
                    
                rigpath_tracking_1 = 'E:/Projects/Ablation/datajoint/video/'
                rigpath_tracking_2 = subject_now
                VideoDate1 = str(df_wr_row[1].VideoDate)
                if len(VideoDate1)==5:
                    VideoDate = '0'+ VideoDate1
                elif len(VideoDate1)==7:
                    VideoDate = '0'+ VideoDate1
                rigpath_tracking_3 = rigpath_tracking_1 + rigpath_tracking_2 + '/' + rigpath_tracking_2 + '_'+ VideoDate + '_front'
                
                rigpath_tracking = pathlib.Path(rigpath_tracking_3)
                
                #df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
                if IDs == None:
                    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}   
                
                h2o = subject_now
                session = df_wr_row[1].Date
                trials = (experiment.SessionTrial() & session).fetch('trial')
                
                log.info('got session: {} ({} trials)'.format(session, len(trials)))
                
                #sdate = session['session_date']
                #sdate_sml = date_now #"{}{:02d}{:02d}".format(sdate.year, sdate.month, sdate.day)

                paths = rigpath_tracking
                devices = tracking.TrackingDevice().fetch(as_dict=True)
                
                # paths like: <root>/<h2o>/YYYY-MM-DD/tracking
                tracking_files = []
                for d in (d for d in devices):
                    tdev = d['tracking_device']
                    tpos = d['tracking_position']
                    tdat = paths
                    log.info('checking {} for tracking data'.format(tdat))               
                    
                    
#                    if not tpath.exists():
#                        log.warning('tracking path {} n/a - skipping'.format(tpath))
#                        continue
#                    
#                    camtrial = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos)
#                    campath = tpath / camtrial
#                    
#                    log.info('trying camera position trial map: {}'.format(campath))
#                    
#                    if not campath.exists():
#                        log.info('skipping {} - does not exist'.format(campath))
#                        continue
#                    
#                    tmap = load_campath(campath)  # file:trial
#                    n_tmap = len(tmap)
#                    log.info('loading tracking data for {} trials'.format(n_tmap))

                    i = 0                    
                    VideoTrialNum = df_wr_row[1].VideoTrialNum
                    
                    #tpath = pathlib.Path(tdat, h2o, VideoDate, 'tracking')
                    ppp = list(range(0,VideoTrialNum))
                    for tt in reversed(range(VideoTrialNum)):  # load tracking for trial
                        
                        i += 1
#                        if i % 50 == 0:
#                            log.info('item {}/{}, trial #{} ({:.2f}%)'
#                                     .format(i, n_tmap, t, (i/n_tmap)*100))
#                        else:
#                            log.debug('item {}/{}, trial #{} ({:.2f}%)'
#                                      .format(i, n_tmap, t, (i/n_tmap)*100))
        
                        # ex: dl59_side_1-0000.csv / h2o_position_tn-0000.csv
                        tfile = '{}_{}_{}_{}-*.csv'.format(h2o, VideoDate ,tpos, tt)
                        tfull = list(tdat.glob(tfile))
                        if not tfull or len(tfull) > 1:
                            log.info('file mismatch: file: {} trial: ({})'.format(
                                tt, tfull))
                            continue
        
                        tfull = tfull[-1]
                        trk = load_tracking(tfull)
                        
        
                        recs = {}
                        
                        #key_source = experiment.Session - tracking.Tracking                        
                        rec_base = dict(trial=ppp[tt], tracking_device=tdev)
                        #print(rec_base)
                        for k in trk:
                            if k == 'samples':
                                recs['tracking'] = {
                                    'subject_id' : skey['subject_id'], 
                                    'session' : skey['session'],
                                    **rec_base,
                                    'tracking_samples': len(trk['samples']['ts']),
                                }
                                
                            else:
                                rec = dict(rec_base)
        
                                for attr in trk[k]:
                                    rec_key = '{}_{}'.format(k, attr)
                                    rec[rec_key] = np.array(trk[k][attr])
        
                                recs[k] = rec
                        
                        
                        tracking.Tracking.insert1(
                            recs['tracking'], allow_direct_insert=True)
                        
                        #if len(recs['nose']) > 3000:
                            #continue
                            
                        recs['nose'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['nose'],
                                }
                        
                        #print(recs['nose']['nose_x'])
                        if 'nose' in recs:
                            tracking.Tracking.NoseTracking.insert1(
                                recs['nose'], allow_direct_insert=True)
                            
                        recs['tongue_mid'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['tongue_mid'],
                                }
        
                        if 'tongue_mid' in recs:
                            tracking.Tracking.TongueTracking.insert1(
                                recs['tongue_mid'], allow_direct_insert=True)
                            
                        recs['jaw'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['jaw'],
                                }
        
                        if 'jaw' in recs:
                            tracking.Tracking.JawTracking.insert1(
                                recs['jaw'], allow_direct_insert=True)
                        
                        recs['tongue_left'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['tongue_left'],
                                }
        
                        if 'tongue_left' in recs:
                            tracking.Tracking.LeftTongueTracking.insert1(
                                recs['tongue_left'], allow_direct_insert=True)
                            
                        recs['tongue_right'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['tongue_right'],
                                }
        
                        if 'tongue_right' in recs:
                            tracking.Tracking.RightTongueTracking.insert1(
                                recs['tongue_right'], allow_direct_insert=True)
#                            fmap = {'paw_left_x': 'left_paw_x',  # remap field names
#                                    'paw_left_y': 'left_paw_y',
#                                    'paw_left_likelihood': 'left_paw_likelihood'}
        
#                            tracking.Tracking.LeftPawTracking.insert1({
#                                **{k: v for k, v in recs['paw_left'].items()
#                                   if k not in fmap},
#                                **{fmap[k]: v for k, v in recs['paw_left'].items()
#                                   if k in fmap}}, allow_direct_insert=True)
                        
                        recs['right_lickport'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['right_lickport'],
                                }
                        
                        if 'right_lickport' in recs:
                            tracking.Tracking.RightLickPortTracking.insert1(
                                recs['right_lickport'], allow_direct_insert=True)
#                            fmap = {'paw_right_x': 'right_paw_x',  # remap field names
#                                    'paw_right_y': 'right_paw_y',
#                                    'paw_right_likelihood': 'right_paw_likelihood'}
#        
#                            tracking.Tracking.RightPawTracking.insert1({
#                                **{k: v for k, v in recs['paw_right'].items()
#                                   if k not in fmap},
#                                **{fmap[k]: v for k, v in recs['paw_right'].items()
#                                   if k in fmap}}, allow_direct_insert=True)
                        
                        recs['left_lickport'] = {
                                'subject_id' : skey['subject_id'], 
                                'session' : skey['session'],
                                **recs['left_lickport'],
                                }
                        
                        if 'left_lickport' in recs:
                            tracking.Tracking.LeftLickPortTracking.insert1(
                                recs['left_lickport'], allow_direct_insert=True)
        
#                        tracking_files.append({**key, 'trial': tmap[t], 'tracking_device': tdev,
#                             'tracking_file': str(tfull.relative_to(tdat))})
#        
#                    log.info('... completed {}/{} items.'.format(i, n_tmap))
#        
#                self.insert1(key)
#                self.TrackingFile.insert(tracking_files)
#                   
                            
                        tracking.VideoFiducialsTrial.populate()
                        bottom_tongue.Camera_pixels.populate()
                        print('start!')               
                        bottom_tongue.VideoTongueTrial.populate()
                        sessiontrialdata={              'subject_id':skey['subject_id'],
                                                        'session':skey['session'],
                                                        'trial': tt
                                                        }
                        if len(bottom_tongue.VideoTongueTrial* experiment.Session & experiment.BehaviorTrial  & 'session_date = "'+str(file['session_date'])+'"' &{'trial':tt})==0:
                            print('trial couldn''t be exported, deleting trial')
                            print(tt)
                            dj.config['safemode'] = False
                            (experiment.SessionTrial()&sessiontrialdata).delete()
                            dj.config['safemode'] = True  
                        
                        
                log.info('... done.')
        

        


        
      