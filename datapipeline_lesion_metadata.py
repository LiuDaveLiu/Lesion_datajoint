import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
import json
import time as timer
#% connect to server
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment
#%%
def populatemetadata():
    #%% save metadata from google drive if necessairy
    lastmodify = online_notebook.fetch_lastmodify_time_animal_metadata()
    #print(lastmodify)
    with open(dj.config['locations.metadata_behavior']+'last_modify_time.json') as timedata:
        lastmodify_prev = json.loads(timedata.read())
    if lastmodify != lastmodify_prev:
        print('updating surgery and WR metadata from google drive')
        dj.config['locations.metadata_behavior']
        df_surgery = online_notebook.fetch_animal_metadata()
        df_surgery.to_csv(dj.config['locations.metadata_behavior']+'Surgery.csv')
        IDs = df_surgery['ID'].tolist()
        for ID in IDs:
            df_wr = online_notebook.fetch_water_restriction_metadata(ID)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(dj.config['locations.metadata_behavior']+ID+'.csv') 
        with open(dj.config['locations.metadata_behavior']+'last_modify_time.json', "w") as write_file:
            json.dump(lastmodify, write_file)
        print('surgery and WR metadata updated')
    
    lastmodify = online_notebook.fetch_lastmodify_time_lab_metadata()
    with open(dj.config['locations.metadata_lab']+'last_modify_time.json') as timedata:
        lastmodify_prev = json.loads(timedata.read())
    if lastmodify != lastmodify_prev:
        print('updating Lab metadata from google drive')
        dj.config['locations.metadata_lab']
        IDs = ['Experimenter','Rig']
        for ID in IDs:
            df_wr = online_notebook.fetch_lab_metadata(ID)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(dj.config['locations.metadata_lab']+ID+'.csv') 

        with open(dj.config['locations.metadata_lab']+'last_modify_time.json', "w") as write_file:
            json.dump(lastmodify, write_file)
        print('Lab metadata updated')
    
    #%% add users
    df_experimenters = pd.read_csv(dj.config['locations.metadata_lab']+'Experimenter.csv')
    experimenterdata = list()
    for experimenter in df_experimenters.iterrows():
        experimenter = experimenter[1]
        dictnow = {'username':experimenter['username'],'fullname':experimenter['fullname']}
        experimenterdata.append(dictnow)
    print('adding experimenters')
    for experimenternow in experimenterdata:
        try:
            lab.Person().insert1(experimenternow)
        except:# dj.DuplicateError:
            print('duplicate. experimenter: ',experimenternow['username'], ' already exists')
    
    #%% add rigs
    df_rigs = pd.read_csv(dj.config['locations.metadata_lab']+'Rig.csv')
    rigdata = list()
    for rig in df_rigs.iterrows():
        rig = rig[1]
        dictnow = {'rig':rig['rig'],'room':rig['room'],'rig_description':rig['rig_description']}
        rigdata.append(dictnow)
    print('adding rigs')
    for rignow in rigdata:
        try:
            lab.Rig().insert1(rignow)
        except dj.errors.DuplicateError:
            print('duplicate. rig: ',rignow['rig'], ' already exists')
            
    #%% populate subjects, surgeries and water restrictions
    print('adding surgeries and stuff')
    df_surgery = pd.read_csv(dj.config['locations.metadata_behavior']+'Surgery.csv')
    #%%
    for item in df_surgery.iterrows():
        if item[1]['status'] == 'experiment':
            subjectdata = {
                    'subject_id': int(item[1]['animal#']),
                    'username': item[1]['experimenter'],
                    'cage_number': item[1]['cage#'],
                    'date_of_birth': item[1]['DOB'],
                    'sex': item[1]['sex'],                    
                    'animal_source':None ,
                    }
            #print(subjectdata)
            try:
                lab.Subject.insert1(subjectdata)                
            except dj.errors.DuplicateError:
                print('duplicate. animal :',item[1]['animal#'], ' already exists')
            
                  
            surgeryidx = 1
            
            #while 'surgery date ('+str(surgeryidx)+')' in item[1].keys() and item[1]['surgery date ('+str(surgeryidx)+')'] and type(item[1]['surgery date ('+str(surgeryidx)+')']) == str:
#                start_time = datetime.strptime(item[1]['surgery date ('+str(surgeryidx)+')']+' '+item[1]['surgery time ('+str(surgeryidx)+')'],'%Y-%m-%d %H:%M')
#                end_time = start_time + timedelta(minutes = int(item[1]['surgery length (min) ('+str(surgeryidx)+')']))
            surgerydata = {
                    'surgery_id': surgeryidx,
                    'subject_id':item[1]['animal#'],
                    'username': item[1]['experimenter'],
                    'surgery_description': 'lesion',
                    }                        
#                    'subject_id':int(item[1]['animal#']),
#                    'username': item[1]['experimenter'],                    
#                    'brain_area': item[1]['BrainArea'],
#                    'hemisphere': item[1]['Hemisphere'],             
                    
            try:
                lab.Surgery.insert1(surgerydata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ')
                      
                      
            surgerylesiondata = {
                    'surgery_id': surgeryidx,
                    'lesion_id': surgeryidx,
                    'subject_id':item[1]['animal#'],
                    'method': item[1]['LesionMethod']
                    }
            try:
                lab.Surgery.Lesion.insert1(surgerylesiondata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery lesion animal ',item[1]['animal#'], ' already exists: ')
                      
                      
            brainareadata = {
                    #'surgery_id': surgeryidx,                                     
                    'brain_area': item[1]['BrainArea'],
                    'subject_id':item[1]['animal#'],
                    }
            try:
                lab.BrainArea.insert1(brainareadata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery brain area animal ',item[1]['animal#'], ' already exists: ')
                      
            hemispheredata = {                    
                    #'surgery_id': surgeryidx,
                    'hemisphere': item[1]['Hemisphere'],                    
                    'subject_id':item[1]['animal#'],
                    }
            try:
                lab.Hemisphere.insert1(hemispheredata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery hemisphere area animal ',item[1]['animal#'], ' already exists: ')
                      
                      
            trainingmethoddata = {                    
                    'training_method':item[1]['TrainingMethod'],
                    'subject_id':item[1]['animal#'],
                    }
            try:
                lab.Training.insert1(trainingmethoddata)
            except dj.errors.DuplicateError:
                print('duplicate. training animal ',item[1]['animal#'], ' already exists: ')
             
                #print(surgerydata)
#%%                #checking craniotomies
                #%
#                cranioidx = 1
#                while 'craniotomy diameter ('+str(cranioidx)+')' in item[1].keys() and item[1]['craniotomy diameter ('+str(cranioidx)+')'] and (type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == int or type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == float):
#                    if item[1]['craniotomy surgery id ('+str(cranioidx)+')'] == surgeryidx:
#                        proceduredata = {
#                                'surgery_id': surgeryidx,
#                                'subject_id':item[1]['animal#'],
#                                'procedure_id':cranioidx,
#                                'skull_reference':item[1]['craniotomy reference ('+str(cranioidx)+')'],
#                                'ml_location':item[1]['craniotomy lateral ('+str(cranioidx)+')'],
#                                'ap_location':item[1]['craniotomy anterior ('+str(cranioidx)+')'],
#                                'surgery_procedure_description': 'craniotomy: ' + item[1]['craniotomy comments ('+str(cranioidx)+')'],
#                                }
#                        try:
#                            lab.Surgery.Procedure().insert1(proceduredata)
#                        except dj.DuplicateError:
#                            print('duplicate cranio for animal ',item[1]['animal#'], ' already exists: ', cranioidx)
#                    cranioidx += 1
#                #% 
                
#                virusinjidx = 1
#                while 'virus inj surgery id ('+str(virusinjidx)+')' in item[1].keys() and item[1]['virus inj virus id ('+str(virusinjidx)+')'] and item[1]['virus inj surgery id ('+str(virusinjidx)+')']:
#                    if item[1]['virus inj surgery id ('+str(virusinjidx)+')'] == surgeryidx:
#    # =============================================================================
#    #                     print('waiting')
#    #                     timer.sleep(1000)
#    # =============================================================================
#                        if '[' in item[1]['virus inj lateral ('+str(virusinjidx)+')']:
#                            virus_ml_locations = eval(item[1]['virus inj lateral ('+str(virusinjidx)+')'])
#                            virus_ap_locations = eval(item[1]['virus inj anterior ('+str(virusinjidx)+')'])
#                            virus_dv_locations = eval(item[1]['virus inj ventral ('+str(virusinjidx)+')'])
#                            virus_volumes = eval(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])
#                        else:
#                            virus_ml_locations = [int(item[1]['virus inj lateral ('+str(virusinjidx)+')'])]
#                            virus_ap_locations = [int(item[1]['virus inj anterior ('+str(virusinjidx)+')'])]
#                            virus_dv_locations = [int(item[1]['virus inj ventral ('+str(virusinjidx)+')'])]
#                            virus_volumes = [int(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])]
#                            
#                        for virus_ml_location,virus_ap_location,virus_dv_location,virus_volume in zip(virus_ml_locations,virus_ap_locations,virus_dv_locations,virus_volumes):
#                            injidx = len(lab.Surgery.VirusInjection() & surgerydata) +1
#                            virusinjdata = {
#                                    'surgery_id': surgeryidx,
#                                    'subject_id':item[1]['animal#'],
#                                    'injection_id':injidx,
#                                    'virus_id':item[1]['virus inj virus id ('+str(virusinjidx)+')'],
#                                    'skull_reference':item[1]['virus inj reference ('+str(virusinjidx)+')'],
#                                    'ml_location':virus_ml_location,
#                                    'ap_location':virus_ap_location,
#                                    'dv_location':virus_dv_location,
#                                    'volume':virus_volume,
#                                    'dilution':item[1]['virus inj dilution ('+str(virusinjidx)+')'],
#                                    'description': 'virus injection: ' + item[1]['virus inj comments ('+str(virusinjidx)+')'],
#                                    }
#                            try:
#                                lab.Surgery.VirusInjection().insert1(virusinjdata)
#                            except dj.DuplicateError:
#                                print('duplicate virus injection for animal ',item[1]['animal#'], ' already exists: ', injidx)
#                    virusinjidx += 1    
#                #%
#                
#                surgeryidx += 1
                    
                #%%
            if item[1]['ID']:
                #df_wr = online_notebook.fetch_water_restriction_metadata(item[1]['ID'])
                try:
                    df_wr = pd.read_csv(dj.config['locations.metadata_behavior']+item[1]['ID']+'.csv')
                except:
                    df_wr = None
                if type(df_wr) == pd.DataFrame:
                    wrdata = {                            
                            'water_restriction_number': item[1]['ID'],
                            'subject_id':int(item[1]['animal#']),
                            'cage_number': item[1]['cage#'],
                            'wr_start_date': '0001-01-01',
                            'wr_start_weight': 0,
                            }
                try:
                    lab.WaterRestriction().insert1(wrdata)
                except dj.errors.DuplicateError:
                    print('duplicate. water restriction :',item[1]['animal#'], ' already exists')
                    
            
                
#                if type(df_wr) == pd.DataFrame:
#                    taskdata = {
#                            'task_description': df_wr['Task type'],                           
#                            }
#                    taskdata1 = {
#                            'task_protocol_description': item[1]['TaskProtocol'],                            
#                            }
#                    statusData = {
#                            'status': item[1]['Status'],
#                            }
#                    videoFrameNum = {
#                            'tracking_frame_num': 
#                            }
#                try:
#                    experiment.Task().insert1(taskdata)
#                    experiment.TaskProtocol().insert1(taskdata1)
#                    experiment.SessionStatus().insert1(statusData)
#                except dj.DuplicateError:
#                    print('duplicate. taskdata :',item[1]['animal#'], ' already exists')
            
            
            
            
                            
            
                            