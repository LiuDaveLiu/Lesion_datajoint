import json
project = 'foraging'
with open('dj_local_conf.json') as json_file:
    variables = json.load(json_file)
variables['project'] = project
with open('dj_local_conf.json', 'w') as outfile:
    json.dump(variables, outfile)
import datapipeline_lesion_metadata
import datapipeline_lesion_behavior
#import datapipeline_elphys
#%%
datapipeline_lesion_metadata.populatemetadata()
datapipeline_lesion_behavior.populatebehavior()
#datapipeline_behavior.populatemytables()
#datapipeline_elphys.populateelphys()
#datapipeline_elphys.populatemytables()

#%%

#%%
# =============================================================================
# 
# df_subject_wr=pd.DataFrame(lab.WaterRestriction() * experiment.Session() * experiment.SessionDetails)
# subject_names = df_subject_wr['water_restriction_number'].unique()
# subject_names.sort()
# 
# for wr_name in subject_names:
#     subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
#     key = {
#             'subject_id':subject_id
#             }
#     sessionstats = pd.DataFrame((behavioranal.SessionStats() *experiment.Session() * experiment.SessionDetails()) &key)
#     if any(sessionstats['session_trialnum']<100) and wr_name != 'FOR04':
#         break
# =============================================================================



# =============================================================================
# p_reward_L, p_reward_R, n_trials = foraging_model.generate_block_structure(n_trials_base=80,n_trials_sd=10,blocknum = 3, reward_ratio_pairs=np.array([[.4,.05],[.3857,.0643],[.3375,.1125]]))
# rewards_random = foraging_model.run_task(p_reward_L,
#                                           p_reward_R,
#                                           n_trials,
#                                           unchosen_rewards_to_keep = 1,
#                                           subject = 'clever',
#                                           min_rewardnum = 30, 
#                                           filter_tau_fast =3,
#                                           filter_tau_slow = 5, 
#                                           filter_tau_slow_amplitude = 0,
#                                           filter_constant = 0.00,
#                                           softmax_temperature =1,
#                                           differential = True,
#                                           plot = True)
# 
# =============================================================================

        
    