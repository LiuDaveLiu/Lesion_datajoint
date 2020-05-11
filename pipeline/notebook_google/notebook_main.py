import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# use creds to create a client to interact with the Google Drive API
scope = ['https://www.googleapis.com/auth/analytics.readonly',
      'https://www.googleapis.com/auth/drive',
      'https://www.googleapis.com/auth/spreadsheets',
      ]#['https://spreadsheets.google.com/feeds']
#creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
creds = ServiceAccountCredentials.from_json_keyfile_name('./creds/online_notebook_drive_api.json', scope)
client = gspread.authorize(creds)
creds_drive = ServiceAccountCredentials.from_json_keyfile_name('./creds/online_notebook_drive_api.json', scope)

#%% open 

def fetch_lastmodify_time(spreadsheetname):
    modifiedtime = None
    ID = None
    service = build('drive', 'v3', credentials=creds)
    wb = client.open(spreadsheetname)
    ID = wb.id
    if ID:
        modifiedtime = service.files().get(fileId = ID,fields = 'modifiedTime').execute()
    return modifiedtime
def fetch_animal_metadata():
    #%%
    wb = client.open("Surgery, water restriction and training")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    idx_main = sheetnames.index('Surgery')
    main_sheet = wb.get_worksheet(idx_main)
    df = pd.DataFrame(main_sheet.get_all_records())
    #%%
    return df

def fetch_water_restriction_metadata(ID):
    #%%
    wb = client.open("Surgery, water restriction and training")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
        #%%
    if ID in sheetnames:
        idx_now = sheetnames.index(ID)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(ID+'!A1:Y100',params)
            temp = temp['values']
            header = temp.pop(0)
            data = list()
            for row in temp:
                if len(row) < len(header):
                    row.append('')
                if len(row) == len(header):
                    data.append(row)
            df = pd.DataFrame(data, columns = header)
            return df
        else:
            return None
    else:
        return None

def fetch_lastmodify_time_animal_metadata():
    return fetch_lastmodify_time('Surgery, water restriction and training')

def fetch_lastmodify_time_lab_metadata():
    return fetch_lastmodify_time('Lab metadata')


def fetch_lab_metadata(ID):
    #%%
    wb = client.open("Lab metadata")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
        #%%
    if ID in sheetnames:
        idx_now = sheetnames.index(ID)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(ID+'!A1:Y100',params)
            temp = temp['values']
            header = temp.pop(0)
            data = list()
            for row in temp:
                if len(row) < len(header):
                    row.append('')
                if len(row) == len(header):
                    data.append(row)
            df = pd.DataFrame(data, columns = header)
            return df
        else:
            return None
    else:
        return None
# =============================================================================
#     modifiedtime = None
#     ID = None
#     service = build('drive', 'v3', credentials=creds)
#     wb = client.open("Surgery, water restriction and training")
#     ID = wb.id
#     if ID:
#         modifiedtime = service.files().get(fileId = ID,fields = 'modifiedTime').execute()
#     return modifiedtime
# =============================================================================



#%%


    
    
#%%
# =============================================================================
# #%%
# 
# 
# # Find a workbook by name and open the first sheet
# # Make sure you use the right name here.
# wb = client.open("waterport calibration")
# sheet = wb.get_worksheet(0)
# # Extract and print all of the values
# df = pd.DataFrame(sheet.get_all_records())
# df = df[(df['num']!='Average') & (df['num']!= 'STD')]
# Left = np.mean(df[df['Direction']=='Left'])
# Right = np.mean(df[df['Direction']=='Right'])
# for vol in range(10,60,10):
#     
#     print(vol)
# plt.plot(Left)
# plt.plot(Right)
# =============================================================================


