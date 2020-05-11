# -*- coding: utf-8 -*-
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('Lesion.json', scope)
client = gspread.authorize(creds)

#%% open
#sheet = client.open('Lesion').sheet1
#lesion = sheet.get_all_records()

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
    wb = client.open("Lesion")
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
    wb = client.open("Lesion")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
        #%%
    if ID in sheetnames:
        idx_now = sheetnames.index(ID)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(ID+'!A1:O100',params)
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
    return fetch_lastmodify_time('Lesion')

def fetch_lastmodify_time_lab_metadata():
    return fetch_lastmodify_time('Lesion metadata')


def fetch_lab_metadata(ID):
    #%%
    wb = client.open("Lesion metadata")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
        #%%
    if ID in sheetnames:
        idx_now = sheetnames.index(ID)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(ID+'!A1:O100',params)
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