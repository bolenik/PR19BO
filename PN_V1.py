
import fnmatch,os
import pandas as pd
import geopandas as gpd
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import pyproj as proj
import folium
import folium.plugins as plugins
from folium.plugins import HeatMapWithTime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
# or: requests.get(url).content




month_list = ['Januar', 'Februar', 'Marec', 'April', 'Maj', 'Junij', 
              'Julij', 'Avgust', 'September', 'Oktober', 'November', 
              'December']


#monthName = {u'1':'Januar', u'2':'Februar', u'3':'Marec', 
#       u'4':'April', u'5':'Maj', u'6':'Junij', u'7':'Julij', u'8':'Avgust', u'9':'September',
#       u'10':'Oktober', u'11':'November', u'12':'December'}

monthName = {1:'Januar', 2:'Februar', 3:'Marec', 
       4:'April', 5:'Maj', 6:'Junij', 7:'Julij', 8:'Avgust', 9:'September',
       10:'Oktober', 11:'November', 12:'December'}


day = {u'Monday':'Ponedeljek', u'Tuesday':'Torek', u'Wednesday':'Sreda', 
       u'Thursday':'Četrtek', u'Friday':'Petek', u'Saturday':'Sobota', u'Sunday':'Nedelja'}
dayNumber = {u'Monday':'1', u'Tuesday':'2', u'Wednesday':'3', 
       u'Thursday':'4', u'Friday':'5', u'Saturday':'6', u'Sunday':'7'}

monthLength_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

tripleMonthLength_list = [31, 31, 31, 28, 28, 28, 31, 31, 31, 30, 30, 30, 
                          31, 31, 31, 30, 30, 30, 31, 31, 31, 31, 31, 31,
                          30, 30, 30, 31, 31, 31, 30, 30, 30, 31, 31, 31]

df1=''

def PreberiPodatkeUE():
    df_Prebivalstvo = pd.read_csv("./PrebivalstvoUE.csv", error_bad_lines=False,sep=';',encoding="ANSI")
    df_Prebivalstvo = pd.melt(df_Prebivalstvo, id_vars=['UpravnaEnota'], var_name=['2016,2017,2018'])
    df_Prebivalstvo.columns = ['UpravnaEnota', 'Leto','SteviloPrebivalcev']
    df_Prebivalstvo['Leto'] = pd.to_numeric(df_Prebivalstvo['Leto'], errors='coerce')
    df_Prebivalstvo['SteviloPrebivalcev'] = pd.to_numeric(df_Prebivalstvo['SteviloPrebivalcev'], errors='coerce')
    df_Prebivalstvo['UpravnaEnota'] = df_Prebivalstvo['UpravnaEnota'].str.upper()
    df_Prebivalstvo.head()

def DownloadFiles():
    for i in range(2010,2020,1):
        path = "https://www.policija.si/baza/pn" +  str(i) + ".zip"
        #path = "https://www.policija.si/baza/pn2010.zip"
        resp = urlopen(path)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall("./files")

def MergeFiles():
    os.chdir("c:/_PERSONAL/FRI/PR/Projekt/files")
    results = pd.DataFrame([])
    
    # test = os.listdir(os.getcwd())
    # Testiranje kode
    # df1 = pd.read_csv('./PN2018.csv', error_bad_lines=False,sep=';',encoding="ANSI")

    for counter, file in enumerate(glob.glob("PN*.csv")):
        namedf = pd.read_csv("./" +file, error_bad_lines=False,sep=';',encoding="ANSI")
        results = results.append(namedf)
    
    results.to_csv('./PrometneNesrece.csv',sep=';',encoding="ANSI")

def ReadFiles(fileName):
    fileName = 'PN2018'
    df1 = pd.read_csv('./' + fileName + '.csv', error_bad_lines=False,sep=';',encoding="ANSI")
    return df1

def ReadAllFiles():
    df_PN_All = pd.read_csv('./PrometneNesrece.csv', error_bad_lines=False,sep=';',encoding="ANSI")
    df_PN_All.drop('Unnamed: 0', axis=1, inplace=True)
    return df_PN_All

def GetSummary(df1):
    summary = df1.describe()
    print (summary)

def ChangeLonLat(df1):
    p2 = proj.Proj(init="epsg:3794")
    
    lon, lat = p2(393648,84611,inverse=True)
    
    
def convertCoords(row):
    p2 = proj.Proj(init="epsg:3794")
    x2,y2 = p2(row['GeoKoordinataY'],row['GeoKoordinataX'],inverse=True)
    return pd.Series({'newLong':y2,'newLat':x2})

   
    
def CleanDataNullValues(df1):
    # Check missing values
    
    num_cols = df1._get_numeric_data().columns
    for n in df1.columns:
        if n not in num_cols:
            print (n)
            test = df1[df1[n].isnull()].index.tolist()
            if len(test) > 0:
                if (n != "StarostniRazred") & (n != "VozniskiStazVLetihRazred"):
                    df1[n].fillna("Ni podatka", inplace = True)


def CleanDataReplaceAge(df1):
    #Check if any of records has data in starost < 0
    #Check if any of records has data in vozniskiStaz < 0 or vozniskiStaz > 90
    
    #popravimo dataframe, tako da upoštevamo samo starosti več kot 0, predvidevamo, da je starost 0 napaka v podatkih
    
    df1 = df1[(df1['Starost'] > 0)]
    return df1
    
def ReplaceValues(df1):
    #spremenimo datumPN v pravilni datumski format
    df1['DatumPNConverted'] = pd.to_datetime(df1['DatumPN'], format='%d.%m.%Y')
    #df1['DatumPN'] = df1['DatumPN'].map(lambda date_string: datetime.strptime(date_string, "%d.%m.%Y"))

def PripraviPodatkeZaAnalizo(df1):
    df1 = df1
    ReplaceValues(df1)
    df1['ZaporednaStevilkaPN'] = pd.to_numeric(df1['ZaporednaStevilkaPN'], errors='coerce')
    df1['stPN'] = df1['Leto'].map(str) + "" + df1['ZaporednaStevilkaPN'].map(str)
    df1['stPN'] = pd.to_numeric(df1['stPN'], errors='coerce')
    df1['Dan'] = df1['DatumPNConverted'].dt.day
    df1['Leto'] = df1['DatumPNConverted'].dt.year
    df1['Mesec'] = df1['DatumPNConverted'].dt.month
    df1['MesecNaziv'] = df1['Mesec'].replace(monthName, regex=True)
    df1['DanVTednu'] = df1['DatumPNConverted'].dt.day_name()
    df1['DanVTednuStevilka'] = df1['DanVTednu'].replace(dayNumber, regex=True)
    df1['DanVTednu'] = df1['DanVTednu'].replace(day, regex=True)
    df1['DrzavljanstvoSkupina'] = np.where((df1['Drzavljanstvo']!='SLOVENIJA') & (df1['Drzavljanstvo']!='NEZNANO'), 'TUJEC', df1['Drzavljanstvo'])
    df1['DrzavljanstvoSkupina']  = df1['DrzavljanstvoSkupina'].astype('category')
    
    df1['VrednostAlkotesta']  = df1['VrednostAlkotesta'].str.replace(',','.')
    df1['VrednostAlkotesta'] = pd.to_numeric(df1['VrednostAlkotesta'], errors='coerce')
    df1['PrisotnostAlkohola']  = np.where((df1['VrednostAlkotesta']>0), 'DA', 'NE')
    #------------------------------------------------------------------------------------------------------v
    #Pripravi starostne razrede za lažji porazdelitev 
    #------------------------------------------------------------------------------------------------------v
    bins = [0, 16, 18, 21, 50, 65, np.inf]
    names = ['<16', '16-18', '18-21','21-50', '50-65', '65+']
    df1['StarostniRazred'] = pd.cut(df1['Starost'], bins, labels=names)
    df1['StarostniRazred']  = df1['StarostniRazred'].astype('category')
    
    #------------------------------------------------------------------------------------------------------v
    #Pripravi razrede za vozniski staz za lažjo porazdelitev
    #------------------------------------------------------------------------------------------------------v
    bins = [0,2,5, 10,20,30, np.inf]
    names = ['<2', '2-5', '5-10', '10-20', '20-30', '30+']
    df1['VozniskiStazVLetihRazred'] = pd.cut(df1['VozniskiStazVLetih'], bins, labels=names)
    df1['VozniskiStazVLetihRazred'] = np.where((df1['VozniskiStazVLetih'] == 0) & (df1['VozniskiStazVMesecih'] > 0), '<2', df1['VozniskiStazVLetihRazred'])
    df1['VozniskiStazVLetihRazred']  = df1['VozniskiStazVLetihRazred'].astype('category')
    
        
    df1.loc[df1['KlasifikacijaNesrece'] == 'Z MATERIALNO ŠKODO','razred'] = 'BREZ POŠKODB'
    df1.loc[df1['KlasifikacijaNesrece'] == 'S SMRTNIM IZIDOM','razred'] = 'SMRTNI IZID'
    df1.loc[df1['KlasifikacijaNesrece'] == 'Z LAŽJO TELESNO POŠKODBO','razred'] = 'TELESNA POŠKODBA'
    df1.loc[df1['KlasifikacijaNesrece'] == 'S HUDO TELESNO POŠKODBO','razred'] = 'TELESNA POŠKODBA'
    
    df1['SmrtniIzid'] = np.where((df1['razred'] =='SMRTNI IZID') , 1, 0)
    df1['VinjenaOseba'] = np.where ((df1['VrednostAlkotesta'] > 0.24), 1,0)
    
    #Prometna konica. Kako se giba proment v prometni konici
    df1['PromentaKonica'] = np.where((df1['UraPN'] >=6) & (df1['UraPN'] <= 8) , 1, 0)
    df1['PromentaKonica'] = np.where((df1['UraPN'] >=15) & (df1['UraPN'] <= 17) & (df1['PromentaKonica'] == 0) , 1, 0)
    
    df1['Tema'] = np.where((df1['UraPN'] >= 22) & (df1['UraPN'] <= 6), 1, 0)
    
        
    df1['VozniskiStaz'] =  (df1['VozniskiStazVLetih'] * 12  + df1['VozniskiStazVMesecih'] ) /12
    
    df_st = df1.groupby(['stPN']).agg(
            {'Starost':['min','max','mean'],
             'VozniskiStaz':['min','max','mean'],
             'ZaporednaStevilkaOsebeVPN':['size'],
             'VrednostAlkotesta':['mean','max']
                    })
    df_st.columns = ["_".join(x) for x in df_st.columns.ravel()]
    df_st.reset_index(inplace=True)
    df_st.head()
    
    df1 = pd.merge(df1, df_st, on='stPN', how='inner')
    
    #zamenjaj koordinate, tako da se bo lahko pokazalo pravilno sliko prometnih nesreč
    #traja zelo dolgo, da se izvede (5 min +). 
    #v kolikor ni potrebe naj se ne izvaja
    #df1 = df1.join(df1.apply(convertCoords, axis=1))
    
def defDistAccidentsPerYear():
    #*********************************************************************************************************
    #*********************************************************************************************************
    #USE: Uporabi pri vizualizaciji
    #*********************************************************************************************************
    #*********************************************************************************************************
    byDate = df1[['DatumPNConverted','Leto','Mesec','stPN']].groupby(['DatumPNConverted','Mesec','Leto']).count()
    byDate.reset_index(inplace=True)
    byDate = byDate[byDate['Leto']==2018]

    #Identify outliers
    dailyAccMean = byDate['stPN'].mean()
    dailyAccSD = byDate['stPN'].std()
    
    check_high = dailyAccMean + 2*dailyAccSD
    check_low = dailyAccMean - 2*dailyAccSD
    byDate['isOutlier'] = (byDate['stPN'] > check_high)|(byDate['stPN'] < check_low)
    byDate['pltColour'] = '#B9BCC0'
    byDate.loc[byDate['isOutlier'] == True, 'pltColour'] = 'r'
    
    #Calculate the distribution of accidents throughout the day
    byHour = df1[['UraPN','stPN','Leto']].groupby(['UraPN','Leto']).count()
    byHour.reset_index(inplace=True)
    byHour = byHour[byHour['Leto']==2018]
    byHour['stPN'] = byHour['stPN'].apply(lambda x: x/366) # Transform total to per day value
    
    #vizualizacija
    from matplotlib import dates as dates
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2,1, figsize = (12,8))
    plotDates = dates.date2num(byDate['DatumPNConverted'])
    plt.style.use('ggplot')
    months = dates.MonthLocator()  # every month
    monthsFmt = dates.DateFormatter('%b')
    #Fit regression line for byDay data
    pCoeff = np.polyfit(plotDates, byDate['stPN'], 2)
    regLine = np.poly1d(pCoeff)
    
    ax[0].scatter(plotDates, byDate['stPN'],marker= 'x', c= byDate['pltColour'])
    ax[0].plot(plotDates, regLine(plotDates), 'k--')
    ax[0].xaxis.set_major_locator(months)
    ax[0].xaxis.set_major_formatter(monthsFmt)
    ax[0].set_title('Število nesreč po dnevih v letu 2018', fontsize=16)
    ax[0].set_xlabel('DatumPNConverted')
    plt.tight_layout()
    
    #*********************************************************************************************************
    #*********************************************************************************************************
    #USE: Uporabi pri vizualizaciji
    #*********************************************************************************************************
    #*********************************************************************************************************
    ax[1].plot(byHour['UraPN'], byHour['stPN'], c='b')
    ax[1].set_title('Povprečno število nesreč po urah v letu 2018', fontsize=16)
    ax[1].set_xlabel('Ura v dnevu')

#število udeleženecev v prometnih nesrečah po letih
def SteviloUdelezencevPoLetih():
    
    # casualties by year in UK
    nesrece_slo = df1[['Leto','stPN','ZaporednaStevilkaOsebeVPN_size']].groupby(['Leto','ZaporednaStevilkaOsebeVPN_size']).count()
    nesrece_slo.reset_index(inplace=True)
    nesrece_slo = nesrece_slo[nesrece_slo['ZaporednaStevilkaOsebeVPN_size'] < 6]
    
    sns.set(font_scale=1.4)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 2)
    bplot = sns.barplot(x = "ZaporednaStevilkaOsebeVPN_size", y = "stPN", hue = "Leto", #palette="BuGn_d", 
                        data= nesrece_slo[nesrece_slo["stPN"]>100])
    bplot.set_title("Število nesreč po številu udeležencev za zadnje 4 leta v Sloveniji")
    bplot.set_xlabel("Število udeležencev prometne nesreče")
    bplot.set_ylabel("")
    # Put the legend out of the figure
    bplt = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def PrisotnostAlkoholaVPrometniNesrec():
    df_vinjenaOseba = df1[['stPN','KlasifikacijaNesrece','razred','UpravnaEnotaStoritve','DatumPN',
                                 'VinjenaOseba', 'PromentaKonica','SmrtniIzid','Starost','VozniskiStaz',
               'UraPN','VNaselju','Lokacija','VrstaCesteNaselja','SifraCesteNaselja',
               'TekstCesteNaselja','SifraOdsekaUlice','TekstOdsekaUlice','Povzrocitelj',
                  'StacionazaDogodka','OpisKraja','VzrokNesrece','TipNesrece','VrednostAlkotesta',
                  'VremenskeOkoliscine','StanjePrometa','StanjeVozisca','VrstaVozisca','newLong','newLat',
                  'DatumPNConverted','Dan','Leto','Mesec','MesecNaziv','DanVTednu','DanVTednuStevilka']].copy()
    
    
    df_vinjenaOseba = df_vinjenaOseba[df_vinjenaOseba["Povzrocitelj"]=="POVZROČITELJ"]
    
    
    df_vinjenaOsebaSume = df_vinjenaOseba.groupby(['stPN']).agg(
            {'Starost':['min','max','mean'],
             'VozniskiStaz':['min','max','mean'],
             'stPN':['count'],
             'VrednostAlkotesta':['mean','max','min']
                    })
    
    df_vinjenaOsebaSume.columns = ["_".join(x) for x in df_vinjenaOsebaSume.columns.ravel()]
    df_vinjenaOsebaSume.reset_index(inplace=True)
    df_vinjenaOseba = pd.merge(df_vinjenaOseba, df_vinjenaOsebaSume, on='stPN', how='inner')
    
    df_vinjenPovzroc = df_vinjenaOseba[df_vinjenaOseba["VinjenaOseba"]==1]
    df_vsi = df_vinjenaOseba
    
    df_vinjenaOseba = df_vinjenaOseba.rename_axis(None)
    df_vinjenaOseba = df_vinjenaOseba.groupby(['Leto','Mesec','VinjenaOseba'])['stPN'].nunique() 
    df_vinjenaOseba = df_vinjenaOseba.reset_index()[['Leto','Mesec','VinjenaOseba', 'stPN']]
    df_vinjenaOseba.rename(columns={'stPN':'Stevilo nesrec'}, 
                                inplace=True )

    df_vinjenaOseba['MesecNaziv'] = df_vinjenaOseba['Mesec'].replace(monthName, regex=True)
    # create bar plot
    #*********************************************************************************************************************************************************
    #*********************************************************************************************************************************************************
    # prikaži column chart, enega ob drugem
    #*********************************************************************************************************************************************************
    steviloVinjenihOsebPoMesecih = sns.barplot(x='MesecNaziv', y='Stevilo nesrec', 
                                          data=df_vinjenaOseba, 
                                          hue='VinjenaOseba', 
                                          edgecolor='black', alpha=0.75, 
                                          linewidth=1)
    
    # create bar plot
    #*********************************************************************************************************************************************************
    #*********************************************************************************************************************************************************
    # vinjeni povzrocitelji po VzrokNesrece
    #*********************************************************************************************************************************************************
    df_vinjenPovzrocMesec = df_vinjenPovzroc.rename_axis(None)
    df_vinjenPovzrocMesec = df_vinjenPovzrocMesec.groupby(['Leto','Mesec'])['stPN'].nunique() 
    df_vinjenPovzrocMesec = df_vinjenPovzrocMesec.reset_index()[['Leto','Mesec', 'stPN']]
    df_vinjenPovzrocMesec = df_vinjenPovzrocMesec.groupby(['Mesec']).agg(
            {
             'stPN':['sum','mean']
                    })
    df_vinjenPovzrocMesec.columns = ["_".join(x) for x in df_vinjenPovzrocMesec.columns.ravel()]
    df_vinjenPovzrocMesec.reset_index(inplace=True)
    df_vinjenPovzrocMesec.rename(columns={'stPN_sum':'Stevilo nesrec','stPN_mean':'Povprečno število'}, 
                                inplace=True )

    df_vinjenPovzrocMesec['MesecNaziv'] = df_vinjenPovzrocMesec['Mesec'].replace(monthName, regex=True)
    df_vinjenPovzrocMesec.sort_values(by='Mesec', ascending=False)
    df_vinjenPovzrocMesec['MesecNaziv'] = pd.Categorical(df_vinjenPovzrocMesec['MesecNaziv'], ordered=True, categories=month_list)
    
    #vsi 
    df_vinjenOsebaMesec = df_vsi.rename_axis(None)
    df_vinjenOsebaMesec = df_vinjenOsebaMesec.groupby(['Leto','Mesec'])['stPN'].nunique() 
    df_vinjenOsebaMesec = df_vinjenOsebaMesec.reset_index()[['Leto','Mesec', 'stPN']]
    df_vinjenOsebaMesec = df_vinjenOsebaMesec.groupby(['Mesec']).agg(
            {
             'stPN':['sum','mean']
                    })
    df_vinjenOsebaMesec.columns = ["_".join(x) for x in df_vinjenOsebaMesec.columns.ravel()]
    df_vinjenOsebaMesec.reset_index(inplace=True)
    df_vinjenOsebaMesec.rename(columns={'stPN_sum':'Stevilo nesrec','stPN_mean':'Povprečno število'}, 
                                inplace=True )

    df_vinjenOsebaMesec['MesecNaziv'] = df_vinjenOsebaMesec['Mesec'].replace(monthName, regex=True)
    df_vinjenOsebaMesec.sort_values(by='Mesec', ascending=False)
    df_vinjenOsebaMesec['MesecNaziv'] = pd.Categorical(df_vinjenOsebaMesec['MesecNaziv'], ordered=True, categories=month_list)
    # create bar plot
    #*********************************************************************************************************************************************************
    #*********************************************************************************************************************************************************
    # prikaži column chart, enega ob drugem
    #*********************************************************************************************************************************************************
    sns.set_context('paper')  #Everything is sized for a presentation
    steviloVinjenihOsebPoMesecih = sns.barplot(x='MesecNaziv', y='Stevilo nesrec', ci=None,
                                          data=df_vinjenPovzrocMesec, color = 'darkgray',
                                          #hue='VinjenaOseba', 
                                          edgecolor='black', alpha=0.8, 
                                          linewidth=1)
    
    steviloVinjenihOsebPoMesecih = sns.lineplot(x='MesecNaziv', y='Povprečno število', ci=None,
                                          data=df_vinjenPovzrocMesec, color = 'blue',
                                          #hue='VinjenaOseba', 
                                           alpha=0.8, 
                                          linewidth=1)
    
    plt.title("Povprečno število nesreč s smrtnim izidom - prisotnost alkohola", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Povprečno število prometnih nesreč", size=13)
    plt.legend();
    plt.show()
    
    steviloVinjenihOsebPoMesecih = sns.lineplot(x='MesecNaziv', y='Povprečno število', ci=None,
                                          data=df_vinjenOsebaMesec, color = 'blue',
                                          #hue='VinjenaOseba', 
                                           alpha=0.8, 
                                          linewidth=1)
    plt.title("Povprečno število nesreč po mesecih", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Povprečno število prometnih nesreč", size=13)
    plt.legend();
    plt.show()
    
    #kaj se dogaja v mesecu decembru
    # set up data
    df_vinjenOsebaMesecDecember = df_vinjenPovzroc[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','stPN']].copy()
    df_vinjenOsebaMesecDecember = df_vinjenOsebaMesecDecember.query("Mesec == 12")
    #-------------------------------------------------------------------------------------------------------------------
    #priprava podatkov
    #-------------------------------------------------------------------------------------------------------------------
    df_vinjenOsebaMesecDecember = df_vinjenOsebaMesecDecember.rename_axis(None)
    df_vinjenOsebaMesecDecember = df_vinjenOsebaMesecDecember.groupby(['Dan','Leto'])['stPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
    df_vinjenOsebaMesecDecember = df_vinjenOsebaMesecDecember.groupby(['Dan']).agg(
            {
             'Število nesreč':['sum','mean']
                    })
    df_vinjenOsebaMesecDecember.columns = ["_".join(x) for x in df_vinjenOsebaMesecDecember.columns.ravel()]
    df_vinjenOsebaMesecDecember.reset_index(inplace=True)
    df_vinjenOsebaMesecDecember.rename(columns={'Število nesreč_mean':'Povprečno število'}, 
                                inplace=True )

    #    trafficDataByWeel_df.rename(columns={'stPN':'Stevilo nesrec'}, 
#                                inplace=True )
    #-------------------------------------------------------------------------------------------------------------------
    #priprava podatkov
    #-------------------------------------------------------------------------------------------------------------------
    plt.rcParams['figure.figsize'] = [15,5]
    plt.bar(df_vinjenOsebaMesecDecember['Dan'], 
            df_vinjenOsebaMesecDecember['Povprečno število']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    plt.title("Povprečno število nesreč - povzročitelj alkoholizirani voznik", size=16)
    plt.xlabel("Dan", size=13)
    plt.ylabel("Povprečno število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    plt.show()
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid") 
    sns.boxplot(x='Dan', y='Povprečno število', data=df_vinjenOsebaMesecDecember,palette=None,saturation=0.75, width=0.8,whis=1.5)
    plt.title("Povprečno število nesreč - povzročitelj alkoholizirani voznik", size=16)
    plt.xlabel("Dan", size=13)
    plt.ylabel("Povprečno število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')
    
    
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))
    
    ##################################################################################################################
    ##################################################################################################################
    #UPRAVNA ENOTA NORMALIZIRAJ GLEDE NA ŠTEVILO PREBIVALCEV
    ##################################################################################################################
    # Load the example car crash dataset
    crashes = df_vinjenaOseba[['stPN', 'UpravnaEnotaStoritve','VinjenaOseba']].copy()
    crashes = crashes.groupby(['UpravnaEnotaStoritve','VinjenaOseba']).agg(
            {
             'stPN':['sum','nunique']
                    })
    crashes.columns = ["_".join(x) for x in crashes.columns.ravel()]
    crashes.reset_index(inplace=True)
    crashes_pivot = crashes.pivot_table(values=['stPN_nunique'], 
                                                        index=['UpravnaEnotaStoritve'], 
                                                        columns=['VinjenaOseba'])
    
    crashes_pivot.columns = [ 'NiVinjenaOseba', 'VinjenaOseba' ]
    crashes_pivot = crashes_pivot.reset_index()
    crashes_pivot['Skupaj'] = crashes_pivot['NiVinjenaOseba'] + crashes_pivot['VinjenaOseba']
    crashes_pivot['OdstotekVinjenihPovzročiteljev'] = crashes_pivot['Skupaj'] /crashes_pivot['VinjenaOseba']
    # Plot the total crashes
    crashes_pivot = crashes_pivot.query('OdstotekVinjenihPovzročiteljev > 10').sort_values('OdstotekVinjenihPovzročiteljev', ascending=False)
    ##################################################################################################################
    #SCATTER CHART
    ##################################################################################################################
    ax = sns.scatterplot(x="OdstotekVinjenihPovzročiteljev", y="Skupaj", data=crashes_pivot,hue="UpravnaEnotaStoritve")
        # Add a legend and informative axis label
        
    ax.legend(ncol=2, loc="lower right", frameon=True)
    bplt = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set( 
           xlabel="Odstotek prometnih nesreč",
           ylabel="Število prometnih nesreč",
           title="Razmerje prometnih nesreč - vinjeni povzročitelj")
    plt.title("Povprečno število nesreč - povzročitelj alkoholizirani voznik", size=16)
    
    for index, row in crashes_pivot.head(2).iterrows():
        ax.text(row["OdstotekVinjenihPovzročiteljev"]+0.2, row["Skupaj"], row["UpravnaEnotaStoritve"], horizontalalignment='left', size='medium', color='black', weight='semibold')
    
    sns.despine(left=True, bottom=True)
        
        
    sns.set_color_codes("pastel")
    plt.figure(figsize=(10,6))
    sns.barplot(x="OdstotekVinjenihPovzročiteljev", y="UpravnaEnotaStoritve", data=crashes_pivot,
                label="Total", color="b")
    plt.title("Povprečno število nesreč - povzročitelj alkoholizirani voznik", size=16)
    plt.xlabel("Odstotek vinjenih povzročiteljev", size=13)
    plt.ylabel("Upravna enota storitve", size=13)
    
    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="VinjenaOseba", y="UpravnaEnotaStoritve", data=crashes_pivot,
                label="Alcohol-involved", color="b")
    
    ##################################################################################################################
    ##################################################################################################################
    #UPRAVNA ENOTA NORMALIZIRAJ GLEDE NA ŠTEVILO PREBIVALCEV
    ##################################################################################################################
    df_Nesrece_UE = df_vinjenaOseba[['stPN', 'Leto','UpravnaEnotaStoritve','VinjenaOseba']].copy()
    df_Nesrece_UE = df_Nesrece_UE.groupby(['UpravnaEnotaStoritve','Leto','VinjenaOseba']).agg(
            {
             'stPN':['nunique']
                    })
    df_Nesrece_UE.columns = ["_".join(x) for x in df_Nesrece_UE.columns.ravel()]
    df_Nesrece_UE.reset_index(inplace=True)
    df_Nesrece_UE = df_Nesrece_UE.pivot_table(values=['stPN_nunique'], 
                                                        index=['UpravnaEnotaStoritve','Leto'], 
                                                        columns=['VinjenaOseba'])
    
    df_Nesrece_UE.columns = [ 'NiVinjenaOseba', 'VinjenaOseba' ]
    df_Nesrece_UE = df_Nesrece_UE.reset_index()
    df_Nesrece_UE['Skupaj'] = df_Nesrece_UE['NiVinjenaOseba'] + df_Nesrece_UE['VinjenaOseba']
    df_Nesrece_UE = pd.merge(df_Nesrece_UE, df_Prebivalstvo, left_on=['Leto','UpravnaEnotaStoritve'], right_on = ['Leto','UpravnaEnota'], how='left')
    df_Nesrece_UE = df_Nesrece_UE[df_Nesrece_UE["Leto"] == 2018]
    #df_Nesrece_UE = df_Nesrece_UE.replace({'SteviloPrebivalcev': {0: np.nan}}).ffill()
    
    
    df_Nesrece_UE['NormaliziraneNesrece'] = df_Nesrece_UE['Skupaj'] /df_Nesrece_UE['SteviloPrebivalcev']
    df_Nesrece_UE['NormaliziraneNesreceVinjeni'] = df_Nesrece_UE['VinjenaOseba'] /df_Nesrece_UE['SteviloPrebivalcev']
    df_Nesrece_UE = df_Nesrece_UE.sort_values('NormaliziraneNesrece', ascending=False)
    
    sns.set_color_codes("pastel")
    plt.figure(figsize=(16,10))
    plt.subplot(1, 2, 1)
    
    sns.barplot(x="NormaliziraneNesrece", y="UpravnaEnotaStoritve", data=df_Nesrece_UE.head(20),
                label="Total", color="b", ci=None)
    plt.title("Normalizirano število prometnih nesreč na prebivalca", size=14)
    plt.xlabel("Normaliziran delež prometnih nesreč", size=12)
    plt.ylabel("Upravna enota", size=12)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x="NormaliziraneNesreceVinjeni", y="UpravnaEnotaStoritve", data=df_Nesrece_UE.sort_values('NormaliziraneNesreceVinjeni', ascending=False).head(20),
                label="Total", color="r", ci=None)
    plt.title("Normalizirano število prometnih nesreč (Alkohol) na prebivalca", size=14)
    plt.xlabel("Normaliziran delež prometnih nesreč", size=12)
    plt.ylabel("", size=13)
    ##################################################################################################################
    ##################################################################################################################
    #VRSTA CESTE NASELJE
    ##################################################################################################################
    crashes = df_vinjenaOseba[['stPN', 'Leto','VrstaCesteNaselja','VinjenaOseba']].copy()
    crashes = crashes.groupby(['VrstaCesteNaselja','Leto','VinjenaOseba']).agg(
            {
             'stPN':['nunique']
                    })
    crashes.columns = ["_".join(x) for x in crashes.columns.ravel()]
    crashes.reset_index(inplace=True)
    crashes_pivot = crashes.pivot_table(values=['stPN_nunique'], 
                                                        index=['VrstaCesteNaselja','Leto'], 
                                                        columns=['VinjenaOseba'])
    
    crashes_pivot.columns = [ 'NiVinjenaOseba', 'VinjenaOseba' ]
    crashes_pivot = crashes_pivot.reset_index()
    crashes_pivot['Skupaj'] = crashes_pivot['NiVinjenaOseba'] + crashes_pivot['VinjenaOseba']
    crashes_pivot['AC'] = np.where(((crashes_pivot['VrstaCesteNaselja']=='AVTOCESTA') | ( crashes_pivot['VrstaCesteNaselja']=='HITRA CESTA')), 'HITRE CESTE', 'OSTALO')
    #crashes_pivot['OdstotekVinjenihPovzročiteljev'] = crashes_pivot['Skupaj'] /crashes_pivot['VinjenaOseba']
    # Plot the total crashes
    #crashes_pivot = crashes_pivot.query('OdstotekVinjenihPovzročiteljev > 10').sort_values('OdstotekVinjenihPovzročiteljev', ascending=False)
    
    sns.set(font_scale=1.0)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 2)
    bplot = sns.barplot(x = "AC", y = "Skupaj", hue = "Leto", #palette="BuGn_d", 
                        data= crashes_pivot,estimator=sum)
    bplot.set_title("Število nesreč po vrsti ceste za zadnje 4 leta v Sloveniji", size=16)
    bplot.set_xlabel("Vrsta ceste", size=13)
    bplot.set_ylabel("Število prometnih nesreč")
    # Put the legend out of the figure
    bplt = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#zaženi to, če želiš dobiti mapo...
def HeatMap():
    #print(df1.dtypes)
    df1.isna().sum(axis = 0)
    
    print(df1['VremenskeOkoliscine'].unique())
    from folium.plugins import HeatMap
    #first, copy all data [all 2017 county accidents] to our map dataframe
    df_map = df1.copy()
    df_map = df_map.query("KlasifikacijaNesrece!='S SMRTNIM IZIDOM'")
    df_map = df_map.query("Leto>2017")
    #df_map = df1[df1['Total Killed']>= 1.0].copy()
    df_map['count']=1
    df_map[['newLong', 'newLat', 'count']].groupby(['newLong', 'newLat']).sum().sort_values('count', ascending=False).head(10)
    
    df_map.shape
    
    ##print(df1['KlasifikacijaNesrece'].unique())
    base_map = folium.Map(location=[46.051761, 14.493524], control_scale=True, zoom_start=9)
    #    base_map = generateBaseMap()
    base_map
    m = HeatMap(data=df_map[['newLong', 'newLat', 'count']].groupby(['newLong','newLat']).sum().reset_index().values.tolist(), radius=7, max_zoom=10).add_to(base_map)
    m.save('c:\_PERSONAL\FRI\PR\Projekt\smrtni izid 1.html')

    #https://gist.github.com/HariSan1/0245dca9ba3b32caf9b59ff81a4bd9b5
    
def Kolesarji():
    kolesar_df = df1[['stPN','KlasifikacijaNesrece','razred','UpravnaEnotaStoritve','DatumPN','Tema', 'PromentaKonica',
                       'UraPN','VNaselju','Lokacija','VrstaCesteNaselja','SifraCesteNaselja',
                       'TekstCesteNaselja','SifraOdsekaUlice','TekstOdsekaUlice',
                          'StacionazaDogodka','OpisKraja','VzrokNesrece','TipNesrece',
                          'VremenskeOkoliscine','StanjePrometa','StanjeVozisca','VrstaVozisca','newLong','newLat',
                          'DatumPNConverted','Dan','Leto','Mesec','MesecNaziv','DanVTednu','DanVTednuStevilka', 'VrstaUdelezenca']].copy()
    
    koleasar_df = kolesar_df[kolesar_df['VrstaUdelezenca'] == 'KOLESAR']
    
    
def generateBaseMap(default_location=[40.5397293,-74.6273494], default_zoom_start=12):
    base_map = folium.Map(location=[46.051761, 14.493524], control_scale=True, zoom_start=9)
    
#    base_map = generateBaseMap()
    base_map
    m = HeatMap(data=df_map[['newLong', 'newLat', 'count']].groupby(['newLong','newLat']).sum().reset_index().values.tolist(), radius=7, max_zoom=10).add_to(base_map)
    m.save('c:\_PERSONAL\FRI\PR\Projekt\heatmap1.html')
    
def PripravaPodatkovZaNapoved():
    #podatke je potrebno ustrezno pripraviti. 
    #Upoštevati je potrebno, da je 1 prometna nesreča lahko v 1 ali več vrsticah
    #To je odvisno od števila povzročiteljev + udeležencev prometne nesreče
    #za vsako prometno nesrečo je potrebno izračunati povprečno starost udeleženca, maximalno starost udeleženca
    #povprečno starost povzročitelja, #maximalno starost povzročitelja
     # set up data
    df1 = df1[df1['Leto']>2015]

    
    uniquePrometnaNesreca = df1[['stPN','KlasifikacijaNesrece','razred','UpravnaEnotaStoritve','DatumPN','Tema', 'PromentaKonica','SmrtniIzid',
                       'UraPN','VNaselju','Lokacija','VrstaCesteNaselja','SifraCesteNaselja',
                       'TekstCesteNaselja','SifraOdsekaUlice','TekstOdsekaUlice',
                          'StacionazaDogodka','OpisKraja','VzrokNesrece','TipNesrece',
                          'VremenskeOkoliscine','StanjePrometa','StanjeVozisca','VrstaVozisca','newLong','newLat',
                          'DatumPNConverted','Dan','Leto','Mesec','MesecNaziv','DanVTednu','DanVTednuStevilka']].copy()
    
#    uniquePrometnaNesreca['stPN'] = uniquePrometnaNesreca.apply(lambda row: str(uniquePrometnaNesreca.Leto) + 
#                                  str(uniquePrometnaNesreca.ZaporednaStevilkaPN), axis = 1) 
    
#    uniquePrometnaNesreca['stPN'] = int(str(uniquePrometnaNesreca['Leto']) + str(uniquePrometnaNesreca['ZaporednaStevilkaPN']))
#    uniquePrometnaNesreca['stPN'] = uniquePrometnaNesreca['Leto'].map(str) + "" + uniquePrometnaNesreca['ZaporednaStevilkaPN'].map(str)
#    uniquePrometnaNesreca['stPN'] = pd.to_numeric(uniquePrometnaNesreca['stPN'], errors='coerce')
    
    uniquePrometnaNesreca.drop_duplicates(subset=None, keep='first', inplace=True)
    uniquePrometnaNesreca[uniquePrometnaNesreca['stPN'].duplicated() == True]
    
#    uniquePrometnaNesreca.loc[uniquePrometnaNesreca['KlasifikacijaNesrece'] == 'Z MATERIALNO ŠKODO','razred'] = 'BREZ POŠKODB'
#    uniquePrometnaNesreca.loc[uniquePrometnaNesreca['KlasifikacijaNesrece'] == 'S SMRTNIM IZIDOM','razred'] = 'SMRTNI IZID'
#    uniquePrometnaNesreca.loc[uniquePrometnaNesreca['KlasifikacijaNesrece'] == 'Z LAŽJO TELESNO POŠKODBO','razred'] = 'TELESNA POŠKODBA'
#    uniquePrometnaNesreca.loc[uniquePrometnaNesreca['KlasifikacijaNesrece'] == 'S HUDO TELESNO POŠKODBO','razred'] = 'TELESNA POŠKODBA'
#    uniquePrometnaNesreca['razred'].unique()
    
    
    PrometneNesrecePovzrocitelj = df1[df1["Povzrocitelj"]=="POVZROČITELJ"]
    df_PN_Povzr_VozniskiStazVLetih = PrometneNesrecePovzrocitelj.groupby(['ZaporednaStevilkaPN'])['VozniskiStaz'].agg([pd.np.min, pd.np.max, pd.np.mean])
#    df_PN_Povzr_StPovzročiteljev = PrometneNesrecePovzrocitelj.groupby(['ZaporednaStevilkaPN'])['ZaporednaStevilkaOsebeVPN'].agg([pd.np.size])
    
    df_PN_Povzr = PrometneNesrecePovzrocitelj.groupby(['stPN']).agg(
            {'Starost':['min','max','mean'],
             'VozniskiStaz':['min','max','mean'],
             'ZaporednaStevilkaOsebeVPN':['size'],
             'VrednostAlkotesta':['mean','max']
                    })
    df_PN_Povzr.columns = ["_".join(x) for x in df_PN_Povzr.columns.ravel()]
    df_PN_Povzr.reset_index(inplace=True)
    df_PN_Povzr_New = pd.merge(uniquePrometnaNesreca, df_PN_Povzr, on='stPN', how='inner')
        
    
    PrometneNesreceUdelezenec = df1[df1["Povzrocitelj"]=="UDELEŽENEC"]
    df_PN_Udelez = PrometneNesreceUdelezenec.groupby(['stPN']).agg(
            {'Starost':['min','max','mean'],
             'VozniskiStaz':['min','max','mean'],
             'ZaporednaStevilkaOsebeVPN':['size'],
             'VrednostAlkotesta':['max','mean']
                    })
    df_PN_Udelez.columns = ["_".join(x) for x in df_PN_Udelez.columns.ravel()]
    df_PN_Udelez.reset_index(inplace=True)
    
    df_PN_Udelez_New = pd.merge(uniquePrometnaNesreca, df_PN_Udelez, on='stPN', how='inner')
    
    from scipy.stats import zscore
    df_PN_Povzr_New["age_zscore"] = zscore(df_PN_Povzr_New["Starost_max"])
    df_PN_Povzr_New["is_outlier"] = df_PN_Povzr_New["age_zscore"].apply(lambda x: x <= -2.5 or x >= 2.5)
    df_PN_Povzr_New[df_PN_Povzr_New["is_outlier"]]
    ageAndFare = df_PN_Povzr_New[["Starost_mean", "VozniskiStaz_mean"]]
    ageAndFare.plot.scatter(x = "Starost_mean", y = "VozniskiStaz_mean")
    
#    df_PN_Udelez_New = df_PN_Udelez_New.query("VremenskeOkoliscine=='DEŽEVNO'")
    
        
    #We first need to convert the categorical features into numerical values.
    #We can use Sklearn library to do that like this:
    #https://www.kaggle.com/madislemsalu/do-weather-conditions-predict-accident-severity/data
    # Label encoder
    from sklearn.preprocessing import LabelEncoder
    
    lblE = LabelEncoder()
    for i in df_PN_Povzr_New:
        if df_PN_Povzr_New[i].dtype == 'object':
            lblE.fit(df_PN_Povzr_New[i])
            df_PN_Povzr_New[i] = lblE.transform(df_PN_Povzr_New[i])
    
#    cols=['dan_v_tednu','UraPN','PrometnaKonica','VNaselju','VrstaCesteNaselja','VremenskeOkoliscine','StanjePrometa']
#    cols=['KlasifikacijaNesrece','UpravnaEnotaStoritve',
#                       'UraPN','VNaselju',
#                         'OpisKraja','VzrokNesrece','TipNesrece',
#                          'VremenskeOkoliscine','StanjePrometa','StanjeVozisca','VrstaVozisca',
#                          'DanVTednuStevilka', 'Starost_mean', 'VrednostAlkotesta_mean','ZaporednaStevilkaOsebeVPN_size']
#    
#    cols=['KlasifikacijaNesrece','UpravnaEnotaStoritve',
#                       'UraPN','VNaselju',
#                         'VzrokNesrece','TipNesrece',
#                          'StanjeVozisca','VrstaVozisca',
#                          'DanVTednuStevilka']
    
    cols=['razred','UraPN','VNaselju','StanjePrometa','StanjeVozisca','VrstaVozisca','VremenskeOkoliscine','PromentaKonica',
                          'DanVTednuStevilka','Starost_mean','VozniskiStaz_mean','UpravnaEnotaStoritve']
    #'VrstaCesteNaselja', 'Starost_mean', 
    
#    cols=['KlasifikacijaNesrece','UraPN','VNaselju','VrstaCesteNaselja',
#                         'OpisKraja','VzrokNesrece','TipNesrece',
#                          'StanjePrometa','StanjeVozisca','DanVTednuStevilka', 
#                          'Starost_mean', 'VrednostAlkotesta_mean']
#    
#    cols=['KlasifikacijaNesrece','UraPN','VNaselju',
#                         'VremenskeOkoliscine','StanjePrometa','StanjeVozisca','DanVTednuStevilka', 'VrednostAlkotesta_mean']
    
    df_PN_Povzr_New = df_PN_Povzr_New[cols]
    df_PN_Povzr_New.columns
    df_PN_Povzr_New.PromentaKonica.unique()
    
    df_PN_Povzr_New.replace(-1, np.nan, inplace=True) # -1 should be imputed to NaN to be recognized as missing in the next row
    df_PN_Povzr_New=df_PN_Povzr_New.dropna() # I drop all the rows with missing data once again
    df_PN_Povzr_New.shape
    
    
    sns.set(font_scale=1)
    corr = df_PN_Povzr_New.corr()
    cover= np.zeros_like(corr)
    cover[np.triu_indices_from(cover)]=True
    # Heatmap
    with sns.axes_style("white"):
        sns.heatmap(corr, mask=cover, square=True,cmap="RdBu", annot=True)
    
    import matplotlib.pyplot as plt
    corrmat = df_PN_Povzr_New.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10,9))
   #plot heat map
    g=sns.heatmap(df_PN_Povzr_New[top_corr_features].corr(),vmax=.8, square=True)
    
        
    
#    Y = df_PN_Povzr_New.SmrtniIzid.values
    Y = df_PN_Povzr_New.razred.values
    Y
    
    cols = df_PN_Povzr_New.shape[1]
    cols
    del df_PN_Povzr_New['UraPN']
    X = df_PN_Povzr_New.loc[:, df_PN_Povzr_New.columns != 'razred']
    X.columns
    X.shape
    
    #feature importance 
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,Y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    
    
    X_train1, X_test1,Y_train1,Y_test1 = train_test_split(X, Y, test_size=0.33, random_state=99)
    
    
    svc = SVC()
    svc.fit(X_train1, Y_train1)
    Y_pred = svc.predict(X_test1)
    acc_svc1 = round(svc.score(X_test1, Y_test1) * 100, 2)
    acc_svc1
    print_score(svc)
    #prvi 62.92
    
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train1, Y_train1)
    Y_pred = knn.predict(X_test1)
    acc_knn1 = round(knn.score(X_test1, Y_test1) * 100, 2)
    acc_knn1
    #prvi 55.84
    
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
    from sklearn.model_selection import cross_val_score
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train1, Y_train1)
    Y_pred = logreg.predict(X_test1)
    cm1=confusion_matrix(Y_test1, Y_pred)
    print(cm1)

    acc_log1 = round(logreg.score(X_train1, Y_train1) * 100, 2)
    acc_log1
    
    accuracies= cross_val_score(estimator=logreg, X=X_train1, y=Y_train1, cv=10)
    accuracies.mean()
    accuracies.std()
    print('accuracies: ',accuracies)
    print('mean: ',accuracies.mean())
    print('stdev: ',accuracies.std())
    
    print(classification_report(Y_test1, Y_pred))
  
    
    #62.91
    
    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train1, Y_train1)
    Y_pred = gaussian.predict(X_test1)
    acc_gaussian1 = round(gaussian.score(X_test1, Y_test1) * 100, 2)
    acc_gaussian1
    #57.77
    
    # Perceptron
#    perceptron = Perceptron()
#    perceptron.fit(X_train1, Y_train1)
#    Y_pred = perceptron.predict(X_test1)
#    acc_perceptron1 = round(perceptron.score(X_test1, Y_test1) * 100, 2)
#    acc_perceptron1
    
    # Linear SVC
    linear_svc = LinearSVC()
    linear_svc.fit(X_train1, Y_train1)
    Y_pred = linear_svc.predict(X_test1)
    acc_linear_svc1 = round(linear_svc.score(X_test1, Y_test1) * 100, 2)
    acc_linear_svc1
    #58.59
    
    # Stochastic Gradient Descent
    sgd = SGDClassifier()
    sgd.fit(X_train1, Y_train1)
    Y_pred = sgd.predict(X_test1)
    acc_sgd1 = round(sgd.score(X_test1, Y_test1) * 100, 2)
    acc_sgd1
    #62.6
    
    featurecols=['UraPN','VNaselju','StanjePrometa','StanjeVozisca','VrstaVozisca','VremenskeOkoliscine','PromentaKonica',
                          'DanVTednuStevilka']
    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train1, Y_train1)
    Y_pred = decision_tree.predict(X_test1)
    acc_decision_tree1 = round(decision_tree.score(X_test1, Y_test1) * 100, 2)
    acc_decision_tree1
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(Y_test1, Y_pred))  
    print(classification_report(Y_test1, Y_pred))  
    
    
   #    
#    from sklearn.tree import export_graphviz
#    from sklearn import tree
#    from IPython.display import SVG
#    from sklearn.tree import DecisionTreeClassifier, export_graphviz
#    from graphviz import Source
#    from IPython.display import display
#    import pydotplus
#    
#    dot_data = StringIO()
#    export_graphviz(decision_tree, out_file=dot_data,  
#                    filled=True, rounded=True,
#                    special_characters=True,feature_names = featurecols,class_names=['0','1'])
#    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#    graph.write_png('diabetes.png')
#    Image(graph.create_png())
    
#    
#    dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=featurecols)
#    graph = Source(dot_data) 
#    graph.render("name of file.bmp",view = True)
#    
#    graph = Source(tree.export_graphviz(decision_tree, out_file='tree1.dot'
#       , feature_names=X.columns, class_names=['0', '1'] 
#       , filled = True))
#    display(SVG(graph.pipe(format='svg')))
#    
#    # Export as dot file
#    export_graphviz(decision_tree, out_file='tree.dot', 
#                    feature_names = featurecols,
#                    class_names = ['0','1'],
#                    rounded = True, proportion = False, 
#                    precision = 2, filled = True)
#    
#    # Convert to png
#    import pydot
#    (graph,) = pydot.graph_from_dot_file('tree.dot')
#    graph.write_png('somefile.png')
#
#    import os
#    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#    os.system('dot -Tpng tree.dot -o random.png')
#    
#    from subprocess import call
#    call(['dot', '-Tpng', 'c:\_PERSONAL\FRI\PR\Projekt\tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#    
#    # Display in python
#    import matplotlib.pyplot as plt
#    plt.figure(figsize = (14, 18))
#    plt.imshow(plt.imread('tree.png'))
#    plt.axis('off');
#    plt.show();

    #59.54
    
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train1, Y_train1)
    Y_pred = random_forest.predict(X_test1)
    random_forest.score(X_train1, Y_train1)
    acc_random_forest1 = round(random_forest.score(X_test1, Y_test1) * 100, 2)
    acc_random_forest1
    
    #67.95
    
    print("Machine Learning algorithm scores without weather related conditions")
    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc1, acc_knn1, acc_log1, 
              acc_random_forest1, acc_gaussian1,  
              acc_sgd1, acc_linear_svc1, acc_decision_tree1]})
    models.sort_values(by='Score', ascending=False)

    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    prometne_df = prometne_df.rename_axis(None)
    prometne_df = prometne_df.groupby(['Leto','Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine'])['ZaporednaStevilkaPN'].nunique() 
    prometne_df = prometne_df.reset_index()[['Leto','Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine', 'ZaporednaStevilkaPN']]
    prometne_df.rename(columns={'ZaporednaStevilkaPN':'Število nesreč'}, 
                                inplace=True )
    
    stanjePrometa_df = prometne_df.groupby(['StanjePrometa'])['Število nesreč'].sum()
    stanjePrometa_df = stanjePrometa_df.reset_index()[['StanjePrometa','Število nesreč']]
    stanjePrometa_df.sort_values('Število nesreč', inplace=True)
    #prav tako je potrebno pripraviti ustrezni nabor podatkov, da bomo lahko primerjali ali prometna konica vpliva na 
    #resnost nesreče
    
    #kaj se dogaja ob vikendih in praznikih. Ali vpliva dan pred praznikom, dan po prazniku na prometno nesrečo
    #mogoče lahko uporabimo tudi praznike v sosednjih državah
def rmse(x,y): return np.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train1), Y_train1), 
           rmse(m.predict(X_test1), Y_test1),
           m.score(X_train1, Y_train1), 
           m.score(X_test1, Y_test1)]
    
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print (res)     

'''
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
Že rešeno spodaj. ne upoštevaj
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
def StarostPovzrociteljev(df1):
    # ----------------------------------------------------------------------
    # xx. del: Starost povzročitelja
    # ----------------------------------------------------------------------
    df = df1
    # set up data
    StarostPovzrociteljev_df = df[['Leto','Mesec','ZaporednaStevilkaPN','Povzrocitelj','Starost']].copy()
    StarostPovzrociteljev_df = StarostPovzrociteljev_df[StarostPovzrociteljev_df['Povzrocitelj'] == "Povzročitelj"]
    StarostPovzrociteljev_df.hist()
    trafficDataByYear_df = trafficDataByYear_df.groupby(['Leto'])['ZaporednaStevilkaPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
'''   
def PrometneNesrecePoLetih(df):
        # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po letih
    # ----------------------------------------------------------------------
    df = df1
    # set up data
    trafficDataByYear_df = df[['Leto','Mesec','KlasifikacijaNesrece','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByYear_df = trafficDataByYear_df.rename_axis(None)
#    trafficDataByYear_df = trafficDataByYear_df.groupby(['Leto'])['ZaporednaStevilkaPN'].nunique() 
#    trafficDataByYear_df = trafficDataByYear_df.reset_index()[['Leto','ZaporednaStevilkaPN']]
#    trafficDataByYear_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
#                                inplace=True )
#    trafficDataByYear_df.set_index('Leto')
    trafficDataByYear_df = trafficDataByYear_df.groupby(['Leto'])['stPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
    trafficDataByYear_df = trafficDataByYear_df.reset_index()[['Leto','Število oseb','Število nesreč']]
    trafficDataByYear_df.set_index('Leto')
    trafficDataByYear_df = trafficDataByYear_df.query("Leto > 2011 & Leto < 2019")
    
    plt.rcParams['figure.figsize'] = [13,5]
    plt.plot(trafficDataByYear_df['Leto'], 
            trafficDataByYear_df['Število nesreč'],'b-',label = 'Število nesreč'
           )
#    plt.plot(trafficDataByYear_df['Leto'], 
#            trafficDataByYear_df['Število oseb'],'r-',label = 'Število oseb'
#           )
    
#    plt.bar(trafficDataByYear_df['Leto'], 
#            trafficDataByYear_df['Stevilo nesrec']
#            ,align='center', linewidth=1, alpha=0.75
#            ,edgecolor='black')
    plt.title("Število nesreč po letih", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Število nesreč", size=13)
    plt.legend();
    plt.show()
    
#    Primer smrti
    trafficDataByYear_df = df[['Leto','Mesec','KlasifikacijaNesrece','stPN']].copy()
    smrt_df = trafficDataByYear_df.groupby(['Leto','KlasifikacijaNesrece'])['stPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
    smrt_df = smrt_df.reset_index()[['Leto','KlasifikacijaNesrece','Število oseb','Število nesreč']]
    smrt_df.set_index('Leto')
    smrt_df = smrt_df.query("Leto > 2011 & Leto < 2019 & KlasifikacijaNesrece=='S SMRTNIM IZIDOM'")
    
    plt.rcParams['figure.figsize'] = [13,5]
    plt.plot(smrt_df['Leto'], 
            smrt_df['Število nesreč'],'b-',label = 'Število nesreč'
           )
#    plt.plot(trafficDataByYear_df['Leto'], 
#            trafficDataByYear_df['Število oseb'],'r-',label = 'Število oseb'
#           )
    
#    plt.bar(trafficDataByYear_df['Leto'], 
#            trafficDataByYear_df['Stevilo nesrec']
#            ,align='center', linewidth=1, alpha=0.75
#            ,edgecolor='black')
    plt.title("Število prometnih nesreč s smrtnim izidom", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Število PN", size=13)
    plt.legend();
    plt.show()
    
    
#    plt.bar(trafficDataByYear_df['Leto'], 
#            trafficDataByYear_df['Stevilo nesrec']
#            ,align='center', linewidth=1, alpha=0.75
#            ,edgecolor='black')
    plt.rcParams['figure.figsize'] = [13,5]
    plt.plot(trafficDataByYear_df['Leto'], 
            trafficDataByYear_df['Število oseb'] / trafficDataByYear_df['Število nesreč'],'y-',label = 'Povprečno število oseb na prometno nesrečo '
           )
    plt.title("Povprečno število oseb udeleženih v prometnih nesrečah", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Povprečno število oseb", size=13)
    plt.legend();
    plt.show()
    
def PrometneNesrecePoMesecih(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po mesecih
    # ----------------------------------------------------------------------

    # set up data
    trafficDataByMonth_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'VNaselju','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    trafficDataByMonth_df = trafficDataByMonth_df.groupby(['Leto','Mesec'])['stPN'].nunique() 
    trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Leto','Mesec', 'stPN']]
    trafficDataByMonth_df.rename(columns={'stPN':'Stevilo nesrec'}, 
                                inplace=True )

    trafficDataByMonth_df= trafficDataByMonth_df.query("Leto == 2018")
    # add length column to allow normalization by month lengths
    # trafficDataByMonth_df['StDniVMesecu'] = trafficDataByMonth_df['DatumPN'].dt.daysinmonth
    trafficDataByMonth_df['StDniVMesecu'] = monthLength_list
    

    # normalizirano število nesreč po mesecih v letu 2018
    plt.rcParams['figure.figsize'] = [14,5]
    plt.bar(trafficDataByMonth_df['Mesec'], 
            trafficDataByMonth_df['Stevilo nesrec']/trafficDataByMonth_df['StDniVMesecu']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black', tick_label=month_list)
    

    plt.title("Normalizirano število nesreč po mesecih v letu 2018", size=16)
    plt.xlabel("Mesec", size=13)
    plt.ylabel("Normalizirano število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # change date column to month names
    trafficDataByMonth_df['Mesec'] = month_list

    # display results
    plt.show()
    trafficDataByMonth_df
    
def PrometneNesreceVMesecuNaselje(df1):
     # set up data
    trafficDataByMonth_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #število prometnih nesreč po mesecih v naselju in izven naselja
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    trafficDataByMonth_df = trafficDataByMonth_df.groupby(['Leto','Mesec', 'VNaselju'])['stPN'].nunique() 
    trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Leto','Mesec', 'VNaselju', 'stPN']]
    trafficDataByMonth_df.rename(columns={'stPN':'Stevilo nesrec'}, 
                                inplace=True )

    vNaselju_df = trafficDataByMonth_df[trafficDataByMonth_df["VNaselju"]=="DA"]
    vNaselju_df['StDniVMesecu'] = monthLength_list
    niVNaselju_df = trafficDataByMonth_df[trafficDataByMonth_df["VNaselju"]=="NE"]
    niVNaselju_df['StDniVMesecu'] = niVNaselju_df['Mesec'].dt.days_in_month
    # create bar plot
    #*********************************************************************************************************************************************************
    #*********************************************************************************************************************************************************
    # prikaži column chart, enega ob drugem
    #*********************************************************************************************************************************************************
    accidentSeverityByMonth_plt = sns.barplot(x='Mesec', y='Stevilo nesrec', 
                                          data=trafficDataByMonth_df, 
                                          hue='VNaselju', 
                                          edgecolor='black', alpha=0.75, 
                                          linewidth=1)
    
    for i in accidentSeverityByMonth_plt.patches:
        # get_width pulls left or right; get_y pushes up or down
        accidentSeverityByMonth_plt.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
    plt.show(accidentSeverityByMonth_plt)
    
def PrometneNesreceStanjePrometa(df1):
     # set up data
    prometne_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine','ZaporednaStevilkaPN','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    prometne_df = prometne_df.rename_axis(None)
    prometne_df = prometne_df.groupby(['Leto','Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine'])['stPN'].nunique() 
    prometne_df = prometne_df.reset_index()[['Leto','Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine', 'stPN']]
    prometne_df.rename(columns={'stPN':'Število nesreč'}, 
                                inplace=True )
    
    stanjePrometa_df = prometne_df.groupby(['StanjePrometa'])['Število nesreč'].sum()
    stanjePrometa_df = stanjePrometa_df.reset_index()[['StanjePrometa','Število nesreč']]
    stanjePrometa_df.sort_values('Število nesreč', inplace=True)
    
    #Stanje prometa
    ax = stanjePrometa_df.plot(kind='barh', figsize=(10,7),
                                        color="blue", fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title("Prometne nesreče glede na stanje prometa", fontsize=18)
    ax.set_xlabel("Prometne nesreče po tipu nesreče", fontsize=13);
    ax.set_ylabel("Število prometnih nesreč", fontsize=13);
    ax.set_yticklabels(stanjePrometa_df['StanjePrometa']);

    # create a list to collect the plt.patches data
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
        
def PrometneNesreceStanjePrometa(df1):
     # set up data
    prometne_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','StanjePrometa','VrstaUdelezenca','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    prometne_df = prometne_df.rename_axis(None)
    prometne_df = prometne_df.groupby(['Leto','Mesec', 'VNaselju','VrstaUdelezenca','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine'])['stPN'].nunique() 
    prometne_df = prometne_df.reset_index()[['Leto','Mesec', 'VrstaUdelezenca','VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine', 'stPN']]
    prometne_df.rename(columns={'stPN':'Število nesreč'}, 
                                inplace=True )
    
    vu_df = prometne_df.groupby(['VrstaUdelezenca'])['Število nesreč'].sum()
    vu_df = vu_df.reset_index()[['VrstaUdelezenca','Število nesreč']]
    vu_df.sort_values('Število nesreč', inplace=True)
    
    #Stanje prometa
    ax = vu_df.plot(kind='barh', figsize=(10,7),
                                        color="blue", fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title("Prometne nesreče glede na vrsto udeleženca", fontsize=18)
    ax.set_xlabel("Prometne nesreče po vrsti udeleženca", fontsize=13);
    ax.set_ylabel("Število prometnih nesreč", fontsize=13);
    ax.set_yticklabels(vu_df['VrstaUdelezenca']);

#    vu_df.head()
    # create a list to collect the plt.patches data
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
        
        
def PrometneNesreceTipNesrece():
     # set up data
    df_TipNesrece = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','StanjePrometa','VrstaUdelezenca','StanjeVozisca','TipNesrece','VzrokNesrece','VremenskeOkoliscine','razred','stPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    df_TipNesrece = df_TipNesrece.rename_axis(None)
    prometne_df = df_TipNesrece.groupby(['Leto','Mesec', 'VNaselju','VrstaUdelezenca','StanjePrometa','StanjeVozisca','TipNesrece','VzrokNesrece','VremenskeOkoliscine', 'razred'])['stPN'].nunique() 
    df_TipNesrece = df_TipNesrece.reset_index()[['Leto','Mesec', 'VrstaUdelezenca','VNaselju','StanjePrometa','StanjeVozisca','TipNesrece','VzrokNesrece','razred','VremenskeOkoliscine', 'stPN']]
    df_TipNesrece.rename(columns={'stPN':'Število nesreč'}, 
                                inplace=True )
    
    df_TipNesrece.head()
    
    df_Tip_v = df_TipNesrece.groupby(['TipNesrece'])['Število nesreč'].nunique()
    df_Tip_v.head()
    df_Tip_v = df_Tip_v.reset_index()[['TipNesrece','Število nesreč']]
    df_Tip_v.sort_values('Število nesreč', inplace=True)
    
    #Stanje prometa
    ax = df_Tip_v.plot(kind='barh', figsize=(10,7),
                                        color="blue", fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title("Prometne nesreče glede na vrsto udeleženca", fontsize=18)
    ax.set_xlabel("Prometne nesreče po vrsti udeleženca", fontsize=13);
    ax.set_ylabel("Število prometnih nesreč", fontsize=13);
    ax.set_yticklabels(df_Tip_v['TipNesrece']);

#    vu_df.head()
    # create a list to collect the plt.patches data
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
    
def SteviloPrometnihNesrecUdelezenec(df1):
    # ----------------------------------------------------------------------
    # 1. del udeleženci: Število oseb udeleženih v prometnih nesrečah
    # ----------------------------------------------------------------------
    # set up data
    udelezenci_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','stPN']].copy()
    udelezenci_df['StDniVMesecu'] = udelezenci_df['DatumPNConverted'].dt.daysinmonth
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    udelezenci_df = udelezenci_df.rename_axis(None)
    udelezenci_df = udelezenci_df.groupby(['Leto','Mesec','StDniVMesecu','Povzrocitelj'])['stPN'].agg({'Število oseb':'size', 'col4':'count', 'Število nesreč':'nunique'})
    udelezenci_df = udelezenci_df.reset_index()
    udelezenci_df = udelezenci_df.query("Leto == 2018")
    
    
    udelezenci_df_plt = sns.barplot(x='Mesec', y='Število oseb', 
                                          data=udelezenci_df, 
                                          hue='Povzrocitelj', 
                                          edgecolor='black', alpha=0.75, 
                                          linewidth=1)
    plt.title("Število oseb v prometnih nesrečah po mesecih v letu 2018", size=16)
    plt.show()

    
    #-------------------------------------------------------------------------------------------------------------------
    #Pivotiranje tabele za ugotavljanje razlike med številom udeležencev in številom povzročiteljev
    #iz tega lahko sklepamo v katerem mesecu v letu je bilo največ sopotnikov v avtomobilih
    #-------------------------------------------------------------------------------------------------------------------
    
    udelezenci_df_pivot = udelezenci_df.pivot_table(values=['Število oseb'], 
                                                        index=['Mesec'], 
                                                        columns=['Povzrocitelj'])
    udelezenci_df_pivot.columns = [ 'Povzročitelj', 'Udeleženec' ]
    udelezenci_df_pivot = udelezenci_df_pivot.reset_index()
    # add length column to allow normalization by month lengths
    # trafficDataByMonth_df['StDniVMesecu'] = trafficDataByMonth_df['DatumPN'].dt.daysinmonth
    ##potrebno je dobiti število dni v mesecu
    #todo
    #
    udelezenci_df_pivot['StDniVMesecu'] = monthLength_list
    udelezenci_df_pivot['Mesec'] = month_list
#    udelezenci_df_pivot['RazlikaPovzročitelj'] =  udelezenci_df_pivot['Udeleženec'] + udelezenci_df_pivot['Ni podatka'] - udelezenci_df_pivot['Povzročitelj'] 
    udelezenci_df_pivot['RazlikaPovzročitelj'] =  udelezenci_df_pivot['Udeleženec']  - udelezenci_df_pivot['Povzročitelj'] 
    
    #-------------------------------------------------------------------------------------------------------------------
    #Mesec v katerem je največja razlika med udeleženci in povzročitelji
    #-------------------------------------------------------------------------------------------------------------------
    
    ax = udelezenci_df_pivot[['Mesec', 'RazlikaPovzročitelj']].plot(
    x='Mesec', linestyle='-', marker='o')
    plt.title("Razlika med število udeležencev in številom povzročiteljev v prometnih nesrečah")
    plt.show()

    udelezenci_df_plt = sns.barplot(x='Mesec', y='Število oseb', 
                                      data=udelezenci_df, 
                                      hue='Povzrocitelj', 
                                      edgecolor='black', alpha=0.75, 
                                      linewidth=1)
    
    plt.title("Število oseb udeleženih v prometni nesrečah", size=16)
    plt.show()
    
    #-------------------------------------------------------------------------------------------------------------------
    #Drugačen način prikaza bar chart
    #-------------------------------------------------------------------------------------------------------------------
    
    plt.rcParams['figure.figsize'] = [15,5]
    plt.bar(udelezenci_df_pivot['Mesec'], 
            udelezenci_df_pivot['RazlikaPovzročitelj']
            ,align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    

    plt.title("Normalizirano število nesreč po mesecih", size=16)
    plt.xlabel("Mesec", size=13)
    plt.ylabel("Normalizirano število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    plt.show()
    udelezenci_df_pivot
    
def StarostUdelezencevVPrometnihNesrecah(df1):
    # ----------------------------------------------------------------------
    # x. del povzročitelji: Starost udeležencev v prometnih nesrečah histogram
    # ----------------------------------------------------------------------
    # set up data
    starost_df1 = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','VrstaUdelezenca', 'Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN', 'stPN']].copy()
    
#    bins = [0, 16, 18, 21, 50, 65, np.inf]
#    names = ['<16', '16-18', '18-21','21-50', '50-65', '65+']
#    
#    starost_df['AgeRange'] = pd.cut(starost_df['Starost'], bins, labels=names)
    
    #********    ******** **************** ********
    #********    ******** **************** ********
    # use: distribucija starosti povzročiteljev prometnih nesreč
    #********    ******** **************** ********
    starost_povz_df_povz = starost_df1[starost_df["Povzrocitelj"]=="POVZROČITELJ"]
    #*******************************************************************
    #distribucija starosti povzročiteljev prometnih nesreč
    #*******************************************************************
    starost_povz_df_povz.hist(column='Starost', bins=50)
    plt.title("Distribucija starosti povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    #*******************************************************************
    #distribucija starosti povzročiteljev prometnih nesreč po vrsti udeleženca
    #*******************************************************************
    starost_povz_df_povz.VrstaUdelezenca.unique()
    list_of_values = ['VOZNIK OSEBNEGA AVTOMOBILA','KOLESAR','VOZNIK KOLESA Z MOTORJEM','VOZNIK MOTORNEGA KOLESA','VOZNIK TOVORNEGA VOZILA']
    starost_povz_df_povz = starost_povz_df_povz.query("VrstaUdelezenca in @list_of_values")
    starost_povz_df_povz.hist(column='Starost', bins=50)
    plt.title("Distribucija starosti povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    starost_povz_df_povz.hist(column='Starost', bins=50,by ='VrstaUdelezenca')
    
    starost_povz_df_hist = starost_povz_df_povz[['Starost', 'VrstaUdelezenca']].copy()
    starost_povz_df_hist.hist(column='Starost', bins=50,by ='VrstaUdelezenca')
    
    starost_povz_df_povz.hist(column='VozniskiStazVLetih', by ='VrstaUdelezenca', bins=30)
    plt.title("Distribucija vozniškega staleža povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    
    # plotting
    for i in (starost_povz_df_povz['VrstaUdelezenca'].unique()):
        fig, ax = plt.subplots(len(names),1)
        group_data = data.loc[data['Group']==i]
        for number, col_name in enumerate(names):
            ax[number].hist(data[col_name]);
            ax[number].set_title("Histogram for " + col_name)
        plt.tight_layout()
        plt.savefig(i+'.pdf') # provide a valid path
    
    
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')


    
    #preveri starosti. Če je starost manj kot 0 se zapisov ne upošteva, predvidevamo tudi, če je starost več kot 100 let 
    #tega ne upoštevamo
    

def PrometneNesrecePoDnevuVTednu(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po dnevih z odstopanjem
    # ----------------------------------------------------------------------

    # set up data
    trafficDataByWeel_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','stPN']].copy()
    trafficDataByWeel_df = trafficDataByWeel_df.query("Leto < 2019")
    #-------------------------------------------------------------------------------------------------------------------
    #priprava podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByWeel_df = trafficDataByWeel_df.rename_axis(None)
    trafficDataByWeel_df = trafficDataByWeel_df.groupby(['DanVTednu','DanVTednuStevilka','Leto'])['stPN'].agg({'Število oseb':'size', 'col4':'count', 'Število nesreč':'nunique'})
    #    trafficDataByWeel_df.rename(columns={'stPN':'Stevilo nesrec'}, 
#                                inplace=True )
    trafficDataByWeel_df = trafficDataByWeel_df.reset_index()
    
    trafficDataByWeel_df.sort_values('DanVTednuStevilka',inplace=True)
          
        
    #-------------------------------------------------------------------------------------------------------------------
    #priprava podatkov
    #-------------------------------------------------------------------------------------------------------------------
    plt.rcParams['figure.figsize'] = [15,5]
    plt.bar(trafficDataByWeel_df['DanVTednu'], 
            trafficDataByWeel_df['Število nesreč']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    plt.title("Število nesreč po dnevu v tednu", size=16)
    plt.xlabel("Dan v tednu", size=13)
    plt.ylabel("Število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    plt.show()
    trafficDataByWeel_df
    
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid") 
    sns.boxplot(x='DanVTednu', y='Število nesreč', data=trafficDataByWeel_df,palette=None,saturation=0.75, width=0.8,whis=1.5)
    sns.boxplot(x='DanVTednu', y='Število oseb', data=trafficDataByWeel_df,palette=None,saturation=0.75, width=0.8,whis=1.5)
    
    

def PrometneNesrecePoUriVDnevu(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po dnevu
    # ----------------------------------------------------------------------
    # set up data
    trafficDataByHour_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','UraPN','stPN', 'KlasifikacijaNesrece']].copy()
    
    
    trafficDataByHour_df = df_vinjenPovzroc[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','UraPN','stPN', 'KlasifikacijaNesrece']].copy()
    trafficDataByHour_df = trafficDataByHour_df[trafficDataByHour_df["Leto"]==2018]
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
#    trafficDataByHour_df['KlasifikacijaNesrece'].unique()
#    trafficDataByHour_df = trafficDataByHour_df[trafficDataByHour_df['KlasifikacijaNesrece']=='S SMRTNIM IZIDOM']
    trafficDataByHour_df = trafficDataByHour_df.rename_axis(None)
    trafficDataByHour_df = trafficDataByHour_df.groupby(['DanVTednu','DanVTednuStevilka','UraPN'])['stPN'].nunique() 
    trafficDataByHour_df = trafficDataByHour_df.reset_index()[['DanVTednu','DanVTednuStevilka','UraPN', 'stPN']]
    trafficDataByHour_df.rename(columns={'stPN':'Stevilo nesrec'}, 
                                inplace=True )

    trafficDataByHour_df.sort_values('UraPN',inplace=True)
    
#    trafficDataByHour_df = trafficDataByHour_df[['DanVTednu','Stevilo nesrec','UraPN']]
    
#    trafficDataByHour_df = trafficDataByHour_df[trafficDataByHour_df["DanVTednu"]=='Sobota']
    
    trafficDataByHour_df_pivot = trafficDataByHour_df.pivot_table(values=['Stevilo nesrec'], 
                                                        index=['UraPN'], 
                                                        columns=['DanVTednuStevilka'])
    trafficDataByHour_df_pivot = trafficDataByHour_df_pivot.fillna(0)
    trafficDataByHour_df_pivot.columns = ['Ponedeljek','Torek','Sreda','Četrtek','Petek','Sobota','Nedelja']
    
    x_axis = trafficDataByHour_df_pivot.index

    # Plot each weekday and assigning color to be consistent with previous charts
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, trafficDataByHour_df_pivot['Ponedeljek'], color='#e41a1c')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Torek'], color='#377eb8')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Sreda'], color='#4daf4a')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Četrtek'], color='#984ea3')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Petek'], color='#ff7f00')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Sobota'], color='#998ec3')
    plt.plot(x_axis, trafficDataByHour_df_pivot['Nedelja'], color='#542788')
    
    # Determine y-axis
    y_max = 700
    step = 30
    y_axis = np.arange(0, y_max+step, step)
    # format y-ticks as comma separated
    y_axis_fmt = ["{:,.0f}".format(y) for y in y_axis]
    # set y-axis limits
    plt.ylim(min(y_axis), max(y_axis))
    
    # Format axes ticks and labels
    plt.xticks(np.arange(len(x_axis)), x_axis, fontsize=13)
    plt.yticks(y_axis, y_axis_fmt, fontsize=13)
    plt.xlabel('Ura dneva', fontsize=13)
    plt.ylabel('Število prometnih nesreč', fontsize=13)
    
    plt.legend(fontsize=11, loc='upper left')
    plt.title("Število prometnih nesreč po dnevih in urah", fontsize=13)
    plt.show()

    #*****************************************************************************************
    #USE:Število prometnih nesreč po uri v dnevu
    #*****************************************************************************************
    trafficDataByHour_df_pivot.sortlevel(level=0, ascending=True, inplace=True)
    custom_style = {
        'font.family':'Segoe UI',
        'xtick.color': '#FFFFFF',
        'ytick.color': '#FFFFFF'}
    sns.set_style("darkgrid", rc=custom_style)
#    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16,12),  facecolor='#111111')
    sns.heatmap(trafficDataByHour_df_pivot, cmap='magma', cbar=False, annot=True,fmt='g',
                annot_kws={'size': 10, 'alpha': 0.25, 'family': 'Segoe UI', 'weight': 'light'})
    
    plt.title('Povprečno število prometnih nesreč po dnevu v tednu in uri', 
              fontsize=17,  color='white')
    plt.xlabel("Dan v tednu", color='white')
    plt.ylabel("Ura v dnevu", color='white')
    plt.savefig('nypd-heatmap-out.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150)


    # število nesreč glede na uro v dnevu
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = [15,5]
    ax.plot(trafficDataByHour_df['UraPN'], trafficDataByHour_df['Stevilo nesrec'], color = 'black')
    fig.autofmt_xdate()
    plt.show()
    
    plt.bar(trafficDataByHour_df['UraPN'], 
            trafficDataByHour_df['Stevilo nesrec']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    plt.title("Število nesreč po dnevu v tednu", size=16)
    plt.xlabel("Dan v tednu", size=13)
    plt.ylabel("Število nesreč", size=13)
    plt.legend(trafficDataByHour_df['DanVTednu'],loc='upper left')
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    plt.show()
    trafficDataByHour_df[trafficDataByHour_df["DanVTednu"]=='Sobota']
"""
----------------    --------------------------------------------------------------------------------
----------------    --------------------------------------------------------------------------------
----------------    --------------------------------------------------------------------------------

def PrometneNesrecePoMesecih(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po mesecih
    # ----------------------------------------------------------------------

    # set up data
    trafficDataByMonth_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaPN']].copy()
  #  trafficDataByMonth_df.index = trafficDataByMonth_df['DatumPN']
#    trafficDataByMonth_df.index = trafficDataByMonth_df['DatumPN']
    #trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    #-------------------------------------------------------------------------------------------------------------------
    testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    #test = trafficDataByMonth_df[(trafficDataByMonth_df["Dan"] == 2) & (trafficDataByMonth_df["Mesec"] == 1)]
    #test = trafficDataByMonth_df.groupby(['Leto','Mesec','Dan','DatumPN'])['ZaporednaStevilkaPN'].nunique() 
    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    trafficDataByMonth_df = trafficDataByMonth_df.groupby(['Leto','Mesec'])['ZaporednaStevilkaPN'].nunique() 
#    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
#        
#    trafficDataByMonth_df = pd.DataFrame(trafficDataByMonth_df.resample('M').nunique()['ZaporednaStevilkaPN'])
    #trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['DatumPN', 'ZaporednaStevilkaPN']]
    trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Leto','Mesec', 'ZaporednaStevilkaPN']]
    trafficDataByMonth_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
                                inplace=True )

    # add length column to allow normalization by month lengths
    # trafficDataByMonth_df['StDniVMesecu'] = trafficDataByMonth_df['DatumPN'].dt.daysinmonth
    trafficDataByMonth_df['StDniVMesecu'] = monthLength_list
#    trafficDataByMonth_df["DatumPN"] = pd.to_datetime(trafficDataByMonth_df["DatumPN"])

    # normalizirano število nesreč po mesecih v letu 2018
    plt.rcParams['figure.figsize'] = [15,5]
    plt.bar(trafficDataByMonth_df['Mesec'], 
            trafficDataByMonth_df['Stevilo nesrec']/trafficDataByMonth_df['StDniVMesecu']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black', tick_label=month_list)
    plt.title("Normalizirano število nesreč po mesecih", size=16)
    plt.xlabel("Mesec", size=13)
    plt.ylabel("Normalizirano število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # change date column to month names
    trafficDataByMonth_df['Mesec'] = month_list

    # display results
    plt.show()
    trafficDataByMonth_df
----------------    --------------------------------------------------------------------------------
----------------    --------------------------------------------------------------------------------
----------------    --------------------------------------------------------------------------------
"""
    
def PrometneNesrecePoKlasifikacijiNesrece(df1):
    # ----------------------------------------------------------------------
    #2. del: Število prometnih nesreč po klasifikaciji nesreče
    # ----------------------------------------------------------------------
    
    dfVzrokPrometneNesrece = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaPN', 'KlasifikacijaNesrece','stPN']].copy()
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.rename_axis(None)
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.groupby(['KlasifikacijaNesrece'])['stPN'].nunique()
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.reset_index()[['KlasifikacijaNesrece', 'stPN']]
    dfVzrokPrometneNesrece.rename(columns={'stPN':'Stevilo nesrec'}, 
                                inplace=True )
    
    dfVzrokPrometneNesrece.sort_values('Stevilo nesrec',inplace=True)
    
    
    ax = dfVzrokPrometneNesrece.plot(kind='barh', figsize=(10,7),
                                        color="blue", fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title("Prometne nesreče po klasifikaciji", fontsize=18)
    ax.set_xlabel("Število prometnih nesreč", fontsize=18);
    ax.set_ylabel("Klasifikacija prometne nesreče", fontsize=18);
    ax.set_yticklabels(dfVzrokPrometneNesrece['KlasifikacijaNesrece']);
        
    # create a list to collect the plt.patches data
    totals = []
    
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
        """
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()/total)*100, 2))+'%', fontsize=15,
                color='dimgrey')
        """
    
    
    # normalizirano število nesreč po mesecih v letu 2018
    """
    plt.rcParams['figure.figsize'] = [15,5]
    plt.barh(dfVzrokPrometneNesrece['KlasifikacijaNesrece'], 
            dfVzrokPrometneNesrece['Stevilo nesrec']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    plt.title("Klasifikacija prometne nesreče", size=16)
    plt.xlabel("Število prometnih nesreč", size=13)
    plt.ylabel("Klasifikacija prometne nesreče", size=13)
    plt.savefig('Images/nesrecePoKlasifikaciji.png')

    # display results
    plt.show()
    """
    
    
def VplivVremenskihOkoliščin(df1):
    # ----------------------------------------------------------------------
    #5. del: Vpliv uporabe varnostnega pasu na posledice pri udeležencih prometnih nesreč
    # ----------------------------------------------------------------------
    vreme_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'stPN', 'ZaporednaStevilkaOsebeVPN','Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN','PoskodbaUdelezenca','VremenskeOkoliscine','UporabaVarnostnegaPasu', 'VrstaUdelezenca']].copy()
    
    
    #pogledamo samo VOZNIK OSEBNEGA AVTOMOBILA
    vreme_df = vreme_df.groupby(['VremenskeOkoliscine'])['stPN'].nunique()
    vreme_df = vreme_df.reset_index()
    vreme_df.rename(columns={'stPN':'Število prometnih nesreč'}, 
                                inplace=True )
    
    plt.figure(figsize=(10,6))
    g = sns.barplot(x='VremenskeOkoliscine', y='Število prometnih nesreč', hue='VremenskeOkoliscine', data=vreme_df, 
                  edgecolor='black', alpha=1)
       
    plt.title("Prometne nesreče glede na vremenske okoliščine", size=16)    
    plt.ylabel("Število prometnih nesreč", size=13)
    plt.xlabel("Vremenske okoliščine", size=13)
    plt.show()

    
def VplivUporabeVarnostnegaPasu(df1):
    # ----------------------------------------------------------------------
    #5. del: Vpliv uporabe varnostnega pasu na posledice pri udeležencih prometnih nesreč
    # ----------------------------------------------------------------------
    uporabaVarPas_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN','PoskodbaUdelezenca','VrstaUdelezenca','UporabaVarnostnegaPasu']].copy()
    
    #pogledamo samo povzročitelje
    uporabaVarPas_df = uporabaVarPas_df[(uporabaVarPas_df["PoskodbaUdelezenca"]=="SMRT") | (uporabaVarPas_df["PoskodbaUdelezenca"]=="HUDA TELESNA POŠKODBA")]
    
    #pogledamo samo VOZNIK OSEBNEGA AVTOMOBILA
    uporabaVarPas_df = uporabaVarPas_df[(uporabaVarPas_df["VrstaUdelezenca"]=="VOZNIK OSEBNEGA AVTOMOBILA") |
            (uporabaVarPas_df["VrstaUdelezenca"]=="POTNIK") |
            (uporabaVarPas_df["VrstaUdelezenca"]=="VOZNIK TOVORNEGA VOZILA")]
    uporabaVarPas_df = uporabaVarPas_df.groupby(['UporabaVarnostnegaPasu','PoskodbaUdelezenca'])['ZaporednaStevilkaPN'].nunique()
    uporabaVarPas_df = uporabaVarPas_df.reset_index()
    uporabaVarPas_df.rename(columns={'ZaporednaStevilkaPN':'Število udeležencev'}, 
                                inplace=True )
    
    plt.figure(figsize=(10,6))
    g = sns.barplot(x='PoskodbaUdelezenca', y='Število udeležencev', hue='UporabaVarnostnegaPasu', data=uporabaVarPas_df, 
                  edgecolor='black', alpha=1)
       
    plt.title("Uporaba varnostnega pasu v nesrečah z smrtjo in hudo telesno poškodbo", size=16)    
    plt.ylabel("Število udeležencev", size=13)
    plt.xlabel("Poškodba udeleženca", size=13)
    plt.show()

#    plt.title("Število nesreč po dnevu v tednu", size=16)
#    plt.xlabel("Dan v tednu", size=13)
#    plt.ylabel("Število nesreč", size=13)
#    plt.legend(uporabaVarPas_df['PoskodbaUdelezenca'],loc='upper left')
#    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    
    
def PrometneNesrecePoTipuNesrece(df1):
    # ----------------------------------------------------------------------
    #3. del: Število prometnih nesreč po tipu nesreče
    # ----------------------------------------------------------------------
    dfTipNesrece = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaPN', 'TipNesrece']].copy()
    dfTipNesrece = dfTipNesrece.rename_axis(None)
    dfTipNesrece = dfTipNesrece.groupby(['TipNesrece'])['ZaporednaStevilkaPN'].nunique()
    dfTipNesrece = dfTipNesrece.reset_index()[['TipNesrece', 'ZaporednaStevilkaPN']]
    dfTipNesrece.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
                                inplace=True )
    
    dfTipNesrece.sort_values('Stevilo nesrec',inplace=True)
    
    DisplayHorizontalBarChart(dfTipNesrece,"Prometne nesreče po tipu nesreče","Število prometnih nesreč","Tip nesrece",dfTipNesrece['TipNesrece'])
    
#    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    BACKUP za prikaz bar charta
#    ax = dfTipNesrece.plot(kind='barh', figsize=(10,7),
#                                        color="blue", fontsize=13);
#    ax.set_alpha(0.8)
#    ax.set_title("Prometne nesreče po klasifikaciji", fontsize=18)
#    ax.set_xlabel("Število prometnih nesreč", fontsize=18);
#    ax.set_ylabel("Tip nesrece", fontsize=18);
#    ax.set_yticklabels(dfTipNesrece['TipNesrece']);
#        
#    # create a list to collect the plt.patches data
#    totals = []
#    
#    # find the values and append to list
#    for i in ax.patches:
#        totals.append(i.get_width())
#    
#    # set individual bar lables using above list
#    total = sum(totals)
#    
#    # set individual bar lables using above list
#    for i in ax.patches:
#        # get_width pulls left or right; get_y pushes up or down
#        ax.text(i.get_width()+.10, i.get_y()+.22, \
#                str(round((i.get_width()), 2))+'', fontsize=12,
#                color='dimgrey')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

def DisplayHorizontalBarChart(df, title, xlabel,ylabel, ysticks):
    ax = df.plot(kind='barh', figsize=(10,7),
                                        color="blue", fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18);
    ax.set_ylabel(ylabel, fontsize=18);
    ax.set_yticklabels(ysticks);
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.}".format(int(x))))
        
    # create a list to collect the plt.patches data
    totals = []
    
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.10, i.get_y()+.22, \
                str(round((i.get_width()), 2))+'', fontsize=12,
                color='dimgrey')
 

if __name__ == '__main__':
    os.chdir("c:/_PERSONAL/FRI/PR/Projekt/files")
#    fileName = 'PN2018'
    fileName = 'PrometneNesrece'
    #preberi prometne nesreče za leto 2018
    df1 = ReadFiles(fileName)
    #preberi vse prometne nesreče
    #to ne bomo uporabili, ker želimo zaenkrat pogledati kaj se je dogajalo v letu 2018
    df1 = ReadAllFiles()
    ReplaceValues(df1)
    CleanDataNullValues(df1)
    CleanDataReplaceAge(df1)
    PripraviPodatkeZaAnalizo(df1)
    
    df1.head()
    PrometneNesrecePoMesecih(df1)
    print('Over')

#test = zipfile.namelist()
#for line in zipfile.open(file).readlines():
#    print(line.decode('utf-8'))