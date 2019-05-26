from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import fnmatch,os
import pandas as pd
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import pyproj as proj
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

    AgeMoreThanZero = df1[(df1['Starost'] > 0)]
    return AgeMoreThanZero
    
def ReplaceValues(df1):
    df1['DatumPNConverted'] = pd.to_datetime(df1['DatumPN'], format='%d.%m.%Y')
    #df1['DatumPN'] = df1['DatumPN'].map(lambda date_string: datetime.strptime(date_string, "%d.%m.%Y"))

def ReplaceDataStructure(df1):
    df1 = df
    ReplaceValues(df1)
    df1['ZaporednaStevilkaPN'] = pd.to_numeric(df1['ZaporednaStevilkaPN'], errors='coerce')
    df1['Dan'] = df1['DatumPNConverted'].dt.day
    df1['Leto'] = df1['DatumPNConverted'].dt.year
    df1['Mesec'] = df1['DatumPNConverted'].dt.month
    df1['MesecNaziv'] = df1['Mesec'].replace(monthName, regex=True)
    df1['DanVTednu'] = df1['DatumPNConverted'].dt.day_name()
    df1['DanVTednuStevilka'] = df1['DanVTednu'].replace(dayNumber, regex=True)
    df1['DanVTednu'] = df1['DanVTednu'].replace(day, regex=True)
    df1['DrzavljanstvoSkupina'] = np.where((df1['Drzavljanstvo']!='SLOVENIJA') & (df1['Drzavljanstvo']!='NEZNANO'), 'TUJEC', df1['Drzavljanstvo'])
    df1['DrzavljanstvoSkupina']  = df1['DrzavljanstvoSkupina'].astype('category')
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
    
    df1 = df1.join(df1.apply(convertCoords, axis=1))
    
#    
#def Maps():
#    import plotly
#    plotly.tools.set_config_file(world_readable=True)
##    plotly.offline.init_notebook_mode(connected=True)
#    plotly.tools.set_credentials_file(username='BorutOlenik', api_key='FRBZkKnJ1oCH5dywVO4H')
#    
#    import plotly.plotly as py
#    scl = [[0,'#5D56D3'], [0.5,'#7CD96E'], [1,'#CC655B']]
#    
#    data = [dict(type = 'scattergeo',
#                 lon = df1['newLong'],
#                 lat = df1['newLat'],
#                 mode = 'markers',
#                 marker = dict(
#                     size = 1,
#                     opacity = 0.75,
#                     reversescale = True,
#                     autocolorscale = False,
#                     symbol = 'circle',
#                     colorscale = scl,
#                     color = '#ff0000',
#                     cmax = 3,
#                     colorbar=dict(
#                         title='KlasifikacijaNesrece')))]
#    
#    layout = dict(title = '<b>2014 Great Britain & Wales Traffic Accidents</b>',
#                  width=1000,
#                  height=1000,
#                  geo = dict(scope = 'europe',
#                             projection=dict(type='eckert4'),
#                             lonaxis = dict(showgrid = True,
#                                            gridwidth = 0.5,
#                                            range= [-6, 2.59],
#                                            gridcolor='#000000',
#                                            dtick = 5),
#                             lataxis = dict(showgrid = True,
#                                            gridwidth = 0.5,
#                                            range = [49.48, 56],
#                                            gridcolor ='#000000',
#                                            dtick = 5),
#                showland = True,
#                landcolor = 'aliceblue',
#                subunitcolor = '#E5E5E5',
#                countrycolor = '#000000',
#            ))
#    
#    # create figure
#    fig = dict(data=data, layout=layout)
#    py.image.save_as(fig, filename='./2014 Traffic Accidents.png')
#    
#    # display plot
#    py.image.ishow(fig)

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
    trafficDataByYear_df = df[['Leto','Mesec','KlasifikacijaNesrece','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByYear_df = trafficDataByYear_df.rename_axis(None)
#    trafficDataByYear_df = trafficDataByYear_df.groupby(['Leto'])['ZaporednaStevilkaPN'].nunique() 
#    trafficDataByYear_df = trafficDataByYear_df.reset_index()[['Leto','ZaporednaStevilkaPN']]
#    trafficDataByYear_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
#                                inplace=True )
#    trafficDataByYear_df.set_index('Leto')
    trafficDataByYear_df = trafficDataByYear_df.groupby(['Leto'])['ZaporednaStevilkaPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
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
    
    smrt_df = trafficDataByYear_df.groupby(['Leto','KlasifikacijaNesrece'])['ZaporednaStevilkaPN'].agg({'Število oseb':'size', 'Število nesreč':'nunique'})
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
    trafficDataByMonth_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'VNaselju','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    trafficDataByMonth_df = trafficDataByMonth_df.groupby(['Leto','Mesec'])['ZaporednaStevilkaPN'].nunique() 
    trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Leto','Mesec', 'ZaporednaStevilkaPN']]
    trafficDataByMonth_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
                                inplace=True )

    trafficDataByMonth_df= trafficDataByMonth_df.query("Leto == 2018")
    # add length column to allow normalization by month lengths
    # trafficDataByMonth_df['StDniVMesecu'] = trafficDataByMonth_df['DatumPN'].dt.daysinmonth
    trafficDataByMonth_df['StDniVMesecu'] = monthLength_list
    

    # normalizirano število nesreč po mesecih v letu 2018
    plt.rcParams['figure.figsize'] = [13,5]
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
    trafficDataByMonth_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByMonth_df = trafficDataByMonth_df.rename_axis(None)
    trafficDataByMonth_df = trafficDataByMonth_df.groupby(['Leto','Mesec', 'VNaselju'])['ZaporednaStevilkaPN'].nunique() 
    trafficDataByMonth_df = trafficDataByMonth_df.reset_index()[['Leto','Mesec', 'VNaselju', 'ZaporednaStevilkaPN']]
    trafficDataByMonth_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
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
    plt.show(accidentSeverityByMonth_plt)
    
def PrometneNesreceStanjePrometa(df1):
     # set up data
    prometne_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine','ZaporednaStevilkaPN']].copy()
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
    prometne_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'VNaselju','StanjePrometa','VrstaUdelezenca','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    prometne_df = prometne_df.rename_axis(None)
    prometne_df = prometne_df.groupby(['Leto','Mesec', 'VNaselju','VrstaUdelezenca','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine'])['ZaporednaStevilkaPN'].nunique() 
    prometne_df = prometne_df.reset_index()[['Leto','Mesec', 'VrstaUdelezenca','VNaselju','StanjePrometa','StanjeVozisca','OpisKraja','VzrokNesrece','VremenskeOkoliscine', 'ZaporednaStevilkaPN']]
    prometne_df.rename(columns={'ZaporednaStevilkaPN':'Število nesreč'}, 
                                inplace=True )
    
    stanjePrometa_df = prometne_df.groupby(['VrstaUdelezenca'])['Število nesreč'].sum()
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
    
def SteviloPrometnihNesrecUdelezenec(df1):
    # ----------------------------------------------------------------------
    # 1. del udeleženci: Število oseb udeleženih v prometnih nesrečah
    # ----------------------------------------------------------------------
    # set up data
    udelezenci_df = df1[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN']].copy()
    udelezenci_df['StDniVMesecu'] = udelezenci_df['DatumPNConverted'].dt.daysinmonth
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    udelezenci_df = udelezenci_df.rename_axis(None)
    udelezenci_df = udelezenci_df.groupby(['Leto','Mesec','StDniVMesecu','Povzrocitelj'])['ZaporednaStevilkaPN'].agg({'Število oseb':'size', 'col4':'count', 'Število nesreč':'nunique'})
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
    starost_df = df1
    starost_df = starost_df[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','VrstaUdelezenca', 'Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN']].copy()
    
#    bins = [0, 16, 18, 21, 50, 65, np.inf]
#    names = ['<16', '16-18', '18-21','21-50', '50-65', '65+']
#    
#    starost_df['AgeRange'] = pd.cut(starost_df['Starost'], bins, labels=names)
        
    starost_povz_df = starost_df[starost_df["Povzrocitelj"]=="POVZROČITELJ"]
    
    starost_povz_df.hist(column='Starost', bins=50)
    plt.title("Distribucija starosti povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    list_of_values = ['VOZNIK OSEBNEGA AVTOMOBILA','KOLESAR','VOZNIK KOLESA Z MOTORJEM']
    starost_povz_df = starost_povz_df.query("VrstaUdelezenca in @list_of_values")
    starost_povz_df.hist(column='Starost', bins=50)
    plt.title("Distribucija starosti povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    starost_povz_df.hist(column='Starost', bins=50,by ='VrstaUdelezenca')
    
    starost_povz_df_hist = starost_povz_df[['Starost', 'VrstaUdelezenca']].copy()
    starost_povz_df_hist.hist(by ='VrstaUdelezenca')
    starost_povz_df.hist(column='VozniskiStazVLetih', by ='VrstaUdelezenca', bins=30)
    plt.title("Distribucija vozniškega staleža povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
    
    
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')


    
    #preveri starosti. Če je starost manj kot 0 se zapisov ne upošteva, predvidevamo tudi, če je starost več kot 100 let 
    #tega ne upoštevamo
    

def PrometneNesrecePoDnevuVTednu(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po mesecih
    # ----------------------------------------------------------------------

    # set up data
    trafficDataByWeel_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByWeel_df = trafficDataByWeel_df.rename_axis(None)
    trafficDataByWeel_df = trafficDataByWeel_df.groupby(['DanVTednu','DanVTednuStevilka'])['ZaporednaStevilkaPN'].nunique() 
    trafficDataByWeel_df = trafficDataByWeel_df.reset_index()[['DanVTednu','DanVTednuStevilka', 'ZaporednaStevilkaPN']]
    trafficDataByWeel_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
                                inplace=True )

    trafficDataByWeel_df.sort_values('DanVTednuStevilka',inplace=True)
    
    # normalizirano število nesreč po mesecih v letu 2018
    plt.rcParams['figure.figsize'] = [15,5]
    plt.bar(trafficDataByWeel_df['DanVTednu'], 
            trafficDataByWeel_df['Stevilo nesrec']
            ,  align='center', linewidth=1, alpha=0.75
            ,edgecolor='black')
    plt.title("Število nesreč po dnevu v tednu", size=16)
    plt.xlabel("Dan v tednu", size=13)
    plt.ylabel("Število nesreč", size=13)
    plt.savefig('Images/normalizedAccidentByMonth.png')

    # display results
    plt.show()
    trafficDataByWeel_df
    

def PrometneNesrecePoUriVDnevu(df1):
    # ----------------------------------------------------------------------
    # 1. del: Število prometnih nesreč po mesecih
    # ----------------------------------------------------------------------

    # set up data
    trafficDataByHour_df = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'DanVTednu','DanVTednuStevilka','UraPN','ZaporednaStevilkaPN']].copy()
    #-------------------------------------------------------------------------------------------------------------------
    #testiranje modela in pravilnost podatkov
    #-------------------------------------------------------------------------------------------------------------------
    trafficDataByHour_df = trafficDataByHour_df.rename_axis(None)
    trafficDataByHour_df = trafficDataByHour_df.groupby(['DanVTednu','DanVTednuStevilka','UraPN'])['ZaporednaStevilkaPN'].nunique() 
    trafficDataByHour_df = trafficDataByHour_df.reset_index()[['DanVTednu','DanVTednuStevilka','UraPN', 'ZaporednaStevilkaPN']]
    trafficDataByHour_df.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
                                inplace=True )

    trafficDataByHour_df.sort_values('UraPN',inplace=True)
    
    trafficDataByHour_df_pivot = trafficDataByHour_df.pivot_table(values=['Stevilo nesrec'], 
                                                        index=['UraPN'], 
                                                        columns=['DanVTednu'])
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
    y_max = 350
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


    # normalizirano število nesreč po mesecih v letu 2018
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
    trafficDataByHour_df
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
    
    dfVzrokPrometneNesrece = df1[['DatumPN', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaPN', 'KlasifikacijaNesrece']].copy()
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.rename_axis(None)
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.groupby(['KlasifikacijaNesrece'])['ZaporednaStevilkaPN'].nunique()
    dfVzrokPrometneNesrece = dfVzrokPrometneNesrece.reset_index()[['KlasifikacijaNesrece', 'ZaporednaStevilkaPN']]
    dfVzrokPrometneNesrece.rename(columns={'ZaporednaStevilkaPN':'Stevilo nesrec'}, 
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
 



#test = zipfile.namelist()
#for line in zipfile.open(file).readlines():
#    print(line.decode('utf-8'))