**Borut Olenik**
----------------

**Analiza prometnih nesreč v Sloveniji**
===========================================

Projekt pri predmetu Podatkovno rudarjenje
------------------------------------------

MENTOR: doc. dr. Tomaž Curk

V projektni nalogi obravnavam problematiko prometnih nesreč v Sloveniji. V ta
namen analiziram podatke prometnih nesreč, ki sem jih pridobil na spletni strani
slovenske policije
<https://www.policija.si/o-slovenski-policiji/statistika/prometna-varnost>.
Osredotočil se bom na analizo samih prometnih nesreč, kot tudi na udeležence
prometnih nesreč. Iz analiz želim spoznati kateri dejavniki vplivajo na prometne
nesreče.

UVOD
====

V predlogu projektne naloge sem opisal podatke prometnih nesreč, ki sem jih
našel na spletni strani slovenske policije na naslovu
<https://www.policija.si/o-slovenski-policiji/statistika/prometna-varnost>.
Struktura podatkov, ki sem jih opisal ob predlogu projekta se je drastično
spremenila. V vmesnem obdobju so na strani slovenske policije spremenili
strukturo podatkov, zato sem moral analize, ki sem jih imel že pripravljene
spremeniti na novo strukturo podatkov.

OPIS PODATKOV
=============

**Opis spremembe podatkov prometnih nesreč**

Ob predlogu projekta so bile strukture podatkov prometnih nesreč v Sloveniji
podane v obliki 2 datotek – podatkovnih baz, ki so za vsako leto opisovale
prometne nesreče v 1 datoteki ter udeležence prometnih nesreč v drugi datoteki.

Nova struktura podatkov je sestavljena iz 1 datoteke za vsako leto. Datoteke so
v csv obliki, ločeni z podpičjem(;) ter poimenovane PN[leto prometne
nesreče].csv. Na spletni strani so shranjene v obliki zip datotek.

Nova struktura baze prometnih nesreč je opisana na koncu poročila.

**Število prebivalcev po upravnih enotah**

Zaradi boljšega razumevanja prometnih nesreč se mi zdi pomembno, da se poleg
samih nesreč preveri tudi druge podatkovne baze s pomočjo katerih dobimo boljši
občutek prometnih nesreč. Za ta namen sem iz Statističnega urada Republike
Slovenije pridobil podatke o številu prebivalcev po upravnih enotah. Podatke sem
pridobil na spletni strani
<https://pxweb.stat.si/pxweb/Dialog/varval.asp?ma=05C3002S&ti=&path=../Database/Dem_soc/05_prebivalstvo/10_stevilo_preb/15_05C30_prebivalstvo_upravne/&lang=2>

**Vremenski podatki**

Zaradi boljše analize prometnih nesreč me je zanimalo, ali imajo zunanji
dejavniki, kot je temperatura zraka, višina snežne podlage vpliv na prometno
nesrečo. V ta namen sem pridobil ARSO zgodovinske podatke o vremenu. Podatke sem
pridobil za nekaj pomembni mest ter jih shranil v CSV datoteki. Podatki so javno
dostopni na spletni strani <http://meteo.arso.gov.si/met/sl/archive/>.

PRIPRAVA PODATKOV
=================

Podatki prometnih nesreč v Sloveniji so shranjeni v različnih datotekah na
spletu. Zaradi optimizacije in možnosti pridobivanja podatkov v prihodnosti sem
pripravil proceduro, s pomočjo katere podatke pridobimo iz spleta. Za potrebe
analiz sem se odločil za prometne nesreče od leta 2010 naprej. Datoteke sem
ekstrahiral na lokalni disk. Za potrebe celovite analize jih združimo v eno
podatkovno bazo.

Pridobivanje podatkov
---------------------
```
for i in range(2010,2020,1):
path = "https://www.policija.si/baza/pn" + str(i) + ".zip"
\#path = "https://www.policija.si/baza/pn2010.zip"
resp = urlopen(path)
zipfile = ZipFile(BytesIO(resp.read()))
zipfile.extractall("./files")
```

Združevanje podatkov
--------------------
```
os.chdir("c:/_PERSONAL/FRI/PR/Projekt/files")
results = pd.DataFrame([])
for counter, file in enumerate(glob.glob("PN\*.csv")):
    namedf = pd.read_csv("./" +file,error_bad_lines=False,sep=';',encoding="ANSI")
    results = results.append(namedf)
    results.to_csv('./PrometneNesrece.csv')
```
Čiščenje podatkov
-----------------

Ker je podatkovna baza prometnih nesreč pripravljena v obliki, ki ni najbolj
primerna za napredne analize je potrebno podatke ustrezno prečistiti.

**Štetje prometnih nesreč**

Izziv, ki sem ga zasledil je bila skupna podatkovna baza za nesreče in
udeležence nesreč. V ta namen je potrebno ob analizi prometnih nesreč – štetje
prometnih nesreč upoštevati razlikovalno štetje prometnih nesreč po stolpcu
*ZaporednaStevilkaPN.* V primeru analize udeležencev prometnih nesreč ni potrebe
po razlikovalnem štetju, temveč se uporabi normalno štetje.

**Ureditev datumskih atributov**

Datum prometne nesreče je v obliki DD.MM.YYYY (02.03.2018). To je format, ki ni
primeren za analizo, zato sem spremenil format datuma v YYYY-MM-DD. Poleg tega
sem datum razdelil na dodatne atribute: leto, mesec, mesec naziv, dan, dan v
tednu, dan v tednu številka.

**Ureditev starostnih razredov udeležencev prometnih nesreč (bins)**

Starostne razrede udeležencev prometnih nesreč sem razdelil v razrede s pomočjo
katerih bom lažje analiziral kako vpliva starost povzročitelja na prometno
nesrečo. Podatke sem razdelil v naslednje razrede: '\<16', '16-18',
'18-21','21-50', '50-65', '65+'

**Ureditev razredov za stalež vozniškega dovoljenja**

Ali stalež vozniškega dovoljenja vpliva na prometno nesrečo? S pomočjo razredov
staležev vozniškega dovoljenja sem segmentiral udeležence oz. povzročitelje
prometnih nesreč v 6 segmentov: '\<2', '2-5', '5-10', '10-20', '20-30', '30+'

**Ureditev razreda državljanstev**

Pri analizi prometnih nesreč želim primerjati koliko prometnih nesreč povzročijo
slovenski državljani in koliko prometnih nesreč povzročijo tuji državljani.

```
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
```

Glavne ugotovitve
=================

Podatkovne ugotovitve
---------------------

Pri analizi podatkov sem ugotovil, da obstaja množica podatkov, kjer je starost
povzročitelja oz. udeleženca prometne nesreče manj kot 0 let, zato sem pri
analizi udeležencev prometne nesreče take podatke odstranil.

Prav tako sem pri analizi staleža vozniškega dovoljenja odstranil podatke, kjer
je stalež vozniškega dovoljenja večji od starosti osebe, ker je to napaka v podatkih
```Starost osebe – Stalež vozniškega dovoljenja < 0```

Razdelitev prometnih nesreč po letih
------------------------------------

Iz analize vidimo, da je število prometnih nesreč do leta 2017 bilo v upadu.
Medtem, ko je leta 2018 število prometnih nesreč naraslo. Zanimivo pa je
dejstvo, da se število prometnih nesreč s smrtnim izidom iz leta v leto niža.
Velik vpliva na to imajo najbrž preventivne akcije, dobra osveščenost ljudi.
```
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
```

Povprečno število oseb v prometnih nesrečah po letih
----------------------------------------------------

Iz analize povprečnega števila oseb udeleženih v prometnih nesrečah lahko
sklepamo, da so v prometnih nesrečah v povprečju udeležene manj kot 2 osebi.
Največje povprečno število udeleženih oseb v prometnih nesrečah je bilo leta
2015. Za podrobno analizo povprečnega števila oseb bom v nadaljevanju projekta
razdelil osebe po vrsti udeleženca.
```
plt.rcParams['figure.figsize'] = [13,5]
    plt.plot(trafficDataByYear_df['Leto'], 
            trafficDataByYear_df['Število oseb'] / trafficDataByYear_df['Število nesreč'],'y-',label = 'Povprečno število oseb na prometno nesrečo '
           )
    plt.title("Povprečno število oseb udeleženih v prometnih nesrečah", size=16)
    plt.xlabel("Leto", size=13)
    plt.ylabel("Povprečno število oseb", size=13)
    plt.legend();
    plt.show()
```


Distribucija starosti povzročiteljev prometnih nesreč
-----------------------------------------------------

Analiza starosti povzročiteljev prometnih nesreč kaže na to, da največ prometnih
nesreč povzročijo osebe v zgodnjih 20 letih. To so osebe, ki imajo kratek
vozniški stalež. Torej neizkušeni vozniki, ki v večini primerov precenjujejo
svoje sposobnosti.
```
  starost_df = df1
    starost_df = starost_df[['DatumPNConverted', 'Leto', 'Dan', 'Mesec', 'ZaporednaStevilkaOsebeVPN','VrstaUdelezenca', 'Povzrocitelj','Starost', 
                         'VozniskiStazVLetih','Spol','ZaporednaStevilkaPN']].copy()
    
    bins = [0, 16, 18, 21, 50, 65, np.inf]
    names = ['<16', '16-18', '18-21','21-50', '50-65', '65+']
    
    starost_df['AgeRange'] = pd.cut(starost_df['Starost'], bins, labels=names)
        
    starost_povz_df = starost_df[starost_df["Povzrocitelj"]=="POVZROČITELJ"]
    
    starost_povz_df.hist(column='Starost', bins=50)
    plt.title("Distribucija starosti povrzočiteljev prometnih nesreč", size=16)
    plt.xlabel("Starost", size=13)
    plt.ylabel("Število prometnih nesreč", size=13)
```
Struktura podatkov
==================

Struktura podatkovne baze prometnih nesreč
------------------------------------------

Nova struktura baze prometnih nesreč (PN):

-   številka za štetje in ločevanje posamezne prometne nesreče

-   klasifikacija nesreče glede na posledice (Izračuna se avtomatično glede na
    najhujšo posledico pri udeležencih v prometni nesreči)

-   upravna enota, na območju katere se je zgodila prometna nesreča

-   datum nesreče (format: dd.mm.llll) 

-   ura nesreče (format: hh) 

-   indikator ali se je nesreča zgodila v naselju (D) ali izven (N)

-   lokacija nesreče

-   vrsta ceste ali naselja na kateri je prišlo do nesreče

-   oznaka ceste ali šifra naselja kjer je prišlo do nesreče

-   tekst ceste ali naselja, kjer je prišlo do nesreče

-   oznaka odseka ceste ali šifra ulice, kjer je prišlo do nesreče

-   tekst odseka ali ulice, kjer je prišlo do nesreče

-   točna stacionaža ali hišna številka, kjer je prišlo do nesreče

-   opis prizorišča nesreče

-   glavni vzrok nesreče

-   tip nesreče

-   vremenske okoliščine v času nesreče

-   stanje prometa v času nesreče

-   stanje vozišča v času nesreče

-   stanje površine vozišča v času nesreče

-   Geo Koordinata X (Gauß-Krüger-jev koordinatni sistem)

-   Geo Koordinata Y (Gauß-Krüger-jev koordinatni sistem)

-   številka za štetje in ločevanje oseb, udeleženih v prometnih nesrečah

-   kot kaj nastopa oseba v prometni nesreči

-   starost osebe (LL)

-   spol

-   upravna enota stalnega prebivališča

-   državljanstvo osebe

-   poškodba osebe

-   vrsta udeleženca v prometu

-   ali je oseba uporabljala varnostni pas ali čelado (polje se interpretira v
    odvisnosti od vrste udeleženca) (Da/Ne)

-   vozniški staž osebe za kategorijo, ki jo potrebuje glede na vrsto udeleženca
    v prometu (LL)

-   vozniški staž osebe za kategorijo, ki jo potrebuje glede na vrsto udeleženca
    v prometu (MM)

-   vrednost alkotesta za osebo, če je bil opravljen (n.nn)

-   vrednost strokovnega pregleda za osebo, če je bil odrejen in so rezultati že
    znani (n.nn)
