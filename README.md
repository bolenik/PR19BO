Borut Olenik
Analiza prometnih nesreč v Sloveniji
Projekt pri predmetu Podatkovno rudarjenje
MENTOR: doc. dr. Tomaž Curk

V projektni nalogi obravnavam problematiko prometnih nesreč v Sloveniji. V ta namen analiziram podatke prometnih nesreč, ki sem jih pridobil na spletni strani slovenske policije https://www.policija.si/o-slovenski-policiji/statistika/prometna-varnost. Osredotočil se bom na analizo samih prometnih nesreč, kot tudi na udeležence prometnih nesreč. Iz analiz želim spoznati kateri dejavniki vplivajo na prometne nesreče.
UVOD
V predlogu projektne naloge sem opisal podatke prometnih nesreč, ki sem jih našel na spletni strani slovenske policije na naslovu https://www.policija.si/o-slovenski-policiji/statistika/prometna-varnost. Struktura podatkov, ki sem jih opisal ob predlogu projekta se je drastično spremenila. V vmesnem obdobju so na strani slovenske policije spremenili strukturo podatkov, zato sem moral analize, ki sem jih imel že pripravljene spremeniti na novo strukturo podatkov.
OPIS PODATKOV
Opis spremembe podatkov prometnih nesreč
Ob predlogu projekta so bile strukture podatkov prometnih nesreč v Sloveniji podane v obliki 2 datotek – podatkovnih baz, ki so za vsako leto opisovale prometne nesreče v 1 datoteki ter udeležence prometnih nesreč v drugi datoteki. 
Nova struktura podatkov je sestavljena iz 1 datoteke za vsako leto. Datoteke so v csv obliki, ločeni z podpičjem(;) ter poimenovane PN[leto prometne nesreče].csv. Na spletni strani so shranjene v obliki zip datotek.
Nova struktura baze prometnih nesreč je opisana na koncu poročila.
Število prebivalcev po upravnih enotah
Zaradi boljšega razumevanja prometnih nesreč se mi zdi pomembno, da se poleg samih nesreč preveri tudi druge podatkovne baze s pomočjo katerih dobimo boljši občutek prometnih nesreč. Za ta namen sem iz Statističnega urada Republike Slovenije pridobil podatke o številu prebivalcev po upravnih enotah. Podatke sem pridobil na spletni strani https://pxweb.stat.si/pxweb/Dialog/varval.asp?ma=05C3002S&ti=&path=../Database/Dem_soc/05_prebivalstvo/10_stevilo_preb/15_05C30_prebivalstvo_upravne/&lang=2

Vremenski podatki
Zaradi boljše analize prometnih nesreč me je zanimalo, ali imajo zunanji dejavniki, kot je temperatura zraka, višina snežne podlage vpliv na prometno nesrečo. V ta namen sem pridobil ARSO zgodovinske podatke o vremenu. Podatke sem pridobil za nekaj pomembni mest ter jih shranil v CSV datoteki. Podatki so javno dostopni na spletni strani http://meteo.arso.gov.si/met/sl/archive/.
PRIPRAVA PODATKOV
Podatki prometnih nesreč v Sloveniji so shranjeni v različnih datotekah na spletu. Zaradi optimizacije in možnosti pridobivanja podatkov v prihodnosti sem pripravil proceduro, s pomočjo katere podatke pridobimo iz spleta. Za potrebe analiz sem se odločil za prometne nesreče od leta 2010 naprej. Datoteke sem ekstrahiral na lokalni disk. Za potrebe celovite analize jih združimo v eno podatkovno bazo.
Pridobivanje podatkov
for i in range(2010,2020,1):
        path = "https://www.policija.si/baza/pn" +  str(i) + ".zip"
        #path = "https://www.policija.si/baza/pn2010.zip"
        resp = urlopen(path)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall("./files")
Združevanje podatkov
os.chdir("c:/_PERSONAL/FRI/PR/Projekt/files")
    results = pd.DataFrame([])
     for counter, file in enumerate(glob.glob("PN*.csv")):
        namedf = pd.read_csv("./" +file, error_bad_lines=False,sep=';',encoding="ANSI")
        results = results.append(namedf)
    results.to_csv('./PrometneNesrece.csv')

Čiščenje podatkov
Ker je podatkovna baza prometnih nesreč pripravljena v obliki, ki ni najbolj primerna za napredne analize je potrebno podatke ustrezno prečistiti. 
Štetje prometnih nesreč
Izziv, ki sem ga zasledil je bila skupna podatkovna baza za nesreče in udeležence nesreč. V ta namen je potrebno ob analizi prometnih nesreč – štetje prometnih nesreč upoštevati razlikovalno štetje prometnih nesreč po stolpcu ZaporednaStevilkaPN. V primeru analize udeležencev prometnih nesreč ni potrebe po razlikovalnem štetju, temveč se uporabi normalno štetje. 
Ureditev datumskih atributov
Datum prometne nesreče je v obliki DD.MM.YYYY (02.03.2018). To je format, ki ni primeren za analizo, zato sem spremenil format datuma v YYYY-MM-DD. Poleg tega sem datum razdelil na dodatne atribute: leto, mesec, mesec naziv, dan, dan v tednu, dan v tednu številka.
Ureditev starostnih razredov udeležencev prometnih nesreč (bins)
Starostne razrede udeležencev prometnih nesreč sem razdelil v razrede s pomočjo katerih bom lažje analiziral kako vpliva starost povzročitelja na prometno nesrečo.  Podatke sem razdelil v naslednje razrede: '<16', '16-18', '18-21','21-50', '50-65', '65+'
Ureditev razredov za stalež vozniškega dovoljenja
Ali stalež vozniškega dovoljenja vpliva na prometno nesrečo? S pomočjo razredov staležev vozniškega dovoljenja sem segmentiral udeležence oz. povzročitelje prometnih nesreč v 6 segmentov: '<2', '2-5', '5-10', '10-20', '20-30', '30+'
Ureditev razreda državljanstev
Pri analizi prometnih nesreč želim primerjati koliko prometnih nesreč povzročijo slovenski državljani in koliko prometnih nesreč povzročijo tuji državljani.
Glavne ugotovitve
Podatkovne ugotovitve
Pri analizi podatkov sem ugotovil, da obstaja množica podatkov, kjer je starost povzročitelja oz. udeleženca prometne nesreče manj kot 0 let, zato sem pri analizi udeležencev prometne nesreče take podatke odstranil.
Prav tako sem pri analizi staleža vozniškega dovoljenja odstranil podatke, kjer je Starost osebe – Stalež vozniškega dovoljenja < 0.
Razdelitev prometnih nesreč po letih
Iz analize vidimo, da je število prometnih nesreč do leta 2017 bilo v upadu. Medtem, ko je leta 2018 število prometnih nesreč naraslo. Zanimivo pa je dejstvo, da se število prometnih nesreč s smrtnim izidom iz leta v leto niža. Velik vpliva na to imajo najbrž preventivne akcije, dobra osveščenost ljudi.
 
 
 
Povprečno število oseb v prometnih nesrečah po letih
Iz analize povprečnega števila oseb udeleženih v prometnih nesrečah lahko sklepamo, da so v prometnih nesrečah v povprečju udeležene manj kot 2 osebi. Največje povprečno število udeleženih oseb v prometnih nesrečah je bilo leta 2015. Za podrobno analizo povprečnega števila oseb bom v nadaljevanju projekta razdelil osebe po vrsti udeleženca.
 
Distribucija starosti povzročiteljev prometnih nesreč
Analiza starosti povzročiteljev prometnih nesreč kaže na to, da največ prometnih nesreč povzročijo osebe v zgodnjih 20 letih. To so osebe, ki imajo kratek vozniški stalež. Torej neizkušeni vozniki, ki v večini primerov precenjujejo svoje sposobnosti.
 
