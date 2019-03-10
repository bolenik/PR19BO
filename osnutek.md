Problematika prometnih nesreč v Sloveniji

V projektni nalogi bom obravnaval problematiko prometnih nesreč v Sloveniji. Podatki, ki jih bom uporabil v projektni nalogi bom pridobil iz spletne strani slovenske policije. Podatki so na voljo na spletni strani https://www.policija.si/o-slovenski-policiji/statistika/prometna-varnost. Na navedenem spletnem naslovu so prosto dostopne statistične letne datoteke z podatki o prometnih nesrečah v Sloveniji. Podatki so na voljo v zip datoteki za vsako leto posebej. V vsaki datoteki so podatki razdeljeni na dogodke in osebe. Datoteka dogodki opisuje posamezno prometno nesrečo, medtem, ko datoteka osebe opisujejo udeležence prometnih nesreč. Datoteki sta med seboj v relaciji in sicer preko enoznačnega atributa Številka prometne nesreče. Številka prometne nesreče je številka pod katero policija vodi posamezno prometno nesrečo.

V nadaljevanju so navedeni podatki posameznih datotek:

Struktura baze za prometne nesreče (PN):

• številka prometne nesreče - to je enoznačna številka zadeve pod katero policija vodi posamezno prometno nesrečo
• klasifikacija nesreče glede na posledice (Izračuna se avtomatično glede na najhujšo posledico pri udeležencih v prometni nesreči)
• upravna enota, na območju katere se je zgodila prometna nesreča 
• datum nesreče (format: dd/mm/llll) 
• ura nesreče (format: hh) 
• indikator ali se je nesreča zgodila v naselju (1) ali izven (0)
• kategorija ceste na kateri je prišlo do nesreče 
• oznaka ceste ali šifra naselja kjer je prišlo do nesreče
• oznaka odseka ceste ali šifra ulice, kjer je prišlo do nesreče
• tekst ceste ali naselja, kjer je prišlo do nesreče
• tekst odseka ali ulice, kjer je prišlo do nesreče
• točna stacionaža ali hišna številka, kjer je prišlo do nesreče
• opis prizorišča nesreče
• glavni vzrok nesreče
• tip nesreče
• vremenske okoliščine v času nesreče
• stanje prometa v času nesreče
• stanje vozišča v času nesreče 
• stanje površine vozišča v času nesreče 
• Geo Koordinata X (Gauß-Krüger-jev koordinatni sistem)
• Geo Koordinata Y (Gauß-Krüger-jev koordinatni sistem)

Struktura baze oseb v PN:

• številka zadeve, povezovalni parameter na bazo prometnih nesreč
• kot kaj nastopa oseba v prometni nesreči ( 1 = povzročitelj, 0 = udeleženec)
• starost osebe (LL)
• spol (1 = M, 2 = Ž)
• upravna enota stalnega prebivališča
• državljanstvo osebe
• poškodba osebe
• vrsta udeleženca v prometu
• ali je oseba uporabljala varnostni pas ali čelado (polje se interpretira v odvisnosti od vrste udeleženca) (Da/Ne)
• vozniški staž osebe za kategorijo, ki jo potrebuje glede na vrsto udeleženca v prometu (LL)
• vozniški staž osebe za kategorijo, ki jo potrebuje glede na vrsto udeleženca v prometu (MM)
• vrednost alkotesta za osebo, če je bil opravljen (n.nn)
• vrednost strokovnega pregleda za osebo, če je bil odrejen in so rezultati že znani (n.nn)
V projektni nalogi bom podrobno analiziral prometne nesreče. Od enostavnih analiz, kjer bom prikazal koliko prometnih nesreč se zgodi po določenih (zgoraj omenjenih) atributih, do naprednejših analiz, kjer bom poskušal z podatkovnim rudarjenjem ugotoviti odvisnost prometnih nesreč glede na različne atribute. V nalogi bom poleg osnovnih vprašanj v povezavi z prometnimi nesrečami, poskušal odgovoriti tudi na naslednja vprašanja:

1. Kakšna je odvisnost prometnih nesreč od alkoholiziranosti povzročitelja?
2. Ali je več prometnih nesreč z bolj ali manj alkoholiziranimi povzročitelji?
3. Kakšna je odvisnost prometnih, ki jih povzročijo alkoholizirani povzročitelji, glede na vremenske okoliščine,  lokacijo prometne nesreče (v naselju ali izven naselja, upravna enota), število udeležencev prometne nesreče
4. Kakšna je stopnja prometnih nesreč glede na upravno enoto v odvisnosti od števila prebivalstva (podatke bom pridobil na spletni strani Statističnega urada Republike Slovenije (https://www.stat.si/statweb/News/Index/7442)
5. Časovna razdelitev prometnih nesreč: Število prometnih nesreč po letih, mesecih, dnevih, urah. Kaj lahko vpliva na prometne nesreče po posameznih segmentih? Kakšen je vpliv praznikov v Sloveniji na prometne nesreče? Je prometnih nesreč več, ali manj. Podatke o praznikih bom pridobil iz spleta.
6. Kakšen vpliv ima vozniški stalež na prometne nesreče? Ali ljudje s kratkim vozniških staležem povzročijo več nesreč od tistih z daljšim vozniških staležem. 
7. Kakšen vpliv ima starost povzročitelja na prometne nesreče?
8. Ali se vozniki, ki se vozijo z več potniki bolj varni vozniki? Kakšna je korelacija med številom udeležencev in številom prometnih nesreč?
Ker se podatki prometnih nesreč letno objavljajo na spletni strani Policije, bom projektno nalogo zastavil na način, da bo uporabna tudi za prihodnja leta.
