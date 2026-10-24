Balabolka (Text Extract Utility), version 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW: https://www.cross-plus-a.com/it/btext.htm
E-mail: crossa@list.ru

Licenza: gratuita
Sistema operativo: Microsoft Windows 7/8/10/11


Il programma permette di estrarre il testo da vari tipi di file.
Il testo estratto può essere riunito in un singolo file e/o suddiviso in vari file.
Al testo possono essere applicate le regole dei dizionari di correzione della pronuncia del programma Balabolka.
Formati supportati per i file di ingresso: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
L'interfaccia IFilter verrà utilizzata per i file con estensione sconosciuta.

L'applicazione può usare la libreria 7z.dll (32bit). Essa aiuta ad estrarre testo e immagini da documenti all'interno di archivi (ZIP, RAR, eccetera). 7z.dll è un componente del software 7-Zip.
https://www.7-zip.org


*** Riga di comando ***

blb2txt [opzioni ...]


*** Opzioni della riga di comando ***

-f <nome_file>
   Nome del file o specificazione del gruppo di file da cui si estrae il testo. La riga di comando può contenere varie opzioni [-f].

-fl <nome_file>
   Imposta il nome del file di testo con l'elenco dei file in ingresso, un nome di file per riga. La riga di comando può contenere varie opzioni [-fl].

-v <nome_cartella>
   Nome della cartella in cui salvare il file con il testo estratto.

-p <testo>
   Schema del nome di file con il testo estratto (ad esempio "Documento"). In sua assenza viene usato il nome del file in ingresso.
   Usare la variabile %FileName% per inserire il nome del file in ingresso nel nome del file in uscita.
   Usare la variabile %FirstLine% per inserire la prima riga di testo nel nome del file di uscita.
   Usare la variabile %Header% per inserire il titolo del capitolo.
   Usare la variabile %Number% per cambiare la posizione del numero di sequenza entro il nome del file di uscita.
   Usare la variabile %Title% per inserire il titolo del documento HTML (solo per i file HTML).
   Attenzione! È necessario raddoppiare ogni carattere di percentuale (%) in uno script batch. Ad esempio: -p %%FirstLine%%

-ext <testo>
   Imposta l'estensione per i nomi di file in uscita. Il valore predefinito è "txt".

-out <nome_file>
   Imposta il nome completo per il file di uscita. L'opzione è raccomandata solo quando l'utility viene usata come parte di un altro software.
   Se l'utility viene usata per personalizzare l'importazione di documenti, il programma esterno avvia l'utility da una riga di comando e fornisce il nome completo di un file di testo da generare.

-s
   Cerca i file in ingresso anche nelle sottocartelle.

--create-folder o -cf
   Genera una sottocartella in uscita per ciascun file in ingresso. I nomi dei file sono impiegati come nomi delle sottocartelle.

-i
   Legge il testo dal flusso in ingresso (STDIN). Se si specifica questa opzione, l'opzione [-f] è ignorata.

-o
   Scrive il testo estratto nel flusso di uscita (STDOUT). Se si specifica questa opzione, le opzioni [-v] e [-p] sono ignorate.

-u
   Riunisce il testo di vari file in un file singolo.

-b
   Aggiunge un numero progressivo all'inizio del nome del file di uscita.

-a
   Aggiunge un numero progressivo alla fine del nome del file di uscita.

-n <numero_intero>
   Imposta il numero iniziale della sequenza dei file di uscita. Il valore predefinito è 1.

-e <codifica>
   Codifica dei file con il testo estratto ("ansi", "utf8" o "unicode"). Il valore predefinito è "ansi".

-t <numero_intero>
   Divide il testo destinazione specificando la grandezza delle parti (come numero di caratteri).

-k <parola_chiave>
   Divide il testo in ingresso sulla particolare parola chiave. L'opzione distingue maiuscole/minuscole. La riga di comando può contenere varie opzioni [-k].

-r <parola_chiave>
   Divide il testo in ingresso sulla parola chiave e la rimuove. L'opzione distingue maiuscole/minuscole. La riga di comando può contenere varie opzioni [-r].

-w
   Divide il testo su due righe vuote consecutive.

-l
   Divide il testo sulle righe in cui tutte le lettere sono maiuscole.

-c
   Divide il testo secondo un indice. L'applicazione estrae le posizioni di inizio capitolo dal testo in ingresso (se il file contiene tale informazione).

-toc
   Genera un sommario e divide il testo. L'applicazione divide il testo estratto per parole chiave (come "capitolo").
   Se l'opzione viene usata insieme con l'opzione [-c], l'applicazione tenterà di estrarre un sommario dal documento; se fallisce, verrà generato un nuovo sommario.

-m <numero_intero>
   Imposta la grandezza minima delle parti di testo per la divisione, espressa come numero di caratteri.

-j <numero_intero>
   Ignora l'inizio di capitolo se la grandezza del capitolo precedente è minore del valore specificato (in caratteri). L'opzione si usa insieme con l'opzione [-c] e [-toc].

-hh <testo>
   Inserisce il testo davanti alle intestazioni. (ad esempio: ## Capitolo 1).

-d <nome_file>
   Usa un dizionario per la correzione della pronuncia (*.BXD, *.DIC o *.REX). La riga di comando può contenere varie opzioni [-d].

-if
   Usa l'interfaccia IFilter per estrarre il testo. Se non funziona, il metodo predefinito verrà usato dall'applicazione.

-g <nome_cartella>
   Imposta il nome della cartella di uscita per il salvataggio delle immagini da un documento.

-cvr <nome_cartella>
   Imposta il nome della cartella di uscita per il salvataggio dell'immagine di copertina di un libro.

--clone-file-time o -cft
   Clonare l'orario di creazione/modifica/accesso del file in ingresso nel file in uscita. Se l'applicazione combina file di testo o suddivide il testo estratto, l'opzione è ignorata.

-x <file_type>
   Imposta il tipo di file in ingresso. Permette di definire un formato per i documenti in ingresso aventi nome di file con estensione sconosciuta. Ad esempio: -x doc

-pwd <testo>
   Specifica la password per estrarre il testo da un file PDF cifrato.

-dll <nome_file>
   Imposta il percorso e il nome per 7z.dll (32bit). Questa libreria aiuta ad estrarre testo e immagini da documenti all'interno di archivi (ZIP, RAR, eccetera). 7z.dll fa parte del software 7-Zip.
   Se l'opzione non viene specificata, l'applicazione e la libreria devono trovarsi nella stessa cartella; in caso contrario l'applicazione non sarà in grado di estrarre i dati dai file archivio.

-dex <file_types>
   Imposta l'elenco dei tipi di file per l'estrazione da archivi. L'opzione specifica una lista di tipi di file separati da virgole, ad esempio: -dex "fb2,epub"
   La riga di comando può comprendere più opzioni [-dex]. Se l'opzione non viene specificata, l'applicazione estrarrà il testo da tutti i file nell'archivio.
   Se è necessario estrarre il testo per tutti i tipi di file gestiti dall'applicazione, usare il valore "all-". Ad esempio: -dex all-

-dne <file_types>
   Imposta l'elenco dei tipi di file da ignorare quando i documenti vengono estratti da archivi. L'opzione specifica una lista di tipi di file separati da virgole, ad esempio: -dne "exe,dll"
   La riga di comando può comprendere più opzioni [-dne]. Se l'opzione non viene specificata, l'applicazione estrarrà il testo da tutti i file nell'archivio.

-dp
   Visualizza le informazioni sullo stato di avanzamento in una finestra console.

-cfg <nome_file>
   Imposta il nome del file di configurazione con le opzioni della riga di comando (un file di testo dove ciascuna riga contiene un'opzione). Se l'opzione non è specificata, viene usato il file "blb2txt.cfg" nella stessa cartella dell'utility.

-? o -h
   Mostra l'elenco delle opzioni disponibili nella riga di comando.

--remove-spaces o -rs
   Elimina gli spazi superflui (due o più spazi bianchi consecutivi, spazi indivisibili).

--remove-hyphens o -rh
   Elimina i trattini di sillabazione alla fine delle righe del testo.

--remove-linebreaks o -rl
   Elimina le interruzioni di riga entro i paragrafi.

--remove-empty-lines o -rm
   Elimina le righe vuote.

--replace-empty-lines o -rp
   Sostituisce più righe vuote con una singola riga vuota.

--remove-square-brackets o -rsb
   Elimina il testo racchiuso fra [parentesi quadre].

--remove-curly-brackets o -rcb
   Elimina il testo racchiuso fra {parentesi graffe}.

--remove-angle-brackets o -rab
   Elimina il testo racchiuso fra <parentesi angolate>.

--remove-round-brackets o -rrb
   Elimina il testo racchiuso fra (parentesi tonde).

--remove-comments o -rc
   Elimina i commenti. I commenti su riga singola iniziano con // e continuano fino alla fine della riga. I commenti su più righe iniziano con /* e terminano con */.

--remove-page-numbers o -rpn
   Elimina i numeri di pagina (può essere utile per file di tipo DjVu/PDF).

--fix-ocr-errors o -ocr
   Corregge gli errori dovuti all'OCR (solo per lingue con alfabeto cirillico).

--fix-letter-spacing o -ls
   Corregge la spaziatura fra lettere nelle parole (ad esempio: s p a c e, _w_o_r_d).

--add-period o -ap
   Aggiunge un punto se non c'è alcun segno di punteggiatura dopo l'ultima parola del paragrafo.

--extract-summary <numero_intero> o -es <numero_intero>
   Estrae un sommario (chiamato anche "Annotation") dai file FB2/FB3 e lo inserisce all'inizio del testo.
   Valori possibili per il parametro intero:
   0 – omette il sommario (questo valore è usato per impostazione predefinita);
   1..5 – estrae un sommario (il valore determina l'ordine in cui elencare il nome dell'autore e il titolo del libro).

--skip-notes o -sn
   Salta le note, quando l'applicazione estrae il testo dai file DOCX/FB2/FB3/MD/ODT.

--include-notes <numero_intero> o -in <numero_intero>
   Include le note entro il testo, quando l'applicazione estrae il testo da file di tipo DOCX/FB2/FB3/MD/ODT.
   Valori possibili per il parametro numero_intero:
   0 – elimina dal testo i richiami alle note;
   1 – mantiene le normali posizioni delle note entro il testo (questo valore è usato per impostazione predefinita);
   2 – pone le note alla fine delle frasi;
   3 – pone le note alla fine dei paragrafi.

--insert-note-begin <testo> o -inb <testo>
   Inserisce parole all'inizio delle note, quando le note sono incluse entro il testo (ad esempio: Nota dell'editore.).
   L'opzione è usata per file di tipo DOCX/FB2/FB3/MD/ODT.

--insert-note-end <testo> o -ine <testo>
   Inserisce parole alla fine delle note, quando le note sono incluse entro il testo (ad esempio: Fine della nota.).
   L'opzione è usata per file di tipo DOCX/FB2/FB3/MD/ODT.

--extract-tables <numero_intero> o -et <numero_intero>
   Estrae le tabelle dai file DOCX/FB2/FB3/ODT.
   Valori possibili per il parametro numero_intero:
   0 – salta le tabelle;
   1 – estrae il dato da ciascuna cella come una nuova riga di testo (questo è il valore predefinito);
   2 – mantiene la formattazione nell'estrarre una tabella.

--csv-comma
   Le colonne sono separate da una virgola, quando l'applicazione estrae i dati dai file XLS/XLSX/ODS (delimitatore predefinito per i file CSV).

--csv-semicolon
   Le colonne sono separate da un punto e virgola, quando l'applicazione estrae i dati da file XLS/XLSX/ODS.

--csv-space
   Le colonne sono separate da uno spazio vuoto, quando l'applicazione estrae i dati da file XLS/XLSX/ODS.

--csv-tab
   Le colonne sono separate da un carattere tab, quando l'applicazione estrae i dati da file XLS/XLSX/ODS.

--csv-double-quote
   Usa virgolette doppie, se un campo deve essere virgolettato (esportazione dai file XLS/XLSX/ODS).

--csv-single-quote
   Usa virgolette singole, se un campo deve essere virgolettato (esportazione dai file XLS/XLSX/ODS).

--eml-save <nome_cartella>
   Estrae gli allegati dai file di tipo EML e li salva nella cartella specificata.

--eml-att
   Estrae l'elenco degli allegati dai file di tipo EML (nomi dei file allegati al messaggio).

--eml-cc
   Estrae il campo "Cc" dall'intestazione dei file di tipo EML ("carbon copy"; esso specifica destinatari aggiuntivi del messaggio).

--eml-date <formato_data>
   Estrae il campo "Date" dall'intestazione dei file di tipo EML (data e ora locale di composizione e invio del messaggio). Il formato della data è definito mediante specificatori (quali "d", "m", "y", eccetera.). Ad esempio: "dd.mm.yyyy hh:nn:ss".

--eml-from
   Estrae il campo "From" dall'intestazione dei file di tipo EML (l'indirizzo email e facoltativamente il nome dell'autore).

--eml-org
   Estrae il campo "Organization" dall'intestazione dei file di tipo EML (il nome dell'organizzazione tramite cui il mittente del messaggio ha accesso alla rete).

--eml-rt
   Estrae il campo "Reply-To" dall'intestazione dei file di tipo EML (l'indirizzo cui sono destinate le risposte).

--eml-subj
   Estrae il campo "Subject" dall'intestazione dei file di tipo EML (l'oggetto del messaggio).

--eml-to
   Estrae il campo "To" dall'intestazione dei file di tipo EML (l'indirizzo email e facoltativamente il nome deldestinatario del messaggio).


*** Esempi ***

Estrarre il testo da "libro.doc" e salvarlo come "libro.txt" nella cartella di uscita

blb2txt -f "d:\Docs\libro.doc" -v "d:\Text\"


Anche la variante seguente può essere usata se necessario (quando viene specificato un singolo file):

blb2txt -f "d:\Docs\libro.doc" -out "d:\Text\libro.txt"


Estrarre il testo da LIBRO.DOC e salvarlo come "Nuovo Libro.txt":

blb2txt -f "d:\Docs\libro.doc" -v "d:\Text\" -p "Nuovo Libro"


Estrarre il testo dai documenti Microsoft Word e RTF, eliminare le righe vuote e salvare i file di testo con codifica UTF-8:

blb2txt -f "d:\Docs\*.doc" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Estrarre il testo da tutti i file di testo nella cartella specificata, riunirlo e salvarlo come "Document.txt":

blb2txt -f "d:\Docs\*.txt" -v "d:\Text\" -p "Document" -u


Estrarre il testo da DOCUMENT.DOC, dividerlo in parti di grandezza 100 KB e salvarlo nei file di testo "Document 20.txt", "Document 21.txt", eccetera:

blb2txt -f "d:\Docs\document.doc" -v "d:\Text\" -p "Document" -a -n 20 -t 100000


Estrarre il testo da LIBRO.FB2, trovare le parole "CAPITOLO" e "CONTENUTO" per dividere il testo in parti e salvarlo nei file con nomi "Libro 1.txt", "Libro 2.txt", eccetera:

blb2txt -f "d:\Book\libro.fb2" -v "d:\Text\" -p "Libro" -k "CAPITOLO" -k "CONTENUTO"


Estrarre il testo da LIBRO.EPUB, trovare "###" per dividere il testo in parti, eliminare "###" dal testo e salvare ciascuna parte come un nuovo file:

blb2txt -f "d:\Book\libro.epub" -v "d:\Text\" -p "Libro" -r "###"


Estrarre il testo da LIBRO.FB2, dividerlo secondo un sommario, salvare i file e usare i titoli dei capitoli come nomi dei file. I nuovi file di testo non devono essere inferiori a un kilobyte:

blb2txt -f "d:\Book\libro.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024


Ricevere il testo dal flusso STDIN, eliminare gli spazi in eccesso, i ritorni a capo e le righe vuote, scrivere il testo aggiornato nel flusso STDOUT:

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Estrarre il testo da tutti i documenti Wicrosoft Word contenuti in archivi ZIP:

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** File di configurazione ***

Le opzioni della riga di comando possono essere memorizzate nel file "blb2txt.cfg" nella stessa cartella del programma.

Esempio di file di configurazione:
=====================
-f d:\Docs\*.rtf
-f d:\Books\*.epub
-f d:\Books\*.fb2
-v d:\Text
-b
-n 1
-t 25000
-e utf8
-d d:\Dict\regole.bxd
--remove-spaces
--remove-linebreaks
--replace-empty-lines
=====================

Il programma può combinare le opzioni del file di configurazione con quelle della riga di comando.


*** Operazioni ***

Il programma compie le operazioni nell'ordine seguente:

1.  Estrarre il testo dal/dai file in ingresso.
2.  Formattare il testo: eliminare spazi superflui, interruzioni di riga, etc. (se le opzioni sono specificate).
3.  Riunire più file in un singolo file (se l'opzione è specificata).
4.  Dividere il testo (se le opzioni sono specificate).
5.  Applicare le regole per la correzione della pronuncia (se l'opzione è specificata).
6.  Registrare il/i file su disco.


*** Licenza ***

Diritto di utilizzo non commerciale del programma:
- per le persone fisiche: senza alcuna restrizione;
- per le persone giuridiche: soggetto alle restrizioni riportate nel contratto di licenza del software Balabolka.

L'utilizzo commerciale richiede la previa autorizzazione del titolare del copyright.

###
