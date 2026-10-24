Balabolka (Dienstprogramm zur Textextraktion), Version 1.120
Copyright (c) 2013-2026 Ilya Morozov
Alle Rechte vorbehalten

WWW: https://www.cross-plus-a.com/de/btext.htm
E-mail: crossa@list.ru (nur russisch oder englisch)

Lizenzart: Freeware
Plattformen: Microsoft Windows 7/8/10/11


Das Programm ermöglicht es, Text aus verschiedenen Dateiarten zu extrahieren.
Der extrahierte Text kann in einer Datei zusammengefasst oder/und in mehrere Dateien aufgeteilt werden.
Die Liste der Aussprache-Korrektur-Regeln von Balabolka kann auf den Text angewandt werden.
Unterstützte Formate für Input-Dateien: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
Die IFilter-Schnittstelle wird für Dateien mit unbekannten Erweiterungen verwendet.

Die Anwendung kann die Bibliothek 7z.dll (32 Bit) verwenden. Sie hilft dabei, Text und Bilder aus Dokumenten in Archiven (ZIP, RAR usw.) zu extrahieren. 7z.dll ist Teil der Software 7-Zip.
https://www.7-zip.org


*** Anwendung ***

blb2txt [Optionen ...]


*** Befehlszeilen-Optionen ***

-f <Dateiname>
   Bestimmt den Namen der Eingabedatei oder die Maske für eine Gruppe von Eingabedateien. Die Befehlszeile kann einige Optionen enthalten [-f].

-fl <Dateiname>
   Bestimmt den Namen der Textdatei mit der Liste der Eingabedateien (ein Dateiname pro Zeile). Die Befehlszeile kann einige Optionen enthalten [-fl].

-v <Ordner>
   Bestimmt den Namen des Ausgabeordners zum Speichern der Textdateien. 

-p <Textzeile>
   Bestimmt das Muster für den Namen der Ausgabedatei (zum Beispiel "Textdokument"). Wenn nicht vorhanden, wird der Name der Eingabedatei verwendet.
   Verwenden Sie die Variable %FileName%, um den Namen der Eingabedatei in den Namen der Ausgabedatei einzufügen.
   Verwenden Sie die Variable %FirstLine%, um die erste Textzeile als Namen der Ausgabedatei einzufügen.
   Verwenden Sie die Variable %Header%, um den Kapiteltitel als Namen der Ausgabedatei einzufügen.
   Verwenden Sie die Variable %Number%, um die Position der Sequenznummer im Namen der Ausgabedatei zu ändern.
   Use the %Title% variable to insert the title of the HTML document (for HTML files only).
   Achtung! Es ist notwendig, das Prozentzeichen (%) in einer Stapelverarbeitungsdatei zu verdoppeln. Zum Beispiel: -p %%FirstLine%%

-ext <Textzeile>
   Legen Sie die Erweiterung für Ausgabedateinamen fest. Die Standardeinstellung ist "txt".

-out <Dateiname>
   Legt den vollständigen Namen der Ausgabedatei fest. Die Option wird nur dann empfohlen, wenn das Dienstprogramm als Teil einer anderen Software verwendet wird.
   Wenn das Dienstprogramm für den Import benutzerdefinierter Dokumente verwendet wird, führt das externe Programm das Dienstprogramm von einer Befehlszeile aus und übergibt den vollständigen Namen einer zu erstellenden Textdatei.

-s
   Suchen Sie nach Eingabedateien in Unterordnern.

--create-folder oder -cf
   Erstellen Sie für jede Eingabedatei einen Unterordner. Der Dateiname wird als Name des Ausgabeunterordners verwendet.

-i
   Liest Text aus STDIN. Wenn diese Option gewählt ist, wird die Option [-f] ignoriert.

-o
   Schreibt Text in STDOUT. Wenn diese Option gewählt ist, werden die Optionen [-v] und [-p] ignoriert.

-u
   Kombiniert Textdateien in einer Ausgabedatei.

-b
   Setzt die Sequenznummer vor den Namen der Ausgabedatei.

-a
   Setzt die Sequenznummer hinter den Namen der Ausgabedatei.

-n <Zahl>
   Bestimmt die Start-Sequenznummer für die Ausgabedateien. Die Standardzahl ist 1.

-e <Kodierung>
   Bestimmt die Kodierung der Ausgabedateien ("ansi", "utf8" oder "unicode"). Standard ist "ansi".

-t <Zahl>
   Splittet Text nach Zielgröße für Ausgabedateien (als Anzahl der Zeichen).

-k <Schlüsselwort>
   Splittet Text vor einem speziellem Schlüsselwort in der Eingabedatei. Diese Option beachtet die Groß- und Kleinschreibung. Die Befehlszeile kann einige Optionen enthalten [-k].

-r <Schlüsselwort>
   Splittet Text vor einem Schlüsselwort und entfernt dieses von den Ausgabedateien. Diese Option beachtet die Groß- und Kleinschreibung. Die Befehlszeile kann einige Optionen enthalten [-r].

-w
   Splittet Text an zwei Leerzeilen in Folge.

-l
   Splittet Text vor Zeilen, die nur Großbuchstaben enthalten.

-c
   Teilt den Text nach einem Inhaltsverzeichnis. Die Anwendung extrahiert Positionen von Kapitelanfängen aus der Eingabedatei (if the file contains such information).

-toc
   Erzeugt ein Inhaltsverzeichnis und teilt den Text. Die Anwendung teilt den extrahierten Text nach Stichworten (wie "Kapitel" oder "Buch").
   Wenn die Option zusammen mit der Option [-c] verwendet wird, versucht die Anwendung, ein Inhaltsverzeichnis aus dem Dokument zu extrahieren; wenn dies fehlschlägt, wird ein neues Inhaltsverzeichnis erstellt.

-m <Zahl>
   Legt die minimale Größe der Textteile für die Aufteilung fest (als Anzahl der Zeichen).

-j <Zahl>
   Ignoriert den Kapitelanfang, wenn die Größe des vorherigen Kapitels kleiner als der angegebene Wert (in Zeichen) ist. Die Option wird zusammen mit der Option [-c] oder [-toc] verwendet.

-hh <Textzeile>
   Fügt Text vor Überschriften ein (z.B.: ## Kapitel 1).

-d <Dateiname>
   Verwendet ein Wörterbuch zur Aussprache-Korrektur (*.BXD, *.DIC oder *.REX). Die Befehlszeile kann einige Optionen enthalten [-d].

-if
   Verwendet die IFilter-Schnittstelle, um Text zu extrahieren. Wenn dies fehlschlägt, wird die Standardmethode von der Anwendung verwendet.

-g <Ordner>
   Legt den Namen des Ausgabeordners für das Speichern von Bildern aus einem Dokument fest.

-cvr <Ordner>
   Legt den Namen des Ausgabeordners für das Speichern eines Buchumschlagsbildes fest.

--clone-file-time oder -cft
   Clone the Created/Modified/Accessed time of the input file into the output file. If the application combines text files or splits the extracted text, the option is ignored.

-x <Dateityp>
   Legt den Typ der Eingabedatei fest. Damit kann das Format von Eingabedokumenten mit unbekannten Dateinamenerweiterungen definiert werden. Zum Beispiel: -x doc

-pwd <Textzeile>
   Legt das Passwort für die verschlüsselten PDF-Dateien fest.

-dll <Dateiname>
   Sets the path and name for 7z.dll (32bit). This library helps to extract text and images from documents inside archives (ZIP, RAR, etc.). 7z.dll is a part of 7-Zip software.
   If the option is not specified, the application and the library must be in the same folder; otherwise, the application will not be able to extract data from archive files.

-dex <Datentypen>
   Sets the list of file types for extracting from archives. The option contains a comma-separated list of file types, for example: -dex "fb2,epub"
   The command line may contain few options [-dex]. If the option is not specified, the application will extract text from all files in an archive.
   If it is necessary to extract text for all file types supported by the application, use the value "all-". For example: -dex all-

-dne <Datentypen>
   Sets the list of file types to ignore when documents are extracted from archives. The option contains a comma-separated list of file types, for example: -dne "exe,dll"
   The command line may contain few options [-dne]. If the option is not specified, the application will extract text from all files in an archive.

-dp
   Anzeige von Fortschrittsinformationen in einem Konsolenfenster.

-cfg <Dateiname>
   Legt den Namen der Konfigurationsdatei mit den Befehlszeilenoptionen fest (eine Textdatei, in der jede Zeile eine Option enthält). Wird die Option nicht angegeben, wird die Datei "blb2txt.cfg" verwendet, die sich im selben Ordner wie das Dienstprogramm befindet.

-h
   Druckt die Liste der verfügbaren Befehlszeilen-Optionen.

--remove-spaces oder -rs
   Entfernt überschüssige Leerzeichen (zwei oder mehr Leerzeichen in Folge, geschützte Leerzeichen).

--remove-hyphens oder -rh
   Entfernt Bindestriche am Ende von Textzeilen.

--remove-linebreaks oder -rl
   Entfernt Zeilenumbrüche innerhalb von Absätzen.

--remove-empty-lines oder -rm
   Entfernt Leerzeilen.

--replace-empty-lines oder -rp
   Ersetzt mehrere Leerzeilen durch eine einzige Leerzeile.

--remove-square-brackets oder -rsb
   Entfernt Text in [eckigen Klammern].

--remove-curly-brackets oder -rcb
   Entfernt Text in {geschweiften Klammern}.

--remove-angle-brackets oder -rab
   Entfernt Text in <spitzen Klammern>.

--remove-round-brackets oder -rrb
   Entfernt Text in (runden Klammern).

--remove-comments oder -rc
   Entfernt Kommentare. Einzeilige Kommentare beginnen mit // und werden bis zum Ende der Zeile fortgesetzt. Mehrzeilige Kommentare beginnen mit /* und enden mit */.

--remove-page-numbers oder -rpn
   Entfernt Seitenzahlen (kann für DjVu/PDF-Dateien nützlich sein).

--fix-ocr-errors oder -ocr
   Behebt OCR-Fehler (nur für Sprachen mit kyrillischen Alphabeten).

--fix-letter-spacing oder -ls
   Buchstabenabstand entfernen (zum Beispiel: W o r t, _T_e_x_t).

--add-period oder -ap
   Fügt einen Punkt hinzu, wenn nach dem letzten Wort des Absatzes kein Satzzeichen steht.

--extract-summary <Zahl> oder -es <Zahl>
   Extrahiert eine Zusammenfassung (auch "Annotation" genannt) aus FB2/FB3-Dateien und fügt sie am Anfang des Textes ein.
   Mögliche Werte für den ganzzahligen Parameter:
   0 – überspringt eine Zusammenfassung (dieser Wert wird standardmäßig verwendet);
   1..5 – extrahiert eine Zusammenfassung (ein Wert bestimmt die Reihenfolge, in der ein Autorenname und ein Buchtitel aufgeführt werden).

--skip-notes oder -sn
   Überspringt Notizen, wenn die Anwendung Text aus DOCX/FB2/FB3/MD/ODT-Dateien extrahiert.

--include-notes <Zahl> oder -in <Zahl>
   Enthält Anmerkungen im Text, wenn die Anwendung Text aus DOCX/FB2/FB3/MD/ODT-Dateien extrahiert.
   Mögliche Werte für den Integer-Parameter:
   0 – entfernt Verknüpfungen zu Anmerkungen aus dem Text;
   1 – behält die Standardpositionen der Anmerkungen im Text bei (dieser Wert wird standardmäßig verwendet);
   2 – platziert Anmerkungen am Ende von Sätzen;
   3 – platziert Anmerkungen am Ende von Absätzen.

--insert-note-begin <Textzeile> oder -inb <Textzeile>
   Fügt Wörter am Anfang von Anmerkungen ein, wenn Anmerkungen in den Text eingefügt werden (z.B.: Anmerkung des Herausgebers.).
   Diese Option wird für DOCX/FB2/FB3/MD/ODT-Dateien verwendet.

--insert-note-end <Textzeile> oder -ine <Textzeile>
   Fügt Wörter am Ende von Anmerkungen ein, wenn Anmerkungen in den Text eingefügt werden (z.B.: Ende der Anmerkung.).
   Diese Option wird für DOCX/FB2/FB3/MD/ODT-Dateien verwendet.

--extract-tables <Zahl> oder -et <Zahl>
   Tabellen aus DOCX-/FB2-/FB3-/ODT-Dateien extrahieren.
   Mögliche Werte für den Integer-Parameter:
   0 – überspringt Tabellen beim Extrahieren von Text;
   1 – extrahieren Sie Daten aus jeder Zelle als neue Textzeile (dieser Wert wird standardmäßig verwendet);
   2 – behalten Sie die Formatierung beim Extrahieren einer Tabelle bei.

--csv-comma
   Spalten werden durch ein Komma getrennt, wenn die Anwendung Daten aus XLS/XLSX/ODS-Dateien extrahiert (Standard-Trennzeichen für CSV-Dateien).

--csv-semicolon
   Spalten werden durch ein Semikolon getrennt, wenn die Anwendung Daten aus XLS/XLSX/ODS-Dateien extrahiert.

--csv-space
   Spalten werden durch ein Leerzeichen getrennt, wenn die Anwendung Daten aus XLS/XLSX/ODS-Dateien extrahiert.

--csv-tab
   Spalten werden durch Tab getrennt, wenn die Anwendung Daten aus XLS/XLSX/ODS-Dateien extrahiert.

--csv-double-quote
   Verwendet doppelte Anführungszeichen, wenn ein Feld zitiert werden muss (Export aus XLS/XLSX/ODS-Dateien).

--csv-single-quote
   Verwendet einfache Anführungszeichen, wenn ein Feld zitiert werden muss (Export aus XLS/XLSX/ODS-Dateien).

--eml-save <Ordner>
   Extracts attachments from EML files and saves to a specified folder.

--eml-att
   Extracts the list of attachments from EML files (names of files attached to the message).

--eml-cc
   Extracts the header field "Cc" from EML files ("carbon copy"; it specifies additional recipients of the message).

--eml-date <Datumsformat>
   Extracts the header field "Date" from EML files (the local time and date when the message was composed and sent). A date format are defined by specifiers (such as "d", "m", "y", etc.). For example: "dd.mm.yyyy hh:nn:ss".

--eml-from
   Extracts the header field "From" from EML files (the email address, and optionally the name of the author).

--eml-org
   Extracts the header field "Organization" from EML files (the name of the organization through which the sender of the message has net access).

--eml-rt
   Extracts the header field "Reply-To" from EML files (the address for replies to go to).

--eml-subj
   Extracts the header field "Subject" from EML files (the subject of the message).

--eml-to
   Extracts the header field "To" from EML files (the email address, and optionally the name of the message's recipient).


*** Beispiele ***

Extrahieren Sie den Text aus BOOK.DOC und speichern Sie ihn als BUCH.TXT im Ausgabeordner:

blb2txt -f "d:\Docs\Book.doc" -v "d:\Text\" -p "Buch"


Auch diese Variante kann bei Bedarf verwendet werden (wenn nur eine Eingabedatei angegeben ist):

blb2txt -f "d:\Docs\Book.doc" -out "d:\Text\Buch.txt"


Text aus Microsoft Word- und RTF-Dokumenten extrahieren, Leerzeilen entfernen und Textdateien in UTF-8-Kodierung speichern:

blb2txt -f "d:\Docs\*.docx" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Extrahieren Sie den Text aus allen Textdateien im angegebenen Ordner, fügen Sie ihn zusammen und speichern Sie ihn als DOCUMENT.TXT:

blb2txt -f "d:\Docs\*.*" -v "d:\Text\" -p "Dokument" -u


Extrahieren Sie Text aus DOCUMENT.DOCX, teilen Sie ihn in Teile mit einer Größe von 100 KB auf und speichern Sie ihn als Textdateien "Document 20.txt", "Document 21.txt" usw.:

blb2txt -f "d:\Docs\Document.docx" -v "d:\Text\" -p "Dokument" -a -n 20 -t 100000


Extrahieren Sie Text aus BOOK.FB2, suchen Sie die Wörter "KAPITEL" und "BAND", um den Text in Teile zu unterteilen, und speichern Sie diese als Dateien mit den Namen "Buch 1.txt", "Buch 2.txt" usw.:

blb2txt -f "d:\Book\Book.fb2" -v "d:\Text\" -p "Buch" -k "KAPITEL" -k "BAND"


Text aus BOOK.EPUB extrahieren, "###" suchen, um den Text in Teile zu unterteilen, "###" aus dem Text entfernen und jeden Teil als neue Datei speichern:

blb2txt -f "d:\Book\Book.epub" -v "d:\Text\" -p "Buch" -r "###"


Text aus BOOK.FB2 extrahieren, nach Inhaltsverzeichnis aufteilen, Dateien speichern und Kapitelüberschriften als Dateinamen verwenden. Neue Textdateien dürfen nicht kleiner als ein Kilobyte sein:

blb2txt -f "d:\Book\Book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024


Text aus STDIN abrufen, überflüssige Leerzeichen, Zeilenumbrüche und Leerzeilen entfernen, den aktualisierten Text in STDOUT schreiben:

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Text aus allen Microsoft Word-Dokumenten in ZIP-Archiven extrahieren:

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Konfigurationsdatei ***

Die Befehlszeilen-Optionen können als Konfigurationsdatei "blb2txt.cfg" im Ordner der Konsolen-Anwendung gespeichert werden. 

Beispiel für eine Konfigurationsdatei:
=====================
-f d:\Docs\*.rtf
-f d:\Books\*.epub
-f d:\Books\*.fb2
-v d:\Text
-b
-n 1
-t 25000
-e utf8
-d d:\Dict\rules.bxd
--remove-spaces
--remove-linebreaks
--replace-empty-lines
=====================

Das Programm kann Optionen von der Konfigurationsdatei und der Kommandozeile kombinieren.


*** Operationen ***

Ausführungsreihenfolge der Operationen:

1. Text von Eingabedatei(en) extrahieren.
2. Text formatieren: Leerzeichen, Zeilenumbrüche usw. entfernen (wenn Option gewählt).
3. Dateien in einer Datei zusammenfassen (wenn Option gewählt).
4. Text splitten (wenn Option gewählt).
5. Regeln für Aussprache-Korrektur anwenden (wenn Option gewählt).
6. Ausgabedatei(en) speichern.


*** Lizenzart ***

Sie können Software für nichtkommerzielle Zwecke verwenden und vertreiben. Für die kommerzielle Nutzung oder den Vertrieb benötigen Sie die Genehmigung des Urheberrechtsinhabers.

###
