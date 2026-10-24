Balabolka (tekstin jäsennin), versio 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW: https://www.cross-plus-a.com/fi/btext.htm
E-mail: crossa@list.ru

Lisenssi: Freeware
Käyttöjärestelmä: Microsoft Windows 7/8/10/11


Sovelluksen tarkoitus on jäsentää teksti erilaisista tiedostoista.
Saatu teksti voidaan yhdistää samaan tiedostoon tai jakaa muutamiin pienempiin tiedostoihin.
Tekstin kohdalta voi soveltaa Balabolkan äänestyskorjaussanakirjoja.
Sovellus ylläpitää seuraavat tiedostotyypit: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
The IFilter interface will be used for files with unknown extensions.

The application can use the library 7z.dll (32bit). It helps to extract text and images from documents inside archives (ZIP, RAR, etc.). 7z.dll is a part of 7-Zip software.
https://www.7-zip.org


*** Komentorivi ***

blb2txt [options ...]


*** Komentoriviparametrit ***

-f <tiedoston_nimi>
   Tiedoston nimi tai kaava, josta saadaan tekstiä. Komentoon voi sisältyä muutama [-f] -parametri.

-fl <tiedoston_nimi>
   Sets the name of the text file with the list of input files (one file name per line). Komentoon voi sisältyä muutama [-fl] -parametri.

-v <kansion_nimi>
   Tallenna jäsennetty teksti mainittuun kansioon. 

-p <tekstin_rivi>
   Tallenna jäsennetty teksti tallentamiseen tarkoitettuun tiedostoon (esim. tekstitiedostoon). Jos tätä parametria ei käytetä, maalitiedoston nimi on sama kuin lähteen.
   Use the %FileName% variable to insert the input file name to the output file name.
   Use the %FirstLine% variable to insert the first line of text.
   Use the %Header% variable to insert the chapter title.
   Use the %Number% variable to change the position of the sequence number inside the output file name.
   Use the %Title% variable to insert the title of the HTML document (for HTML files only).
   Warning! It is necessary to double a percent sign (%) in a batch script. For example: -p %%FirstLine%%

-ext <tekstin_rivi>
   Set the extension for output filenames. The default is "txt".

-out <tiedoston_nimi>
   Sets the full name for output file. The option is recommended to specify only when the utility is used as a part of other software.
   If the utility is used for custom document import,  the external program runs the utility from a command line and passes the full name of a text file to create.

-s
   Search input files in subfolders.

--create-folder tai -cf
   Create an output subfolder for each input file. A file name will be used as a name of a subfolder.

-i
   Lue teksti standardisyöttövirrassa (STDIN) oleva teksti. Jos parametri on mainittu, parametri [-f] on välittämättä.

-o
   kirjoita teksti standarditulostevirtaan (STDOUT). Jos parametri on mainittu, parametrit [-v] ja [-p] ovat välittämättä.

-u
   Yhdistä teksti muutamista tiedostoista yhteen.

-b
   Lisää järjestysnumero tiedostonimen alkuun.

-a
   Lisää järjestysnumero tiedostonimen loppuun.

-n <luku>
   Määrää ensimmäinen järjestysnumero. Oletusnumero on 1.

-e <merkistö>
   Määrää jäsennetyn tekstin merkistö ("ansi", "utf8" tai "unicode"). Oletusmerkistö on "ansi".

-t <luku>
   Määrää tekstin jäsentämistapa: käytä määrätty tiedostokoko (as a number of characters).

-k <avainsana>
   Määrää tekstin jäsentämistapa: etsi avainsana. Parametri on merkkikokoriippuvainen. Komentoon voi sisältyä muutama [-k] -parametri.

-r <avainsana>
   Jaa teksti avainsanalla ja poista tämä tekstistä. Parametri on merkkikokoriippuvainen. Komentoon voi sisältyä muutama [-r] -parametri.

-w
   Määrää tekstin jäsentämistapa: etsi kaksi peräkkäistä tyhjää riviä.

-l
   Määrää tekstin jäsentämistapa: etsi rivi, jossa kaikki kirjaimet ovat suuraakkosia.

-c
   Splits text by a table of contents. The application extracts positions of chapter beginnings from the input file (if the file contains such information).

-toc
   Generates a table of contents and splits text. The application splits the extracted text by keywords (like "chapter" or "volume").
   If the option is used together with the option [-c], the application will try to extract a table of contents from the document; if it fails, a new table of contents will be generated.

-m <luku>
   Sets the minimal size of text parts for splitting (as a number of characters).

-j <luku>
   Ignores the chapter beginning if the size of the previous chapter is less than the specified value (in characters). The option is used together with the option [-c] or [-toc].

-hh <tekstin_rivi>
   Inserts text in front of headings (for example: ## Chapter 1).
 
-d <tiedoston_nimi>
   Käytä sanakirja ääntämyksen korjaamiseksi (tiedostolaajennus *.BXD, *.DIC tai *.REX). Komentoon voi sisältyä muutama [-d] -parametri.

-if
   Uses IFilter interface to extract text. If this fails, the default method will be used by the application.

-g <kansion_nimi>
   Sets the name of output folder for saving of images from a document.

-cvr <kansion_nimi>
   Sets the name of output folder for saving of a book cover image.

--clone-file-time tai -cft
   Clone the Created/Modified/Accessed time of the input file into the output file. If the application combines text files or splits the extracted text, the option is ignored.

-x <file_type>
   Sets the input file type. It allows to define a format of input documents with unknown file name extensions. For example: -x doc

-pwd <tekstin_rivi>
   Sets the password for the encrypted PDF files.

-dll <tiedoston_nimi>
   Sets the path and name for 7z.dll (32bit). This library helps to extract text and images from documents inside archives (ZIP, RAR, etc.). 7z.dll is a part of 7-Zip software.
   If the option is not specified, the application and the library must be in the same folder; otherwise, the application will not be able to extract data from archive files.

-dex <file_types>
   Sets the list of file types for extracting from archives. The option contains a comma-separated list of file types, for example: -dex "fb2,epub"
   The command line may contain few options [-dex]. If the option is not specified, the application will extract text from all files in an archive.
   If it is necessary to extract text for all file types supported by the application, use the value "all-". For example: -dex all-

-dne <file_types>
   Sets the list of file types to ignore when documents are extracted from archives. The option contains a comma-separated list of file types, for example: -dne "exe,dll"
   The command line may contain few options [-dne]. If the option is not specified, the application will extract text from all files in an archive.

-dp
   Display progress information in a console window.

-cfg <file_name>
   Sets the name of the configuration file with the command line options (a text file where each line contains one option). If the option is not specified, the file "blb2txt.cfg" in the same folder as the utility will be used.

-h
   Näytä kaikki mahdolliset parametrit.

--remove-spaces tai -rs
   Poista ylimääräiset välit (kaksi tai enemmän peräkkäistä väliä tai kiinteät välit).

--remove-hyphens tai -rh
   Poista rivien lopussa olevat yhdysmerkit.

--remove-linebreaks tai -rl
   Poista kappaleiden sisäiset rivinvaihdot.

--remove-empty-lines tai -rm
   Poista kaikki tyhjät rivit.

--replace-empty-lines tai -rp
   Korvaa muutama tyhjä rivi yhdellä tyhjällä rivillä.

--remove-square-brackets tai -rsb
   Poista teksti [hakasulkujen] sisältä.

--remove-curly-brackets tai -rcb
   Poista teksti {aaltosuluista} sisältä.

--remove-angle-brackets tai -rab
   Poista teksti <kulmasulkujen> sisältä.

--remove-round-brackets tai -rrb
   Removes text in (round brackets).

--remove-comments tai -rc
   Removes comments. Single-line comments start with // and continue until the end of the line. Multiline comments start with /* and end with */.

--remove-page-numbers tai -rpn
   Removes page numbers (it may be useful for DjVu/PDF files).
 
--fix-ocr-errors tai -ocr
   Korjaa tekstintunnistuksen aikana syntyneet virheet (sovellettava vain kyrillistä kirjaimistoa käytettävissä kielissä).

--fix-letter-spacing tai -ls
   Fixes letter-spacing in words (for example: s p a c e, _w_o_r_d).

--add-period tai -ap
   Adds a period if there is no punctuation after the last word of the paragraph.

--extract-summary <luku> tai -es <luku>
   Extracts a summary (also called "annotation") from FB2/FB3 files and inserts at the beginning of text.
   Possible values for the integer parameter:
   0 – skips a summary (this value is used by default);
   1..5 – extracts a summary (a value determines the order in which an author name and a book title are listed).

--skip-notes tai -sn
   Skips notes, when the application extracts text from DOCX/FB2/FB3/MD/ODT files.

--include-notes <luku> tai -in <luku>
   Includes notes inside text, when the application extracts text from DOCX/FB2/FB3/MD/ODT files.
   Possible values for the integer parameter:
   0 – removes links to notes from text;
   1 – keeps default positions of notes inside text (this value is used by default);
   2 – places notes at the end of sentences;
   3 – places notes at the end of paragraphs.

--insert-note-begin <tekstin_rivi> tai -inb <tekstin_rivi>
   Inserts words at the beginning of notes, when notes are included inside text (for example: Editor's note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--insert-note-end <tekstin_rivi> tai -ine <tekstin_rivi>
   Inserts words at the end of notes, when notes are included inside text (for example: End of note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--extract-tables <luku> tai -et <luku>
   Extract tables from DOCX/FB2/FB3/ODT files.
   Possible values for the integer parameter:
   0 – skips tables;
   1 – extract data from each cell as a new text line (this value is used by default);
   2 – keep formatting when extracting a table.

--csv-comma
   Columns are separated by a comma, when the application extracts data from XLS/XLSX/ODS files (default delimiter for CSV files).

--csv-semicolon
   Columns are separated by a semicolon, when the application extracts data from XLS/XLSX/ODS files.

--csv-space
   Columns are separated by a blank space, when the application extracts data from XLS/XLSX/ODS files.

--csv-tab
   Columns are separated by a tab, when the application extracts data from XLS/XLSX/ODS files.

--csv-double-quote
   Uses double-quote characters, if a field must be quoted (export from XLS/XLSX/ODS files).

--csv-single-quote
   Uses single-quote characters, if a field must be quoted (export from XLS/XLSX/ODS files).

--eml-save <kansion_nimi>
   Extracts attachments from EML files and saves to a specified folder.

--eml-att
   Extracts the list of attachments from EML files (names of files attached to the message).

--eml-cc
   Extracts the header field "Cc" from EML files ("carbon copy"; it specifies additional recipients of the message).

--eml-date <date_format>
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


*** Esimerkkejä ***

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\"

blb2txt -f "d:\Docs\book.doc" -out "d:\Text\book.txt"

blb2txt -f "d:\Docs\*.doc" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 --replace-empty-lines

blb2txt -f "d:\Docs\*.*" -v "d:\Text\" -p "Tiedosto" -u

blb2txt -f "d:\Docs\1.doc" -v "d:\Text\" -p "Tiedosto" -a -n 20 -t 100000

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "Kirja" -k "KAPPALE" -k "SISÄLTÖ"

blb2txt -f "d:\Book\book.epub" -v "d:\Text\" -p "Kirja" -r "###"

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\" -d "d:\rex\rules.rex" -d "d:\dic\rules.dic" --remove-spaces --remove-linebreaks

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Konfiguraatiotiedosto ***

Parametrit voi tallentaa "blb2txt.cfg" -nimiseen konfiguraatiotiedostoon, joka sijaitsee sovelluksen kansiossa.

Tiedoston sisällyksen esimerkki:
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

Sovellus voi yhdistää konfiguraatiotiedostossa olevat ja komentorivin parametrit.


*** Toiminta ***

Sovellus toimii seuraavasti:

1.  Tekstin tiedostosta jäsentäminen.
2.  Tekstin muotoilu: ylimääräisten välien ja rivinvaihtojen poistaminen jne. (jos vastaavat parametrit sisältyvät komentoon).
3.  Teksti samaan tiedostoon yhdistäminen (jos vastaava parametri sisältyy komentoon).
4.  Teksti jakaminen (jos vastaavat parametrit sisältyvät komentoon).
5.  Ääntämyksen korjaussääntöjen soveltaminen (jos vastaavat parametrit sisältyvät komentoon).
6.  Tiedoston levylle tallentaminen.


*** Lisenssi ***

Sovelluksen kaupallinen käyttöoikeus:
- toiminimelle rajoituksetta;
- oikeushenkilöitä koskevat rajoitukset on mainittu Balabolkan lisenssisopimuksessa.

Sovelluksen kaupallinen käyttö sallittu ainoastaan oikeudenomistajan luvasta.

###
