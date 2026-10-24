Balabolka (Text Extract Utility), version 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW: https://www.cross-plus-a.com/btext.htm
E-mail: crossa@list.ru

License: Freeware
Operating System: Microsoft Windows 7/8/10/11


The application allows to extract text from the various types of files.
The extracted text can be combined into one file or/and split into few files.
The list of rules for pronunciation correction from Balabolka can be applied to text.
Supported formats for input files: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XSL, XSLX.
The IFilter interface will be used for files with unknown extensions.

The application can use the library 7z.dll (32bit). It helps to extract text and images from documents inside archives (ZIP, RAR, etc.). 7z.dll is a part of 7-Zip software.
https://www.7-zip.org


*** Usage ***

blb2txt [options ...]


*** Command Line Options ***

-f <file_mask>
   Sets the name of input file or the mask for the group of input files. The command line may contain few options [-f].

-fl <file_name>
   Sets the name of the text file with the list of input files (one file name per line). The command line may contain few options [-fl].

-v <folder_name>
   Sets the name of output folder for saving of text files.

-p <text>
   Sets the pattern for output file name (for example, "Text Document"). If absent, the input file name will be used as a pattern.
   Use the %FileName% variable to insert the input file name to the output file name.
   Use the %FirstLine% variable to insert the first line of text.
   Use the %Header% variable to insert the chapter title.
   Use the %Number% variable to change the position of the sequence number inside the output file name.
   Use the %Title% variable to insert the title of the HTML document (for HTML files only).
   Warning! It is necessary to double a percent sign (%) in a batch script. For example: -p %%FirstLine%%

-ext <text>
   Sets the extension for output filenames. The default is "txt".

-out <file_name>
   Sets the path and name for an output file. The option is recommended to specify only when the utility is used as a part of other software.
   If the utility is used for custom document import, the external program runs the utility from a command line and passes the full name of a text file to create.

-s
   Searches input files in subfolders.

--create-folder or -cf
   Creates an output subfolder for each input file. A file name will be used as a name of a subfolder.

-i
   Reads data from STDIN. The file format will be auto-detected from data. If the option is specified, the option [-f] is ignored.

-o
   Writes text to STDOUT. If the option is specified, the options [-v] and [-p] are ignored.

-u
   Combines text files into one output file.

-b
   Adds sequence number before output file name (when text is split).

-a
   Adds sequence number after output file name (when text is split).

-n <integer>
   Sets the starting sequence number for output files (when text is split). The default is 1.

-e <encoding>
   Sets the encoding for output files ("ansi", "utf8" or "unicode"). The default is "ansi".

-t <integer>
   Splits text by output target size of text parts (as a number of characters).

-k <keyword>
   Splits text by a special keyword in input file. The option is case-sensitive. The command line may contain few options [-k].

-r <keyword>
   Splits text by a keyword and removes it from output files. The option is case-sensitive. The command line may contain few options [-r].

-w
   Splits text by two empty lines in succession.

-l
   Splits text by lines where all letters are capital.

-c
   Splits text by a table of contents. The application extracts positions of chapter beginnings from the input file (if the file contains such information).

-toc
   Generates a table of contents and splits text. The application splits the extracted text by keywords (like "chapter" or "volume").
   If the option is used together with the option [-c], the application will try to extract a table of contents from the document; if it fails, a new table of contents will be generated.

-m <integer>
   Sets the minimal size of text parts for splitting (as a number of characters).

-j <integer>
   Ignores the chapter beginning if the size of the previous chapter is less than the specified value (in characters). The option is used together with the option [-c] or [-toc].

-hh <text>
   Inserts text in front of headings (for example: ## Chapter 1).

-d <file_name>
   Applies a dictionary for pronunciation correction (*.BXD, *.DIC or *.REX). The command line may contain few options [-d].
   You may use the desktop application 'Balabolka' to edit a dictionary.

-if
   Uses IFilter interface to extract text. If this fails, the default method will be used by the application.

-g <folder_name>
   Sets the name of output folder for saving of images from a document.

-cvr <folder_name>
   Sets the name of output folder for saving of a book cover image.

--clone-file-time or -cft
   Clones the Created/Modified/Accessed time of the input file into the output file. If the application combines text files or splits the extracted text, the option is ignored.

-x <file_type>
   Sets the input file type. It allows to define a format of input documents with unknown file name extensions. For example: -x doc

-pwd <text>
   Sets the password for the encrypted PDF files.

-dll <file_name>
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
   Prints the list of available command line options.

--remove-spaces or -rs
   Removes excess spaces (two or more blank spaces in succession, no-break spaces).

--remove-hyphens or -rh
   Removes hyphens at the ends of lines in the text.

--remove-linebreaks or -rl
   Removes linebreaks inside paragraphs.

--remove-empty-lines or -rm
   Removes empty lines.

--replace-empty-lines or -rp
   Replaces few empty lines by one empty line.

--remove-square-brackets or -rsb
   Removes text in [square brackets].

--remove-curly-brackets or -rcb
   Removes text in {curly brackets}.

--remove-angle-brackets or -rab
   Removes text in <angle brackets>.

--remove-round-brackets or -rrb
   Removes text in (round brackets).

--remove-comments or -rc
   Removes comments. Single-line comments start with // and continue until the end of the line. Multiline comments start with /* and end with */.

--remove-page-numbers or -rpn
   Removes page numbers (it may be useful for DjVu/PDF files).

--fix-ocr-errors or -ocr
   Fixes OCR errors (for languages with Cyrillic alphabets only).

--fix-letter-spacing or -ls
   Fixes letter-spacing in words (for example: s p a c e, _w_o_r_d).

--add-period or -ap
   Adds a period if there is no punctuation after the last word of the paragraph.

--extract-summary <integer> or -es <integer>
   Extracts a summary (also called "annotation") from FB2/FB3 files and inserts at the beginning of text.
   Possible values for the integer parameter:
   0 - skips a summary (this value is used by default);
   1..5 - extracts a summary (a value determines the order in which an author name and a book title are listed).

--skip-notes or -sn
   Skips notes, when the application extracts text from DOCX/FB2/FB3/MD/ODT files.

--include-notes <integer> or -in <integer>
   Includes notes inside text, when the application extracts text from DOCX/FB2/FB3/MD/ODT files.
   Possible values for the integer parameter:
   0 - removes links to notes from text;
   1 - keeps default positions of notes inside text (this value is used by default);
   2 - places notes at the end of sentences;
   3 - places notes at the end of paragraphs.

--insert-note-begin <text> or -inb <text>
   Inserts words at the beginning of notes, when notes are included inside text (for example: Editor's note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--insert-note-end <text> or -ine <text>
   Inserts words at the end of notes, when notes are included inside text (for example: End of note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--extract-tables <integer> or -et <integer>
   Extract tables from DOCX/FB2/FB3/ODT files.
   Possible values for the integer parameter:
   0 - skips tables;
   1 - extracts data from each cell as a new text line (this value is used by default);
   2 - keeps formatting when extracting a table.

--csv-comma
   Columns are separated by a comma, when the application extracts data from XLS/XLSX/ODS files (default delimiter for CSV files).

--csv-semicolon
   Columns are separated by a semicolon, when the application extracts data from XLS/XLSX/ODS files.

--csv-space
   Columns are separated by a blank space, when the application extracts data from XLS/XLSX/ODS files.

--csv-tab
   Columns are separated by a tab, when the application extracts data from XLS/XLSX/ODS files.

--csv-double-quote
   Uses double-quote characters if a field must be quoted (export from XLS/XLSX/ODS files).

--csv-single-quote
   Uses single-quote characters if a field must be quoted (export from XLS/XLSX/ODS files).

--eml-save <folder_name>
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


*** Examples ***

Extract text from "book.doc" and save as "book.txt" to the output folder:

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\"


Also this variant can be used if necessary (when the only one input file is specified):

blb2txt -f "d:\Docs\book.doc" -out "d:\Text\book.txt"


Extract text from BOOK.DOC and save as "New Book.txt":

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\" -p "New Book"


Extract text from the Microsoft Word and RTF documents, remove empty lines and save text files in UTF-8 encoding:

blb2txt -f "d:\Docs\*.doc" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Extract text from all text files in the specified folder, unite and save as "Document.txt":

blb2txt -f "d:\Docs\*.txt" -v "d:\Text\" -p "Document" -u


Extract text from DOCUMENT.DOC, divide on parts with size 100 KB and save as text files "Document 20.txt", "Document 21.txt", etc.:

blb2txt -f "d:\Docs\document.doc" -v "d:\Text\" -p "Document" -a -n 20 -t 100000


Extract text from BOOK.FB2, find the words "CHAPTER" and "CONTENTS" to divide text on parts and save as files with the names "Book 1.txt", "Book 2.txt", etc.:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "Book" -k "CHAPTER" -k "CONTENTS"


Extract text from BOOK.EPUB, find "###" to divide text on parts, remove "###" from text and save each part as a new file:

blb2txt -f "d:\Book\book.epub" -v "d:\Text\" -p "Book" -r "###"


Extract text from BOOK.FB2, split by a table of contents, save files and use chapter titles as file names. New text files must not be less than one kilobyte:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024


Get text from STDIN, remove excess spaces, linebreaks and empty lines, write the updated text to STDOUT:

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Extract text from all Wicrosoft Word documents inside ZIP archives:

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Configuration File ***

The command line options can be stored as a configuration file "blb2txt.cfg" in the same folder as the utility.

The sample configuration file:
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

The utility may combine options from the configuration file and the command line.


*** Operations ***

The application executes operations in next order:

1.  Extract text from input file(s).
2.  Format text: remove spaces, linebreaks, etc. (if options are specified).
3.  Combine files into one file (if option is specified).
4.  Split text (if options are specified).
5.  Apply rules for pronunciation correction (if option is specified).
6.  Save output file(s).


*** Using Table of Contents ***

The application allows to split text by table of contents. There are two way to get a table of contents:
- extract information about a table of contents from a document; it is available for next formats: AZW, CHM, DOCX, EPUB, FB2, HTML, LIT, MHT, MOBI, ODT, PDB, PRC;
- generate a new table of contents, by using keywords found in text (like "chapter" or "volume"); it is available for next languages: English, Armenian, Belarusian, Bulgarian, Czech, French, German, Italian, Polish, Russian, Spanish, Ukrainian.
Be careful when use this option: the result of text splitting may be unpredictable.

For example, HTML uses the <H1> to <H6> tags to define headings. <H1> defines the most important heading. <H6> defines the least important heading.
The application uses all kinds of the <H> tag to generate a table of contents. The result may contain many items.


*** Licence ***

You are free to use and distribute software for non-commercial purposes. For commercial use or distribution, you need to get permission from the copyright holder. The application can not be used on the territory of Afghanistan, Belarus, Cuba, Iran, Nicaragua, North Korea, Russian Federation, Syria.

###
