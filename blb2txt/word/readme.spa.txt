Balabolka (programa para extraer texto del archivo), versión 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW: https://www.cross-plus-a.com/es/btext.htm
E-mail: crossa@list.ru

Licencia: Gratuito (Freeware)
Sistema operativo: Microsoft Windows 7/8/10/11


El programa está diseñado para extraer textos de archivos de diferentes formatos.
El texto extraído puede ser reunido en un solo archivo y / o distribuido en varios archivos.
Al texto se le puede aplicar las reglas de los diccionarios de corrección de pronunciación del programa Balabolka.
Se soportan los siguientes formatos de archivo: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
La interfaz IFilter se utilizará para archivos con extensiones desconocidas.

La aplicación puede utilizar la biblioteca 7z.dll (32 bits). Ayuda a extraer texto e imágenes de documentos dentro de archivos comprimidos (ZIP, RAR, etc.). 7z.dll forma parte del software 7-Zip.
https://www.7-zip.org


*** Línea de comandos ***

blb2txt [opciones ...]


*** Opciones del comando ***

-f <nombre_de_archivo>
   El nombre de archivo o una máscara para los nombres de los archivos de los cuales se extrae el texto. La línea de comandos puede contener varias opciones [-f].

-fl <nombre_de_archivo>
   Sets the name of the text file with the list of input files (one file name per line). La línea de comandos puede contener varias opciones [-fl].

-v <nombre_de_carpeta>
   Nombre de la carpeta para salvar el archivo con el texto extraído.

-p <texto>
   Plantilla para el nombre del archivo con el texto extraído (por ejemplo, "Documento de texto"). Si no se especifica la opción se usa el nombre del archivo de origen.
   Utilice la variable %FileName% para insertar el nombre del archivo de entrada en el nombre del archivo de salida.
   Utilice la variable %FirstLine% para insertar la primera línea de texto.
   Utilice la variable %Header% para insertar el título del capítulo.
   Utilice la variable %Number% para cambiar la posición del número de secuencia dentro del nombre del archivo de salida.
   Utilice la variable %Title% para insertar el título del documento HTML (solo para archivos HTML).
   ¡Atención! Es necesario duplicar el signo de porcentaje (%) en un archivo de script. Por ejemplo: -p %%FirstLine%%

-ext <texto>
   Establece la extensión para los nombres de los archivos de salida. El valor predeterminado es "txt".

-out <nombre_de_archivo>
   Establece el nombre completo y la ruta del archivo de salida. Se recomienda especificar esta opción solo cuando la utilidad se utiliza como parte de otro software.

-s
   Buscar archivos de entrada en subcarpetas.

--create-folder o -cf
   Cree una subcarpeta de salida para cada archivo de entrada. El nombre del archivo se utilizará como nombre de la nueva subcarpeta.

-i
   Leer el texto del flujo de entrada estándar (STDIN). Si se especifica la opción, se ignora la opción [-f].

-o
   Inscribir el texto en el flujo de salida estándar (STDOUT). Si se especifica la opción, se ignoran las opciones [-v] y [-p].

-u
   Reunir textos de varios archivos en un solo archivo.

-b
   Añadir el número de orden antes del nombre del archivo.

-a
   Añadir el número de orden después del nombre del archivo.

-n <entero>
   Establecer el número de orden inicial del archivo. El valor predeterminado es 1.

-e <codificación>
   Codificación del archivo con el texto extraído ("ansi", "utf8" o "unicode"). El valor predeterminado es "ansi".

-t <entero>
   Especificar el modo de partición del texto: uso de un tamaño predeterminado del archivo (as a number of characters).

-k <palabra_clave>
   Especificar el modo de partición del texto: búsqueda de palabra clave en el archivo de origen. Esta opción distingue entre mayúsculas y minúsculas. La línea de comandos puede contener varias opciones [-k].

-r <palabra_clave>
   Dividir el texto con la palabra clave y retirarla del texto. Esta opción distingue entre mayúsculas y minúsculas. La línea de comandos puede contener varias opciones [-r].

-w
   Especificar el modo de partición del texto: buscar dos líneas en blanco seguidas.

-l
   Especificar el modo de partición del texto: buscar el texto donde todas las letras son mayúsculas.

-c
   Dividir el texto por un índice. La aplicación extrae las posiciones de los comienzos de los capítulos del archivo de entrada (o se generará un nuevo índice si se especifica la opción -toc).

-toc
   Generar un índice y dividir el texto. La aplicación divide el texto extraído por palabras clave (como "capítulo").
   Si se utiliza junto con la opción [-c], la aplicación intentará extraer un índice del documento; si falla, se generará un nuevo índice.

-m <entero>
   Determinar el tamaño mínimo de las partes de texto para dividir (como número de caracteres).

-j <entero>
   Ignora el comienzo del capítulo si el tamaño del capítulo anterior es inferior al valor especificado (en caracteres). La opción se utiliza junto con la opción [-c] o [-toc].

-hh <texto>
   Inserta texto delante de los encabezados (por ejemplo: ## Capítulo 1).

-d <nombre_de_archivo>
   Usar el diccionario para la corrección de la pronunciación (archivo con extensión *.BXD, *.DIC o *.REX). La línea de comandos puede contener varios parámetros [-d].

-if
   Utiliza la interfaz IFilter para extraer texto. Si esto falla, la aplicación utilizará el método predeterminado.

-g <nombre_de_carpeta>
   Definir el nombre de la carpeta de salida para guardar las imágenes de un documento.

-cvr <nombre_de_carpeta>
   Definir el nombre de la carpeta de salida para guardar la imagen de la portada del libro.

--clone-file-time o -cft
   Clonar la hora de Creación/Modificación/Acceso del archivo de entrada en el archivo de salida. Si la aplicación combina archivos de texto o divide el texto extraído, se ignora la opción.

-x <tipo_de_archivo>
   Definir el tipo de archivo de entrada. Permite definir un formato de documentos de entrada con extensiones de nombre de archivo desconocidas. Por ejemplo: -x doc

-pwd <texto>
   Designar una contraseña para sacar el texto del archivo en formato PDF.

-dll <nombre_de_archivo>
   Definir la ruta y el nombre de 7z.dll (32 bits). Esta biblioteca ayuda a extraer texto e imágenes de documentos dentro de archivos comprimidos (ZIP, RAR, etc.). 7z.dll forma parte del software 7-Zip. Si no se especifica la opción, la aplicación y la biblioteca deben estar en la misma carpeta; de lo contrario, la aplicación no podrá extraer datos de los archivos comprimidos.

-dex <tipos_de_archivo>
   Sets the list of file types for extracting from archives. The option contains a comma-separated list of file types, for example: -dex "fb2,epub"
   The command line may contain few options [-dex]. If the option is not specified, the application will extract text from all files in an archive.
   If it is necessary to extract text for all file types supported by the application, use the value "all-". For example: -dex all-

-dne <tipos_de_archivo>
   Sets the list of file types to ignore when documents are extracted from archives. The option contains a comma-separated list of file types, for example: -dne "exe,dll"
   The command line may contain few options [-dne]. If the option is not specified, the application will extract text from all files in an archive.

-dp
   Muestra información del progreso en una ventana de la consola.

-cfg <nombre_de_archivo>
   Establece el nombre del archivo de configuración con las opciones de la línea de comandos (un archivo de texto en el que cada línea contiene una opción). Si no se especifica la opción, se utilizará el archivo "blb2txt.cfg" que se encuentra en la misma carpeta que la utilidad.

-h
   Mostrar descripción de la línea de comandos.

--remove-spaces o -rs
   Eliminar espacios en blanco (dos o más espacios seguidos, espacios sin separación).

--remove-hyphens o -rh
   Eliminar guiones en los extremos de líneas en el texto.

--remove-linebreaks o -rl
   Eliminar saltos de línea dentro de un párrafo.

--remove-empty-lines o -rm
   Eliminar todas las líneas en blanco.

--replace-empty-lines o -rp
   Reemplazar múltiples líneas en blanco una sola línea en blanco.

--remove-square-brackets o -rsb
   Eliminar el texto entre [corchetes].

--remove-curly-brackets o -rcb
   Eliminar el texto entre {llaves}.

--remove-angle-brackets o -rab
   Eliminar el texto entre <paréntesis angulares>.

--remove-round-brackets o -rrb
   Eliminar el texto entre (paréntesis redondos).

--remove-comments o -rc
   Borra los comentarios. Los comentarios de una sola línea comienzan con // y continúan hasta el final de la línea. Los comentarios de varias líneas comienzan con /* y terminan con */.

--remove-page-numbers o -rpn
   Eliminar los números de página (puede ser útil para archivos DjVu/PDF).

--fix-ocr-errors o -ocr
   Corregir errores ocurridos en OCR (sólo para idiomas con alfabeto cirílico).

--fix-letter-spacing o -ls
   Corrija el espaciado entre letras en las palabras (por ejemplo: e s p a c i o, p_a_l_a_b_r_a).

--add-period o -ap
   Añade un punto si no hay signos de puntuación después de la última palabra de un párrafo.

--extract-summary <entero> o -es <entero>
   Extrae un resumen (también llamado "anotación") de archivos FB2/FB3 y lo inserta al principio del texto.
   Valores posibles para el parámetro entero:
   0 – omite el resumen (utilizado por defecto);
   1..5 – extrae un resumen (un valor determina el orden en que se enumeran el nombre del autor y el título del libro).

--skip-notes o -sn
   Skips notes, when the application extracts text from DOCX/FB2/FB3/MD/ODT files.

--include-notes <entero> o -in <entero>
   Incluye notas dentro del texto, cuando la aplicación extrae texto de archivos DOCX/FB2/FB3/MD/ODT.
   Valores posibles para el parámetro entero:
   0 – removes links to notes from text;
   1 – keeps default positions of notes inside text (this value is used by default);
   2 – places notes at the end of sentences;
   3 – places notes at the end of paragraphs.

--insert-note-begin <texto> o -inb <texto>
   Inserts words at the beginning of notes, when notes are included inside text (for example: Editor's note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--insert-note-end <texto> o -ine <texto>
   Inserts words at the end of notes, when notes are included inside text (for example: End of note.).
   The option is used for DOCX/FB2/FB3/MD/ODT files.

--extract-tables <entero> o -et <entero>
   Extract tables from DOCX/FB2/FB3/ODT files.
   Valores posibles para el parámetro entero:
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

--eml-save <nombre_de_carpeta>
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


*** Ejemplos ***

Extraiga el texto de LIBRO.DOC y guárdelo como LIBRO.TXT en la carpeta de salida:

blb2txt -f "d:\Docs\libro.doc" -v "d:\Texto\"


También se puede utilizar esta variante si es necesario (cuando solo se especifica un archivo de entrada):

blb2txt -f "d:\Docs\libro.doc" -out "d:\Texto\libro.txt"


Extraiga texto de documentos Microsoft Word y RTF, elimine las líneas vacías y guarde los archivos de texto con codificación UTF-8:

blb2txt -f "d:\Docs\*.docx" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Extraiga el texto de todos los archivos de texto de la carpeta especificada, únalos y guárdelos como DOCUMENTO.TXT:

blb2txt -f "d:\Docs\*.*" -v "d:\Texto\" -p "Documento" -u


Extraiga el texto de DOCUMENT.DOCX, divídalo en partes de 100 KB y guárdelo como archivos de texto "Documento 20.txt", "Documento 21.txt", etc.:

blb2txt -f "d:\Docs\Document.docx" -v "d:\Texto\" -p "Documento" -a -n 20 -t 100000


Extraiga el texto de LIBRO.FB2, busque las palabras "CAPITULO" e "Indice" para dividir el texto en partes y guárdelo como archivos con los nombres "Libro 1.txt", "Libro 2.txt", etc.:

blb2txt -f "d:\Book\libro.fb2" -v "d:\Texto\" -p "Libro" -k "CAPITULO" -k "Indice"


Extraiga el texto de LIBRO.EPUB, busque "###" para dividir el texto en partes, elimine "###" del texto y guarde cada parte como un nuevo archivo:

blb2txt -f "d:\Book\libro.epub" -v "d:\Texto\" -p "Libro" -r "###"


Extraiga el texto de LIBRO.FB2, divídalo por el índice, guarde los archivos y utilice los títulos de los capítulos como nombres de archivo. Los nuevos archivos de texto no deben tener menos de un kilobyte:

blb2txt -f "d:\Book\libro.fb2" -v "d:\Texto\" -p "%Number% - %Header%" -c -j 1024


Obtener texto de STDIN, eliminar espacios sobrantes, saltos de línea y líneas vacías, escribir el texto actualizado en STDOUT:

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Extraiga texto de todos los documentos de Microsoft Word dentro de archivos ZIP:

blb2txt -f "d:\Archivo\*.zip" -v "d:\Texto\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Archivo de configuración ***

Se puede guardar el archivo de configuración "blb2txt.cfg" en la misma carpeta que la aplicación de consola.

Un ejemplo del contenido del archivo:
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

El programa puede combinar opciones del archivo de configuración y de la línea de comandos.


*** Operaciones ***

El programa cumple operaciones en el orden siguiente:

1.  Extraer el texto del archivo.
2.  Formatear el texto: eliminar espacios extra, saltos de línea, etc. (si tales opciones están especificadas).
3.  Reunir textos en un solo archivo (si tal opción está especificada).
4.  Dividir el texto en partes (si tales opciones están especificadas).
5.  Aplicar las reglas de corrección de pronunciación (si tales opciones están especificadas).
6.  Salvar el archivo o los archivos en el disco.


*** Licencia ***

Derecho al uso no comercial del programa:
- para las personas naturales: sin ningún tipo de restricciones;
- para las personas jurídicas: sujeto a las restricciones que figuran en el "Contrato de licencia" del software Balabolka.

El uso comercial del programa sólo se permite con previa autorización del titular de derechos de autor.

###
