Balabolka (utilitaire d'extraction de texte du fichier), version 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW : https://www.cross-plus-a.com/fr/btext.htm
E-mail : crossa@list.ru

Licence : Freeware
Système d'exploitation : Microsoft Windows 7/8/10/11


Le programme permet d'extraire le texte de différents types de fichiers.
L’extrait du texte peut être combiné en un seul fichier et/ou fractionné en plusieurs fichiers.
Les règles de correction de la prononciation de Balabolka peuvent être appliquées au texte.
Les formats suivants sont soutenus : AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
L’interface IFilter sera utilisée pour les extensions de fichiers inconnues.

L'application peut utiliser la bibliothèque 7z.dll (32 bits). Elle permet d'extraire du texte et des images à partir de documents contenus dans des archives (ZIP, RAR, etc.). 7z.dll fait partie du logiciel 7-Zip.
https://www.7-zip.org


*** Ligne de commande ***

blb2txt [options ...]


*** Paramètres de ligne de commande ***

-f <nom_de_fichier>
   Le nom du fichier d'entrée ou le masque pour un groupe de fichiers d'entrée. La ligne de commande peut contenir quelques options [-f].

-fl <nom_de_fichier>
   Ouvrir le fichier avec la liste des fichiers texte (un nom de fichier par ligne). La ligne de commande peut contenir quelques options [-fl].

-v <nom_de_dossier>
   Le nom du dossier de sortie pour un fichier texte enregistré.

-p <texte>
   Spécifie le modèle pour le nom de fichier de sortie (par exemple, « Document texte »). En cas d'absence, le nom du fichier d'entrée est utilisé.
   Utilisez la variable %FileName% dans le modèle du nom de fichier pour insérer le nom du fichier source (sans extension).
   Utilisez la variable %FirstLine% pour insérer la première ligne de texte dans le nom du fichier de sortie.
   Utilisez la variable %Header% pour insérer un titre du sommaire du document.
   Utilisez la variable %Number% pour modifier la position du numéro de séquence dans le nom du fichier de sortie.
   Utilisez la variable %Title% pour insérer le titre du document HTML (pour les fichiers HTML uniquement).
   Attention ! Le caractère % doit être doublé dans le fichier batch. Par exemple : -p %%FirstLine%%

-ext <texte>
   L’extension du nom de fichier avec un texte extrait. La valeur par défaut est "txt".

-out <nom_de_fichier>
   Le nom complet du fichier de sortie. Il est recommandé d’utiliser ce paramètre uniquement lorsque l’utilitaire est utilisé dans le cadre d’un autre logiciel.
   Si l’utilitaire est utilisé comme programme externe pour l’extraction de texte, la ligne de commande de l’utilitaire contiendra le nom du fichier d’entrée et le nom du fichier de sortie.

-s
   Rechercher les fichiers dans les sous-dossiers.

--create-folder ou -cf
   Créer un sous-dossier pour chaque fichier à partir duquel le texte est extrait. Le nom du fichier sera le nom du sous-dossier.

-i
   Lit le texte de flux d'entrée standard (STDIN). Si l'option est spécifiée, l'option [-f] est ignorée.

-o
   Enregistre le texte dans le flux de sortie standard (STDOUT). Si l'option est spécifiée, les options [-v] et [-p] sont ignorées.

-u
   Combine les fichiers texte en un seul fichier de sortie.

-b
   Ajoute le numéro de séquence devant le nom de fichier de sortie.

-a
   Ajoute le numéro de séquence après le nom de fichier de sortie.

-n <nombre_intégral>
   Spécifie le numéro de séquence de départ pour les fichiers de sortie. La valeur par défaut est 1.

-e <encodage>
   Spécifie l'encodage pour les fichiers de sortie (« ansi », « utf8 » ou « unicode »). La valeur par défaut est « ansi ».

-t <nombre_intégral>
   Spécifie le mode de fractionnement du texte : fichier d’une taille spécifiée (as a number of characters).

-k <mot-clé>
   Fractionne le texte sur le mot-clé spécial dans le fichier d'entrée. L'option est sensible à la casse. La ligne de commande peut contenir plusieurs options [-k].

-r <mot-clé>
   Fractionne le texte sur le mot-clé et le supprime des fichiers de sortie. L'option est sensible à la casse. La ligne de commande peut contenir plusieurs options [-r].

-w
   Fractionne le texte sur deux lignes vides consécutives.

-l
   Fractionne le texte sur les lignes où toutes les lettres sont majuscules.

-c
   Définir une méthode de division du texte : utilisez le sommaire du document. Le programme extrait les positions du début des chapitres du livre électronique (si le fichier du livre contient de telles informations).

-toc
   Définir une méthode de division du texte : créer un sommaire et diviser en parties. Le programme divise le texte à l’aide des mots-clés trouvés (« chapitre », « volume », etc.).
   Si le paramètre est utilisé avec le paramètre [-c], le programme essaiera d’abord d’extraire le sommaire du document ; en cas d’échec, un nouveau sommaire sera créé.

-m <nombre_intégral>
   Définir la taille minimale de chaque partie lors de la division du texte. Le nombre désigne la quantité de caractères (y compris les espaces, la ponctuation, les caractères interligne et les retours chariot).

-j <nombre_intégral>
   Ignorer le début du chapitre suivant lors de la division du texte, si la taille du texte précédente est inférieure à la taille donnée. Le nombre désigne la quantité de caractères (y compris les espaces et les signes de ponctuation). Ce paramètre est utilisé avec les paramètres [-c] ou [-toc].

-hh <texte>
   Insérer un texte avant les titres (par exemple : ## Chapitre 1).

-d <nom_de_fichier>
   Utilise un dictionnaire pour la correction de la prononciation (fichiers *.BXD, *.DIC ou *.REX). La ligne de commande peut contenir plusieurs options [-d].

-if
   Utilise l'interface IFilter pour extraire le texte. Si ce type de format est absent dans le système, la méthode par défaut sera utilisée par l'application.

-g <nom_de_dossier>
   Le nom du dossier pour enregistrer les fichiers graphiques extraits du document.

-cvr <nom_de_dossier>
   Le nom du dossier pour enregistrer la couverture du livre extraite du document.

--clone-file-time ou -cft
   Le fichier texte créé aura la même date et heure de création que le fichier source. Le paramètre est ignoré si le texte est combiné à partir de plusieurs fichiers ou divisé en parties.

-x <file_type>
   Définir l’extension du document d’entrée. Cela permet de spécifier un type de fichier avec une extension inconnue. Par exemple : -x doc

-pwd <texte>
   Spécifie le mot de passe pour extraire le texte des fichiers PDF cryptés.

-dll <nom_de_fichier>
   Le chemin d’accès complet d’un fichier 7z.dll. Cette version 32 bits de la bibliothèque permet d’extraire des données des fichiers archives (ZIP, RAR, etc.). 7z.dll fait partie de l’archiveur 7-Zip.
   Si le paramètre n’est pas spécifié, le programme et la bibliothèque doivent se trouver dans le même dossier. Si le fichier 7z.dll est manquant, le programme ne pourra pas extraire les données des fichiers archives.

-dex <types_fichiers>
   Spécifier les formats des fichiers à extraire des archives. Les types de fichiers doivent être listés, séparés par des virgules, par exemple : -dex "fb2,epub"
   L’invite de commande peut contenir plusieurs paramètres [-dex]. Si le paramètre n’est pas spécifié, le programme extraira le texte des fichiers de tous les formats dans les fichiers archives.
   S’il faut extraire des données des fichiers uniquement dans les formats supportés par le programme, vous devez utiliser la valeur "all-". Par exemple : -dex all-

-dne <types_fichiers>
   Définir les formats des fichiers à ignorer lors de l’extraction des données des fichiers archives. Les types de fichiers doivent être listés, séparés par des virgules, par exemple : -dne "exe,dll"
   L’invite de commande peut contenir plusieurs paramètres [-dne]. Si le paramètre n’est pas spécifié, le programme extraira le texte des fichiers de tous les formats dans les fichiers archives.

-dp
   Afficher les informations sur l’avancement dans la fenêtre de console.

-cfg <file_name>
   Définit le nom du fichier de configuration contenant les options de ligne de commande (un fichier texte où chaque ligne contient une option). Si l'option n'est pas spécifiée, le fichier "blb2txt.cfg" situé dans le même dossier que l'utilitaire sera utilisé.

-h
   Affiche la liste des options de ligne de commande disponibles.

--remove-spaces ou -rs
   Supprime les espaces superflues (deux ou plusieurs espaces de suite, espaces insécables).

--remove-hyphens ou -rh
   Supprime tous les traits d'union à la fin des lignes.

--remove-linebreaks ou -rl
   Supprime les sauts de ligne à l'intérieur des paragraphes.

--remove-empty-lines ou -rm
   Supprime les lignes vides.

--replace-empty-lines ou -rp
   Remplace plusieurs lignes vides d’une seule ligne vide.

--remove-square-brackets ou -rsb
   Supprime le texte entre [crochets].

--remove-curly-brackets ou -rcb
   Supprime le texte entre {accolades}.

--remove-angle-brackets ou -rab
   Supprime le texte entre <chevrons>.

--remove-round-brackets ou -rrb
   Supprime le texte entre (parenthèses).

--remove-comments ou -rc
   Supprimer les commentaires. Les commentaires sur une seule ligne commencent par // et se poursuivent jusqu’à la fin de la ligne. Les commentaires multilignes commencent par /* et se terminent par */.

--remove-page-numbers ou -rpn
   Supprimer les numéros de page (cela peut être utile pour les fichiers DjVu/PDF).

--fix-ocr-errors ou -ocr
   Corrige les erreurs d'OCR (reconnaissance optique de caractères, pour les langues avec les alphabets cyrilliques uniquement).

--fix-letter-spacing ou -ls
   Supprimer les espaces entre les lettres (par exemple : e s p a c e, _m_o_t).

--add-period ou -ap
   Ajouter un point s’il n’y a pas de signe de ponctuation à la fin du paragraphe.

--extract-summary <nombre_intégral> ou -es <nombre_intégral>
   Extraire l’annotation des fichiers FB2/FB3 et l’insérer au début du texte.
   Valeurs possibles pour le paramètre numérique :
   0 – ne pas extraire l’annotation (la valeur est utilisée par défaut) ;
   1..5 – extraire l’annotation (le paramètre détermine l’ordre dans lequel l’auteur et le titre d’un livre seront répertoriés).

--skip-notes ou -sn
   Ignorer la liste des notes à la fin des fichiers DOCX/FB2/FB3/MD/ODT.

--include-notes <nombre_intégral> ou -in <nombre_intégral>
   Ajouter des notes des fichiers DOCX/FB2/FB3/MD/ODT dans le corps du texte.
   Valeurs possibles pour le paramètre numérique :
   0 – supprimer les liens vers les notes du texte ;
   1 – conserver les positions des notes dans le corps du texte (la valeur est utilisée par défaut) ;
   2 – déplacer les notes à la fin des phrases ;
   3 – déplacer les notes à la fin des paragraphes.

--insert-note-begin <text> ou -inb <text>
   Insérer des mots au début des notes de bas de page et des notes dans le corps du texte (par exemple : Note de l’éditeur.).
   Le paramètre est utilisé lors de l’extraction d’un texte à partir des fichiers DOCX/FB2/FB3/MD/ODT.

--insert-note-end <text> ou -ine <text>
   Insérer des mots à la fin des notes de bas de page et des notes dans le corps du texte (par exemple : Fin de note.).
   Le paramètre est utilisé lors de l’extraction d’un texte à partir des fichiers DOCX/FB2/FB3/MD/ODT.

--extract-tables <nombre_intégral> ou -et <nombre_intégral>
   Extraire des tableaux à partir des fichiers au format DOCX/FB2/FB3/ODT.
   Valeurs possibles pour le paramètre numérique :
   0 – supprimer les tableaux ;
   1 – extraire les données de chaque cellule de tableau comme une nouvelle ligne de texte (la valeur est utilisée par défaut) ;
   2 – conserver la mise en forme du tableau lors de l’extraction du texte.

--csv-comma
   Utiliser une virgule pour séparer les valeurs des colonnes lors de l’extraction des données à partir des fichiers XLS/XLSX/ODS (utilisée par défaut).

--csv-semicolon
   Utiliser un point-virgule pour séparer les valeurs des colonnes lors de l’extraction des données à partir des fichiers XLS/XLSX/ODS.

--csv-space
   Utiliser un espace pour séparer les valeurs des colonnes lors de l’extraction des données à partir des fichiers XLS/XLSX/ODS.

--csv-tab
   Utiliser la tabulation pour séparer les valeurs des colonnes lors de l’extraction des données à partir des fichiers XLS/XLSX/ODS.

--csv-double-quote
   Utiliser des guillemets doubles pour séparer les lignes lors de l’extraction des données à partir de fichiers XLS/XLSX/ODS (utilisés par défaut).

--csv-single-quote
   Utiliser des guillemets simples pour séparer les lignes lors de l’extraction des données à partir des fichiers XLS/XLSX/ODS.

--eml-save <nom_de_dossier>
   Extraire les fichiers joints au message, des documents au format EML et les enregistrer dans un dossier spécifié.

--eml-att
   Extraire la liste des noms de fichiers joints au message, des documents au format EML. Les noms de fichiers seront séparés par des virgules.

--eml-cc
   Extraire le champ "Cc" du titre du message dans un fichier au format EML ("carbon copy" : les noms et les adresses des destinataires secondaires du message à qui une copie sera envoyée).

--eml-date <format_de_date>
   Extraire le champ "Date" du titre du message dans un fichier au format EML (la date et l’heure d’envoi du message). Le format de date est spécifié avec des caractères spéciaux ("d", "m", "y", etc.). Par exemple : "dd.mm.yyyy hh:nn:ss".

--eml-from
   Extraire le champ "From" du titre du message dans un fichier au format EML (le nom et l’adresse de l’expéditeur).

--eml-org
   Extraire le champ "Organization" du titre du message dans un fichier au format EML (le nom de l’organisation qui a fourni l’accès au réseau pour envoyer la lettre).

--eml-rt
   Extraire le champ "Reply-To" du titre du message dans un fichier au format EML (le nom et l’adresse auxquels il faut envoyer les réponses à la lettre).

--eml-subj
   Extraire le champ "Subject" du titre du message dans un fichier au format EML (le sujet de la lettre).

--eml-to
   Extraire le champ "To" du titre du message dans un fichier au format EML (le nom et l’adresse du destinataire).


*** Exemples ***

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\"

blb2txt -f "d:\Docs\book.doc" -out "d:\Text\book.txt"

blb2txt -f "d:\Docs\*.doc" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 --replace-empty-lines

blb2txt -f "d:\Docs\*.*" -v "d:\Text\" -p "Document" -u

blb2txt -f "d:\Docs\1.doc" -v "d:\Text\" -p "Document" -a -n 20 -t 100000

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "Livre" -k "CHAPITRE" -k "TABLE DES MATIÈRES"

blb2txt -f "d:\Book\book.epub" -v "d:\Text\" -p "Livre" -r "###"

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\" -d "d:\rex\rules.rex" -d "d:\dic\rules.dic" --remove-spaces --remove-linebreaks

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Fichier de configuration ***

Les options de ligne de commande peuvent être enregistrées en tant que fichier de configuration « blb2txt.cfg » dans le même dossier que l'application console.

Exemple de fichier de configuration :
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

Le programme peut combiner les options du fichier de configuration et celles de la ligne de commande.


*** Opérations ***

L’ordre d'exécution des opérations :

1.  Extraire le texte de fichier(s).
2.  Formater le texte : supprimer les espaces superflues, sauts de ligne, etc. (si l'option est spécifiée).
3.  Combiner le texte en un seul fichier (si l'option est spécifiée).
4.  Fractionner le texte (si l'option est spécifiée).
5.  Appliquer les règles de correction de la prononciation (si l'option est spécifiée).
6.  Enregistrer le(s) fichier(s) de sortie sur disque.


*** Licence ***

Droits d'utilisation non commerciale de l’application :
- personnes physiques – sans restriction,
- personnes morales – avec les restrictions stipulées dans l'Accord de Licence du logiciel Balabolka.

L’utilisation commerciale du logiciel demande l'autorisation du détenteur du copyright.

###
