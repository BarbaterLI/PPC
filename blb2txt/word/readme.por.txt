Balabolka (programa para extrair texto do arquivo), versão 1.120
Copyright (c) 2013-2026 Ilya Morozov
All Rights Reserved

WWW: https://www.cross-plus-a.com/br/btext.htm
E-mail: crossa@list.ru

Licença: Freeware
Sistema operacional: Microsoft Windows 7/8/10/11


O programa é projetado para extrair textos de arquivos de diferentes formatos.
O texto extraído pode ser montado em um único arquivo e / ou distribuídos em vários arquivos.
Ao texto podem ser aplicadas as regras dos dicionários de correção da pronúncia do programa Balabolka.
São suportados os seguintes formatos de arquivo: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
A interface IFilter será usada para arquivos com extensões desconhecidas.

O aplicativo pode usar a biblioteca 7z.dll (32 bits). Ela ajuda a extrair texto e imagens de documentos dentro de arquivos compactados (ZIP, RAR, etc.). 7z.dll faz parte do software 7-Zip.
https://www.7-zip.org


*** Linha de comando ***

blb2txt [opções ...]


*** Opções de comando ***

-f <nome_do_arquivo>
   Nome do arquivo ou uma máscara para os nomes dos arquivos que é extraído a partir do texto. A linha de comando pode conter várias opções [-f].

-fl <nome_do_arquivo>
   Definir o nome do arquivo de texto com a lista de arquivos de entrada (um nome de arquivo por linha). A linha de comando pode conter várias opções [-fl].

-v <nome_da_pasta>
   Nome da pasta para salvar o arquivo com o texto extraído.

-p <texto>
   Modelo para o arquivo com o texto extraído (por exemplo, "Documento de texto"). Se não for especificado o nome será usado o nome do arquivo de origem.
   Use a variável %FileName% para inserir o nome do arquivo de entrada no nome do arquivo de saída.
   Use a variável %FirstLine% para inserir a primeira linha do texto.
   Use a variável %Header% para inserir o título do capítulo.
   Use a variável %Number% para alterar a posição do número de sequência dentro do nome do arquivo de saída.
   Use a variável %Title% para inserir o título do documento HTML (apenas para arquivos HTML).
   Atenção! É necessário duplicar o sinal de porcentagem (%) em um arquivo de script. Por exemplo: -p %%FirstLine%%

-ext <texto>
   Definir a extensão para nomes de arquivos de saída. O padrão é "txt".

-out <nome_de_arquivo>
   Definir o nome completo do arquivo de saída. Recomenda-se especificar essa opção apenas quando o utilitário for usado como parte de outro software.
   Se o utilitário for usado para importação customizada de documentos, o programa externo executa o utilitário a partir de uma linha de comando e passa o nome completo do arquivo de texto a ser criado.

-s
   Pesquisar arquivos de entrada em subpastas.

--create-folder ou -cf
   Criar uma subpasta de saída para cada arquivo de entrada. O nome do arquivo será usado como nome da subpasta de saída.

-i
   Obtém a entrada de texto do STDIN. Se especificado, a opção [-f] é ignorada.

-o
   Inscrever o texto no fluxo da saída padrão (STDOUT). Se especificado, as opções [-v] e [-p] são ignoradas.

-u
   Combinar textos de múltiplos arquivos num arquivo único.

-b
   Adicionar o número de série antes do nome do arquivo.

-a
   Adicionar o número de série após o nome do arquivo.

-n <número>
   Estabelecer o número de série inicial do arquivo. O padrão é 1.

-e <codificação>
   Codificação do arquivo para extrair o texto ("ansi", "utf8" ou "unicode"). O valor padrão é "ansi".

-t <número>
   Especificar o modo de partição do texto: utilização de um tamanho prescrito do arquivo (as a number of characters).

-k <palavra_chave>
   Especificar o modo de partição do texto: busca de palavras-chave do texto no arquivo de origem. Este parâmetro é sensível a maiúsculas. A linha de comando pode conter várias opções [-k].

-r <palavra_chave>
   Dividir o texto com a palavra-chave e remové-la do texto. Esta opção é sensível a maiúsculas. A linha de comando pode conter várias opções [-r].

-w
   Especificar o modo de partição do texto: encontrar duas linhas em branco uma após outra.

-l
   Especificar o modo de partição do texto: buscar o texto onde todas as letras são maiúsculas.

-c
   Dividir o texto por um índice. O aplicativo extrai as posições do início dos capítulos do arquivo de entrada (se o documento contiver essas informações).

-toc
   Gera um índice e divide o texto. O aplicativo divide o texto extraído por palavras-chave (como "capítulo" ou "volume").
   Se a opção for usada junto com a opção [-c], o aplicativo tentará extrair um índice do documento; se não conseguir, um novo índice será gerado.

-m <número>
   Definir o tamanho mínimo das partes do texto para divisão (como um número de caracteres).

-j <número>
   Ignora o início do capítulo se o tamanho do capítulo anterior for menor que o valor especificado (em caracteres). Esta opção é utilizada em conjunto com [-c] ou [-toc].

-hh <texto>
   Inserir texto antes dos títulos (por exemplo: ## Capítulo 1).

-d <nome_do_arquivo>
   Usar um dicionário para a pronúncia correcta (arquivo com extensão *.BXD, *.DIC ou *.REX). A linha de comando pode conter várias opções [-d].

-if
   Usar a interface IFilter para extrair texto de arquivos. Se isso falhar, o método padrão será usado pelo aplicativo.

-g <nome_da_pasta>
   Definir o nome da pasta de saída para salvar imagens de documentos.

-cvr <nome_da_pasta>
   Definir o nome da pasta de saída para salvar a imagem da capa do livro.

--clone-file-time ou -cft
   Clona a hora de Criação/Modificação/Acesso do arquivo de entrada no arquivo de saída. Se o aplicativo combinar arquivos de texto ou dividir o texto extraído, a opção será ignorada.

-x <tipo_de_arquivo>
   Definir o tipo de arquivo de entrada. Permite definir um formato de documentos de entrada com extensões de nome de arquivo desconhecidas. Por exemplo: -x doc

-pwd <texto>
   Definir uma senha para extrair o texto do ficheiro no formato de PDF.

-dll <nome_do_arquivo>
   Definir o caminho e o nome para 7z.dll (32 bits). Esta biblioteca ajuda a extrair texto e imagens de documentos dentro de arquivos (ZIP, RAR, etc.). 7z.dll faz parte do software 7-Zip.
   Se a opção não for especificada, o aplicativo e a biblioteca devem estar na mesma pasta; caso contrário, o aplicativo não poderá extrair dados dos arquivos compactados.

-dex <tipos_de_arquivos>
   Definir a lista de tipos de arquivos para extração de arquivos compactados. A opção contém uma lista separada por vírgulas de tipos de arquivos, por exemplo: -dex "fb2,epub"
   A linha de comando pode conter poucas opções [-dex]. Se a opção não for especificada, o aplicativo extrairá o texto de todos os arquivos em um arquivo compactado.
   Se for necessário extrair o texto de todos os tipos de arquivo suportados pelo aplicativo, use o valor "all-". Por exemplo: -dex all-

-dne <tipos_de_arquivos>
   Definir a lista de tipos de arquivos a serem ignorados quando os documentos forem extraídos dos arquivos.
   A opção contém uma lista separada por vírgulas de tipos de arquivos, por exemplo: -dne "exe,dll"
   A linha de comando pode conter algumas opções -dne. Se a opção não for especificada, o aplicativo extrairá o texto de todos os arquivos em um arquivo.

-dp
   Exibir informações de progresso em uma janela do console.

-cfg <nome_do_arquivo>
   Definir o nome do arquivo de configuração com as opções da linha de comando (um arquivo de texto em que cada linha contém uma opção). Se a opção não for especificada, será utilizado o arquivo "blb2txt.cfg" na mesma pasta do utilitário.

-h
   Mostrar a descrição da linha de comando.

--remove-spaces ou -rs
   Eliminar espaços em branco (dois ou mais espaços um após outro, espaços inseparáveis).

--remove-hyphens ou -rh
   Eliminar traços nas extremidades das linhas do texto.

--remove-linebreaks ou -rl
   Eliminar quebras de linha dentro de um parágrafo.

--remove-empty-lines ou -rm
   Eliminar todas as linhas vazias.

--replace-empty-lines ou -rp
   Substituir múltiplas linhas brancas por uma linha em branco.

--remove-square-brackets ou -rsb
   Eliminar o texto em [colchetes].

--remove-curly-brackets ou -rcb
   Eliminar o texto em {chavetas}.

--remove-angle-brackets ou -rab
   Eliminar o texto entre <sinais de menor e maior>.

--remove-round-brackets ou -rrb
   Eliminar o texto entre (parênteses).

--remove-comments ou -rc
   Eliminar comentários. Comentários de linha única começam com // e continuam até o final da linha. Comentários de várias linhas começam com /* e terminam com */.

--remove-page-numbers ou -rpn
   Eliminar os números das páginas (pode ser útil para arquivos DjVu/PDF).

--fix-ocr-errors ou -ocr
   Corrigir erros encontrados em OCR (só para idiomas com alfabeto cirílico).

--fix-letter-spacing ou -ls
   Correção do espaçamento entre letras nas palavras (por exemplo: e s p a ç o, p_a_l_a_v_r_a).

--add-period ou -ap
   Adicionar um ponto final se não houver pontuação após a última palavra do parágrafo.

--extract-summary <número> ou -es <número>
   Extraia um resumo (também chamado de "anotação") dos arquivos FB2/FB3 e insira-o no início do texto.
   Valores possíveis para o parâmetro inteiro:
   0 – ignorar um resumo (este valor é usado por padrão);
   1..5 – extrai um resumo (um valor determina a ordem em que o nome do autor e o título do livro são listados).

--skip-notes ou -sn
   Ignorar notas quando o aplicativo extrai texto de arquivos DOCX/FB2/FB3/MD/ODT.

--include-notes <número> ou -in <número>
   Incluir notas no texto, quando o aplicativo extrai texto de arquivos DOCX/FB2/FB3/MD/ODT.
   Valores possíveis para o parâmetro inteiro:
   0 – eliminar links para notas do texto;
   1 – mantém as posições padrão das notas dentro do texto (este valor é usado por padrão);
   2 – colocar notas no final das frases;
   3 – colocar notas no final dos parágrafos.

--insert-note-begin <texto> ou -inb <texto>
   Inserir palavras no início das notas, quando as notas estão incluídas no texto (por exemplo: Nota do editor.).
   A opção é usada para arquivos DOCX/FB2/FB3/MD/ODT.

--insert-note-end <texto> ou -ine <texto>
   Inserir palavras no final das notas, quando as notas estiverem incluídas no texto (por exemplo: Fim da nota.).
   A opção é usada para arquivos DOCX/FB2/FB3/MD/ODT.

--extract-tables <número> ou -et <número>
   Extrair tabelas de arquivos DOCX/FB2/FB3/ODT.
   Valores possíveis para o parâmetro inteiro:
   0 – ignorar tabelas no texto;
   1 – extrair dados de cada célula como uma nova linha de texto (este valor é usado por padrão);
   2 – manter a formatação ao extrair uma tabela.

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

--eml-save <nome_da_pasta>
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


*** Exemplos de comandos ***

Extraia o texto do BOOK.DOC e salve como BOOK.TXT na pasta de saída:

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\"


Esta variante também pode ser usada, se necessário (quando apenas um arquivo de entrada é especificado):

blb2txt -f "d:\Docs\book.doc" -out "d:\Text\book.txt"


Extraia texto de documentos do Microsoft Word e RTF, remova linhas em branco e salve arquivos de texto na codificação UTF-8:

blb2txt -f "d:\Docs\*.docx" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Extraia o texto de todos os arquivos de texto na pasta especificada, junte-os e salve como DOCUMENTO.TXT:

blb2txt -f "d:\Docs\*.txt" -v "d:\Text\" -p "Documento" -u


Extraia o texto do DOCUMENT.DOCX, divida em partes com tamanho de 100 KB e salve como arquivos de texto "Documento 20.txt", "Documento 21.txt", etc.:

blb2txt -f "d:\Docs\document.docx" -v "d:\Text\" -p "Documento" -a -n 20 -t 100000


Extraia o texto do BOOK.FB2, encontre as palavras "CAPÍTULO" e "Índice" para dividir o texto em partes e salve como arquivos com os nomes "Livro 1.txt", "Livro 2.txt", etc.:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "Livro" -k "CAPÍTULO" -k "Índice"


Extraia o texto do BOOK.EPUB, encontre "###" para dividir o texto em partes, remova "###" do texto e salve cada parte como um novo arquivo:

blb2txt -f "d:\Book\book.epub" -v "d:\Text\" -p "Livro" -r "###"


Extraia o texto do BOOK.FB2, divida-o pelo índice, salve os arquivos e use os títulos dos capítulos como nomes de arquivo. Os novos arquivos de texto não devem ter menos de um kilobyte:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024


Obter texto do STDIN, remover espaços em excesso, quebras de linha e linhas vazias, gravar o texto atualizado no STDOUT:

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Extraia texto de todos os documentos do Microsoft Word dentro de arquivos ZIP:

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Arquivo de configuração ***

É possível salvar o arquivo de configuração "blb2txt.cfg" na mesma pasta que o aplicativo de console.

Um exemplo do conteúdo do arquivo:
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

O programa pode combinar opções do arquivo de configuração e da linha de comando.


*** Operações ***

O programa cumpre operações na seguinte ordem:

1.  Extrair texto do arquivo.
2.  Formatar o texto: remove os espaços extras, quebras de linha, etc. (se tais opções são especificadas).
3.  Reunir textos num único arquivo (se tal opção é especificada).
4.  Dividir o texto em partes (se tais opções são especificadas).
5.  Aplicar as regras de pronúncia correcta (se tais opções são especificadas).
6.  Salvar o arquivo ou os arquivos no disco.


*** Licença ***

Você está livre para usar e distribuir o software para fins não comerciais. Para uso ou distribuição comercial, você precisa obter permissão do detentor dos direitos autorais.

###
