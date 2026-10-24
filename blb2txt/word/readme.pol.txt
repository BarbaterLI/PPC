Balabolka (oprogramowanie narzędziowe do wydobycia tekstu z plików), wersja 1.120
Copyright (c) 2013-2026 Ilya Morozov
Wszystkie prawa zastrzeżone

WWW: https://www.cross-plus-a.com/pl/btext.htm
E-mail: crossa@list.ru

Licenzja: Freeware
System operacyjny: Microsoft Windows 7/8/10/11


Program pozwala wydobywać tekst z plików różnego formatu.
Wydobyty tekst można połączyć w jeden plik i/lub podzielić na kilka plików.
Do tekstu można zastosować przepisy słowników korekty wymowy programu Balabolka.
Obsługiwane formaty plików: AZW, AZW3, CHM, DjVu (DjVu+OCR), DOC, DOCX, EPUB, FB2 (FB2.ZIP, FBZ), FB3, HTML, LIT, MD, MHT, MOBI, ODP, ODS, ODT, PDB, PDF, PPT, PPTX, PRC, RTF, TCR, TXT, TXTZ, WPD, WRI, XLS, XLSX.
Interfejs IFilter będzie używany dla plików o nieznanych rozszerzeniach.

Aplikacja może korzystać z biblioteki 7z.dll (32 bit). Pomaga wyodrębnić tekst i obrazy z dokumentów znajdujących się w archiwach (ZIP, RAR, itp.). 7z.dll jest częścią oprogramowania 7-Zip.
https://www.7-zip.org


*** Wiersz poleceń ***

blb2txt [parametry ...]


*** Parametry wiersza poleceń ***

-f <nazwa_pliku>
   Nazwa pliku lub maska dla nazw plików, z których należy wydobyć tekst. Wiersz poleceń może zawierać kilka parametrów [-f].

-fl <nazwa_pliku>
   Ustawia nazwę pliku tekstowego z listą plików wejściowych (jedna nazwa pliku na wiersz). Wiersz poleceń może zawierać kilka parametrów [-fl].

-v <nazwa_folderu>
   Nazwa folderu dla zachowania pliku z wydobytym tekstem.

-p <tekst>
   Szablon dla nazwy pliku z wydobytym tekstem (naprzykład, "Dokument tekstowy"). Jeśli parametr nie ustawiony, wykorzystuję się nazwa pliku źródłowego.
   Użyj zmiennej %FileName%, aby wstawić nazwę pliku wejściowego do nazwy pliku wyjściowego.
   Użyj zmiennej %FirstLine%, aby wstawić pierwszą linię tekstu.
   Użyj zmiennej %Header%, aby wstawić tytuł rozdziału.
   Użyj zmiennej %Number%, aby zmienić pozycję numeru sekwencji w nazwie pliku wyjściowego.
   Użyj zmiennej %Title%, aby wstawić tytuł dokumentu HTML (tylko dla plików HTML).
   Ostrzeżenie! Konieczne jest podwojenie znaku procentu (%) w skrypcie wsadowym. Na przykład: -p %%FirstLine%%

-ext <tekst>
   Ustawia rozszerzenie dla nazw plików wyjściowych. Wartość domyślna dorówna "txt".

-out <nazwa_pliku>
   Ustawia pełną nazwę pliku wyjściowego. Opcja zaleca się określenie tylko wtedy, gdy narzędzie jest używane jako część innego oprogramowania.
   Jeśli narzędzie jest używane do niestandardowego importu dokumentów, zewnętrzny program uruchamia narzędzie z wiersza poleceń i przekazuje pełną nazwę pliku tekstowego do utworzenia.

-s
   Wyszukiwanie plików wejściowych w podfolderach.

--create-folder lub -cf
   Utwórz podfolder wyjściowy dla każdego pliku wejściowego. Nazwa pliku zostanie użyta jako nazwa podfolderu.

-i
   Odczytuje dane z standarowego strumienia wejścia (STDIN). Jeśli parametr jest ustawiony, parametr [-f] jest ignorowany.

-o
   Zapisuje tekst do standarowego strumienia wyjścia (STDOUT). Jeśli parametr jest ustawiony, parametry [-v] i [-p] są ignorowane.

-u
   Łączy tekst z kilku plików w jeden plik.

-b
   Dodaje numer porządkowy przed nazwą pliku.

-a
   Dodaje numer porządkowy po nazwie pliku.

-n <liczba>
   Ustaw początkowy numer porządkowy pliku. Wartość domyślna dorówna 1.

-e <kodowanie>
   Ustawia kodowanie plików wyjściowych ("ansi", "utf8" lub "unicode"). Wartość domyślna dorówna "ansi".

-t <liczba>
   Ustawia sposób rozdzielenia tekstu: wykorzystanie wybranego romiaru pliku. Liczba odpowiada ilości znaków.

-k <słowo_kluczowe>
   Ustawia sposób rozdzielenia tekstu: wyszukiwanie słowa kluczowego w pliku źródłowym. Parametr jest zależny od rejestru. Wiersz poleceń może zawierać kilka parametrów [-k].

-r <słowo_kluczowe>
   Rozdziela tekst na słowie kluczowym i usunąć go z tekstu. Parametr jest zależny od rejestru. Wiersz poleceń może zawierać kilka parametrów [-r].

-w
   Rozdziela tekst po dwóch pustych liniach z rzędu.

-l
   Ustawia sposób rozdzielenia tekstu: wyszukiwanie wiersza w którym wszyskie litery są wielkie.

-c
   Dzieli tekst według spisu treści. Aplikacja wyodrębnia pozycje początków rozdziałów z pliku wejściowego (jeśli plik zawiera takie informacje).

-toc
   Generuje spis treści i dzieli tekst. Aplikacja dzieli wyodrębniony tekst według słów kluczowych (jak "rozdział" lub "wolumen").
   Jeśli opcja ta zostanie użyta razem z opcją [-c], aplikacja spróbuje wyodrębnić spis treści z dokumentu; jeśli to się nie powiedzie, zostanie wygenerowany nowy spis treści.

-m <liczba>
   Ustawia minimalny rozmiar części tekstu do podziału (jako liczbę znaków).

-j <liczba>
   Ignoruje początek rozdziału, jeśli rozmiar poprzedniego rozdziału jest mniejszy niż określona wartość (w znakach). Opcja ta jest używana razem z opcją [-c] lub [-toc].

-hh <tekst>
   Wstawia tekst przed nagłówkami (np: ## Chapter 1).

-d <nazwa_pliku>
   Używa słownika dla korekty wymowy (plik z rozszerzeniem *.BXD, *.DIC lub *.REX). Wiersz poleceń może zawierać kilka parametrów [-d].

-if
   Używa interfejsu IFilter do wyodrębniania tekstu. Jeśli to się nie powiedzie, aplikacja użyje metody domyślnej.

-g <nazwa_folderu>
   Ustawia nazwę folderu wyjściowego do zapisywania obrazów z dokumentu.

-cvr <nazwa_folderu>
   Ustawia nazwę folderu wyjściowego do zapisania obrazu okładki książki.

--clone-file-time lub -cft
   Klonowanie czasu utworzenia/modyfikacji/udostępnienia pliku wejściowego do pliku wyjściowego. Jeśli aplikacja łączy pliki tekstowe lub dzieli wyodrębniony tekst, opcja ta jest ignorowana.

-x <typ_pliku>
   Ustawia typ pliku wejściowego. Pozwala zdefiniować format dokumentów wejściowych z nieznanymi rozszerzeniami nazw plików. Na przykład: -x doc

-pwd <tekst>
   Ustawia hasło dla wydobycia tekstu z pliku PDF.

-dll <nazwa_pliku>
   Ustawia ścieżkę i nazwę dla 7z.dll (32 bit). Biblioteka ta pomaga wyodrębniać tekst i obrazy z dokumentów znajdujących się w archiwach (ZIP, RAR, itp.). 7z.dll jest częścią oprogramowania 7-Zip. Jeśli opcja nie zostanie określona, aplikacja i biblioteka muszą znajdować się w tym samym folderze; W przeciwnym razie aplikacja nie będzie w stanie wyodrębnić danych z plików archiwum.

-dex <typy_plików>
   Ustawia listę typów plików do wyodrębnienia z archiwów. Opcja zawiera oddzieloną przecinkami listę typów plików, na przykład: -dex "fb2,epub"
   Linia poleceń może zawierać kilka opcji [-dex]. Jeśli opcja nie zostanie określona, aplikacja wyodrębni tekst ze wszystkich plików w archiwum.
   Jeśli konieczne jest wyodrębnienie tekstu dla wszystkich typów plików obsługiwanych przez aplikację, należy użyć wartości "all-". Na przykład: -dex all-

-dne <typy_plików>
   Ustawia listę typów plików, które mają być ignorowane podczas wyodrębniania dokumentów z archiwów. Opcja zawiera oddzieloną przecinkami listę typów plików, np.: -dne "exe,dll"
   Wiersz poleceń może zawierać kilka opcji [-dne]. Jeśli opcja nie zostanie określona, aplikacja wyodrębni tekst ze wszystkich plików w archiwum.

-dp
   Wyświetl informacje o postępach w oknie konsoli.

-cfg <nazwa_pliku>
   Ustawia nazwę pliku konfiguracyjnego z opcjami wiersza poleceń (plik tekstowy, w którym każda linia zawiera jedną opcję). Jeśli opcja nie zostanie określona, plik "blb2txt.cfg" w tym samym folderze, w którym będzie używane narzędzie.

-h
   Drukuje listę dostępnych opcji wiersza poleceń.

--remove-spaces lub -rs
   Usuwa zbędne spacje (dwie lub wiecej kolejnych, twarde spacje).

--remove-hyphens lub -rh
   Usuwa kreski na końcu wiersza w tekście.

--remove-linebreaks lub -rl
   Usuwa znaki końca linii wewnątrz akapitów.

--remove-empty-lines lub -rm
   Usuwa wszystkie puste wiersze.

--replace-empty-lines lub -rp
   Zamienia kilka pustych wierszy o jeden pusty wiersz.

--remove-square-brackets lub -rsb
   Usuwa tekst wewnątrz [nawiasów kwadratowych].

--remove-curly-brackets lub -rcb
   Usuń tekst wewnątrz {nawiasów klamrowych}.

--remove-angle-brackets lub -rab
   Usuwa tekst wewnątrz <nawiasów ostrokątnych>.

--remove-round-brackets lub -rrb
   Usuwa tekst w (nawiasach okrągłych).

--remove-comments lub -rc
   Usuwa komentarze. Komentarze jednowierszowe zaczynają się od // i kontynuuj do końca wiersza. Komentarze wielowierszowe zaczynają się od /* i kończą na */.

--remove-page-numbers lub -rpn
   Usuń numery stron (może to być przydatne w przypadku plików DjVu/PDF).

--fix-ocr-errors lub -ocr
   Poprawa błędy, które mogą wystąpić podczas rozpoznawania tekstu (tylko alfabet cyryliczny).

--fix-letter-spacing lub -ls
   Poprawia odstępy między literami w słowach (na przykład: s p a c e, _w_o_r_d).

--add-period lub -ap
   Dodaje kropkę, jeśli po ostatnim słowie akapitu nie ma znaku interpunkcyjnego.

--extract-summary <liczba> lub -es <liczba>
   Wyodrębnia podsumowanie (zwane również "adnotacją") z plików FB2/FB3 i wstawia je na początku tekstu.
   Możliwe wartości dla parametru całkowitego:
   0 – pomija podsumowanie (wartość ta jest domyślnie używana);
   1..5 – wyodrębnia podsumowanie (wartość określa kolejność, w której wymieniono nazwę autora i tytuł książki).

--skip-notes lub -sn
   Pomija notatki, gdy aplikacja wyodrębnia tekst z plików DOCX/FB2/FB3/MD/ODT.

--include-notes <liczba> lub -in <liczba>
   Zawiera notatki w tekście, gdy aplikacja wyodrębnia tekst z plików DOCX/FB2/FB3/MD/ODT.
   Możliwe wartości dla parametru całkowitego:
   0 – usuwa linki do notatek z tekstu;
   1 – zachowuje domyślne pozycje notatek w tekście (wartość ta jest domyślnie używana);
   2 – umieszcza notatki na końcu zdań;
   3 – umieszcza notatki na końcu akapitów.

--insert-note-begin <tekst> lub -inb <tekst>
   Wstawia słowa na początku notatek, gdy notatki są zawarte w tekście (na przykład: Edytory notatki.).
   Opcja jest używana do plików DOCX/FB2/FB3/MD/ODT.

--insert-note-end <tekst> lub -ine <tekst>
   Wstawia słowa na końcu notatek, gdy notatki są zawarte w tekście (na przykład: Koniec notatki.).
   Opcja jest używana dla plików DOCX/FB2/FB3/MD/ODT.

--extract-tables <liczba> lub -et <liczba>
   Wyodrębnianie tabel z plików DOCX/FB2/FB3/ODT.
   Możliwe wartości dla parametru całkowitego:
   0 – pomija tabele;
   1 – wyodrębnia dane z każdej komórki jako nowy wiersz tekstu (ta wartość jest używana domyślnie);
   2 – zachowuje formatowania podczas wyodrębniania tabeli.

--csv-comma
   Kolumny są oddzielone przecinkiem, gdy aplikacja wyodrębnia dane z plików XLS/XLSX/ODS (domyślny separator dla plików CSV).

--csv-semicolon
   Kolumny są oddzielone średnikiem, gdy aplikacja wyodrębnia dane z plików XLS/XLSX/ODS.

--csv-space
   Kolumny są oddzielone pustym miejscem, gdy aplikacja wyodrębnia dane z plików XLS/XLSX/ODS.

--csv-tab
   Kolumny są oddzielone tabulatorem, gdy aplikacja wyodrębnia dane z plików XLS/XLSX/ODS.

--csv-double-quote
   Używa znaków podwójnego cudzysłowu, jeśli pole musi być cytowane (eksport z plików XLS/XLSX/ODS).

--csv-single-quote
   Używa pojedynczych cudzysłowów, jeśli pole musi być cytowane (eksport z plików XLS/XLSX/ODS).

--eml-save <nazwa_folderu>
   Wyodrębnia załączniki z plików EML i zapisuje je w określonym folderze.

--eml-att
   Wyodrębnia listę załączników z plików EML (nazwy plików dołączonych do wiadomości).

--eml-cc
   Wyodrębnia pole nagłówka "Cc" z plików EML ("kopia węglowa"; określa dodatkowych odbiorców wiadomości).

--eml-date <date_format>
   Wyodrębnia pole nagłówka "Data" z plików EML (lokalny czas i data utworzenia i wysłania wiadomości). Format daty jest definiowany przez specyfikatory (takie jak "d", "m", "y" itp.). Na przykład: "dd.mm.yyyy hh:nn:ss".

--eml-from
   Wyodrębnia pole nagłówka "Od" z plików EML (adres e-mail i opcjonalnie nazwisko autora).

--eml-org
   Wyodrębnia pole nagłówka "Organizacja" z plików EML (nazwa organizacji, przez którą nadawca wiadomości ma dostęp do sieci).

--eml-rt
   Wyodrębnia pole nagłówka "Odpowiedź-Do" z plików EML (adres, na który mają trafiać odpowiedzi).

--eml-subj
   Wyodrębnia pole nagłówka "Temat" z plików EML (temat wiadomości).

--eml-to
   Wyodrębnia pole nagłówka "Do" z plików EML (adres e-mail i opcjonalnie nazwa odbiorcy wiadomości).


*** Przykłady ***

Wyodrębnij tekst z pliku BOOK.DOC i zapisz jako BOOK.TXT w folderze wyjściowym:

blb2txt -f "d:\Docs\book.doc" -v "d:\Text\"


W razie potrzeby można również użyć tej wersji (gdy określono tylko jeden plik wejściowy):

blb2txt -f "d:\Docs\book.doc" -out "d:\Text\book.txt"


Wyodrębnij tekst z dokumentów Microsoft Word i RTF, usuń puste wiersze i zapisz pliki tekstowe w kodowaniu UTF-8:

blb2txt -f "d:\Docs\*.docx" -f "d:\Docs\*.rtf" -v "d:\Text\" -e utf8 -rm


Wyodrębnij tekst ze wszystkich plików tekstowych w określonym folderze, połącz je i zapisz jako "Document.txt":

blb2txt -f "d:\Docs\*.*" -v "d:\Text\" -p "Dokument" -u


Wyodrębnij tekst z pliku DOCUMENT.DOCX, podziel na części o rozmiarze 100 KB i zapisz jako pliki tekstowe "Document 20.txt", "Document 21.txt" itd.:

blb2txt -f "d:\Docs\Document.docx" -v "d:\Text\" -p "Dokument" -a -n 20 -t 100000


Wyodrębnij tekst z pliku BOOK.FB2, znajdź słowa "ROZDIAŁ" i "SPIS TREŚCI", aby podzielić tekst na części i zapisać jako pliki o nazwach "Książka 1.txt", "Książka 2.txt" itd.:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "Książka" -k "ROZDIAŁ" -k "SPIS TREŚCI"


Wyodrębnij tekst z pliku BOOK.EPUB, znajdź "###", aby podzielić tekst na części, usuń "###" z tekstu i zapisz każdą część jako nowy plik:

blb2txt -f "d:\Book\book.epub" -v "d:\Text\" -p "Książka" -r "###"


Wyodrębnij tekst z pliku BOOK.FB2, podziel go według spisu treści, zapisz pliki i użyj tytułów rozdziałów jako nazw plików. Nowe pliki tekstowe nie mogą być mniejsze niż jeden kilobajt:

blb2txt -f "d:\Book\book.fb2" -v "d:\Text\" -p "%Number% - %Header%" -c -j 1024


Pobierz tekst ze standardowego wejścia (STDIN), usuń zbędne spacje, znaki końca linii i puste linie, a następnie zapisz zaktualizowany tekst do standardowego wyjścia (STDOUT):

blb2txt -i -o --remove-spaces --remove-linebreaks --replace-empty-lines


Wyodrębnij tekst ze wszystkich dokumentów Microsoft Word znajdujących się w archiwach ZIP:

blb2txt -f "d:\Archive\*.zip" -v "d:\Text\" -dll "e:\7-Zip\7z.dll" -dex doc,docx


*** Plik konfiguracyjny ***

Parametry można zachować jak plik konfiguracyjny "blb2txt.cfg" w tym samym folderze co aplikacja konsolowa.

Przykład zawartości pliku:
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

Program może kombinować parametry z pliku konfiguracyjnego i wiersza poleceń.


*** Kolejność operacji ***

Program wykonuje operacje w następującej kolejności:

1.  Wydobyć tekst z pliku.
2.  Formatować tekst: usunąć zbędne spacje, znaki końca linii itp. (jeśli wybrane odpowiednie parametry).
3.  Połączyć tekst w jeden plik (jeśli wybrany odpowiedni parametr).
4.  Rozdzielić tekst na części (jeśli wybrane odpowiednie parametry).
5.  Stosować reguły korekty wymowy (jeśli wybrane odpowiednie parametry).
6.  Zachować plik lub pliki na płytę.


*** Licencja ***

Prawo do użytku niekomercyjnego: 
- dla osób fizycznych - bez ograniczeń;
- dla osób prawnych - z zastrzeżeniem ograniczeń, co określa "Umowa licencyjna" programu Balabolka.

Użytek komercyjny dozwolony jedynie za wcześniejszą zgodą posiadacza praw autorskich.

###
