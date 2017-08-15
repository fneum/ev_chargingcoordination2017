% This is the dmathesis class that gives a format of the thesis
% according to the university of Durham regulation.  To use it, say
% \documentclass{dmathesis} 

% You need to supply file format.tex, where you define the packages
% you want to use etc.

% Good luck and save your time!

% Written by M. Imran and last modified on 2001/06/18.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dmathesis}[2001/06/18 Durham Maths thesis class]

%  Here the fontsize you can use.

\newcommand\@dmathesisptsize{12pt}

\DeclareOption{12pt}{\renewcommand\@dmathesisptsize{12pt}}
\DeclareOption{11pt}{\renewcommand\@dmathesisptsize{11pt}}
\DeclareOption{10pt}{\renewcommand\@dmathesisptsize{10pt}}


\ProcessOptions

\LoadClass[\@dmathesisptsize,a4paper]{report}

\pagestyle{plain}

%  Here the page setup based on the university of Durham regulation 
\setlength{\voffset}{-1in}  
\setlength{\marginparsep}{0in}  
\setlength{\marginparwidth}{0in}  
\setlength{\marginparpush}{0in}  
\setlength{\hoffset}{-1in}  
\setlength{\oddsidemargin}{4cm}  
\setlength{\evensidemargin}{4cm} 
\setlength{\textwidth}{14.5cm}
\setlength{\topmargin}{19mm}
\setlength{\headheight}{5mm}
\setlength{\headsep}{4mm}
\setlength{\textheight}{230mm}
\setlength{\footskip}{15mm}

% Allow internal use of @ symbol
\makeatletter

%  Redefinition  \tableofcontents  in  report.cls
\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{\contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }

%  Redefinition \listoffigures in report.cls
\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\listfigurename}{\listfigurename}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }

%  Redefinition \listoftables in report.cls
\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{\listtablename}{\listtablename}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }

% Definition of a combined figure and table list
\def\listfiguretablename{Figures and Tables}

\newcommand\listoffiguresandtables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \addcontentsline{toc}{chapter}{\listfiguretablename}%
    \chapter*{\listfiguretablename
       \@mkboth{\MakeUppercase\listfiguretablename}{\MakeUppercase\listfiguretablename}}%
    {\bfseries\sffamily\Large Figures\par}%
     \vspace*{10pt}%
    {\parskip2.5\p@\@starttoc{lof}}%
     \mrule%
    {\bfseries\sffamily\Large Tables\par}%
     \vspace*{10pt}%
    {\@starttoc{lot}}
    \if@restonecol\twocolumn\fi
    }


%  Redefinition thebibliograph in report.cls
\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\bibname}{\bibname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}


%% Because of the font change, the page number becomes too large for the
%% horizontal space LaTeX reserves for it by default. Without the following
%% redefines to fix it, this would cause the Chapter entry page numbers
%% to extend a few points into the right margin. The horror!

\renewcommand{\@pnumwidth}{1.75em}
\renewcommand{\@tocrmarg}{2.75em}

% Make at safe again.
\makeatother

%\endinput















