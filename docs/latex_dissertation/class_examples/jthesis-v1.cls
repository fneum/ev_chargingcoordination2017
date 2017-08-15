% Class:   Jthesis.cls
% Purpose: Edinburgh University, Electrical Engineering PhD Thesis style
% Author:  George Taylor, gstaylor@iee.org
% Version: See \ProvidesClass below
% Copyright:
%   (C) 1998 George Taylor
%   This program is subject to the GNU General Public License, this explains
%   the conditions under which you may copy it and why there is no warranty.
%

% $Header: /u9/gst/PhD/ThesisStyleFile/StyleFiles/RCS/jthesis-v1.cls,v 1.13 1998/07/31 10:13:43 gst Exp $
% $Log: jthesis-v1.cls,v $
% Revision 1.13  1998/07/31 10:13:43  gst
% fixed page numbers on list of figures and tables
%
% Revision 1.12  1998/07/29 13:33:11  gst
% added /flushfloats
%
% Revision 1.11  1998/07/22 16:37:17  gst
% chagned nohyphens option
%
% Revision 1.10  1998/07/22 16:11:02  gst
% set partopsep to 0
%
% Revision 1.9  1998/07/22 11:56:04  gst
% changed file name and margins
%
% Revision 1.8  1998/07/21 10:44:53  gst
% fixed \crestfile to use renewcommand
%
% Revision 1.7  1998/07/20 15:05:30  gst
% fixed file name error
%
% Revision 1.6  1998/07/20 15:03:35  gst
% changed a few file names
%
% Revision 1.5  1998/07/20 14:45:41  gst
% fixed headings on contents, list of table/figures and references
%
% Revision 1.4  1998/07/20 14:07:04  gst
% modified during meeting with rudy and sarat
%
% Revision 1.3  1998/07/20 12:55:52  gst
% final for now
%
% Revision 1.2  1998/07/20 12:51:22  gst
% testing rcs
%
% Revision 1.1  1998/07/20 12:50:22  gst
% Initial revision
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Installer customisable things
% Location of university crest (EPS file)
\newcommand{\unicrestfile}{StyleFiles/EdUniCrest.pdf}
% command for the user to change the above
\newcommand{\crestfile}[1]{
  \renewcommand{\unicrestfile}{#1}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO NOT EDIT BELOW THIS LINE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class initialisation and document options
%% and stuff to be executed now

% Requires LaTex2e, not tested on versions older than 1996/12/01.
\NeedsTeXFormat{LaTeX2e}[1996/12/01]

% Provide the new class and output author information.
\ProvidesClass{StyleFiles/jthesis-v1}[1998/07/20 v1.00 PhD Thesis class]
\typeout{Author: George Taylor, G.S.Taylor@ee.ed.ac.uk}

% Document options:
%   nohyphens           (University guidelines say
%                       ``hyphenation should be avoided if possible'')
%   doublespace         Use 1.5 spacing, \begin{singlespace} for exceptions
%   twoside		Produce double sided output

\DeclareOption{nohyphens}{
  \AtEndOfClass{
    % make hyphenation very unlikely
    \hyphenpenalty=10000
    \exhyphenpenalty=10000
    % now be a bit more relaxed about formatting or lots of
    % lines will stick into the right hand margin
    \tolerance=400
    \setlength{\emergencystretch}{5em}
  }
}

\DeclareOption{doublespace}{
  \AtEndOfClass{
    \setstretch{1.4}
  }
}


\DeclareOption{twoside}{
  % if two-side cancel out the 9mm later on (see 'modify margins')
  \AtEndOfClass{
    \addtolength{\hoffset}{9mm}
  }
  \PassOptionsToClass{twoside}{report}
}
  

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

% end of document options
\ProcessOptions


% Base this class on the report class
\LoadClass[a4paper,11pt,final,titlepage,onecolumn]{report}

% Modify margins.
% These were obtained by printing out some test pages, and measuring
% the error relative to the university thesis guidelines.

\addtolength{\topmargin}{-12mm}
\addtolength{\textheight}{15mm}
\addtolength{\oddsidemargin}{7mm}
\addtolength{\evensidemargin}{-25mm}
\addtolength{\textwidth}{19mm}
\addtolength{\headheight}{2pt}
\addtolength{\hoffset}{-9mm}

% Times package gives nicer font and more text per page
\RequirePackage{times}

% Inclusion of postscript and other graphics
% Note this forces the user to use graphicx and not graphics
\RequirePackage{graphicx}

% Headings and footers
\RequirePackage{fancyhdr}

% Nicer looking citations - enforce this for consistent style
\RequirePackage{citesort}

% Double spacing - always use this even when singlespaced
% so that \begin{singlespace} does not break
\RequirePackage{doublespace}
% single spacing by default unless doublespace option used
\setstretch{1}

% Better positioning and control of floats (adds H option)
\RequirePackage{float}
\restylefloat{figure}
\restylefloat{table}

% For user controllable boolean variables without low-level TeX
\RequirePackage{ifthen}

% Needed by \flushfloats
\RequirePackage{afterpage}


% E.Maguire - Commented this bit out due to conflict with subfig package
% Hanging captions, and with bold/italic fonts
% Caption package MUST come after float, rotating and subfigure so
% this code is executed with an \AtBeginDocument so the user
% can load up these other packages with \RequirePackage first
%\AtBeginDocument{
%  \RequirePackage[hang]{caption}
%  \renewcommand{\captionfont}{\itshape}
%  \renewcommand{\captionlabelfont}{\upshape \bfseries}
%}


% roman numbering until \startchapters issued
\pagenumbering{roman}
% user defined headings
\pagestyle{fancy}

% numbering of subsubsections, note these don't appear in the contents
\addtocounter{secnumdepth}{1}

% paragraph formatting
% no indentation for paragraphs, and set gap between paragraphs
\setlength{\parindent}{0mm}
\setlength{\parskip}{2ex}
% because we've set \parskip to a >0 value we set \partopsep to 0
% so that it doesn't matter if you place a blank line before a list
\setlength{\partopsep}{0mm}

% redefine spacing after section/subsection etc
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {0.6ex}%
                                   {\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.1ex}%
                                     {\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.1ex}%
                                     {\normalfont\normalsize\bfseries}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Everything below this point just defines commands/environments
%% and contains no code which is "executed now"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maketitle
% Command to make a title page from information provided with
% \title{} \author{} and \date{}
\renewcommand{\maketitle}{
  \begin{singlespace}
    \parskip 0mm
    % no page numbers on front page
    \thispagestyle{empty}
    % title, we use a parbox so the stuff afterwards and the hrules stay
    % in the space place irrespective of the title length (max 4 lines)
    % no paragraph indentation in title
    \hrule \vspace{3mm}
    \parbox[c][4cm][c]{\linewidth}{
      \begin{center}\huge \bfseries \@title \\ \end{center}
    }
    \vspace{3mm} \hrule
    % author
    \vspace{1.5cm}
    \begin{center} {\Large \it \@author} \end{center}
    % university crest
    \vspace{2.5cm}
    \begin{center}
      \includegraphics[width=2in]{\unicrestfile}
    \end{center}
    \vspace{3.5cm}
    % degree, university, date
    \begin{center}
       A thesis submitted for the degree of Doctor of Philosophy.\\
        {\bf The University of Edinburgh}.\\
        \@date \\
     \end{center}
  \end{singlespace}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% command to ensure on a right hand page but use blank left page
% rather than one with header
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{plain}\cleardoublepage}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% indicate main chapters start
\newcommand{\startchapters}[0]{
  \newpage
  \clearemptydoublepage
  \pagenumbering{arabic}
  \setcounter{page}{1}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% chapter style and behaviour redefinitions

% redefine how chapters and sections affect headings
\renewcommand{\chaptermark}[1]{\markboth{}{#1}}
\renewcommand{\sectionmark}[1]{}

% chapters start on right hand side if boolean chapteronrhs is set
\newboolean{chapteronrhs}
\setboolean{chapteronrhs}{true}

\renewcommand\chapter{\ifthenelse{\boolean{chapteronrhs}}
                        {\clearemptydoublepage}{\clearpage}
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}

% for use with abbreviations etc
\newcommand{\minorchapter}[1]{
\setboolean{chapteronrhs}{false}
\chapter*{#1}
\markboth{}{#1}
\addcontentsline{toc}{subsection}{#1}
\setboolean{chapteronrhs}{true}
}

% chapter heading for \chapter{}
\def\@makechapterhead#1{
  \begin{singlespace}
    \parskip 0mm
    \parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \hrule \vspace{1.5ex}
      \vspace{-\parskip}
      \begin{flushright}
        \Huge \rm \@chapapp \space \thechapter
        \par\nobreak
      \end{flushright}
    \fi
    \vspace{-3ex}
    \begin{flushright}
      \rm \Huge \bfseries #1 \par\nobreak
    \end{flushright}
    \vspace{2ex} \hrule
    \vskip 20\p@
  \end{singlespace}
}
% chapter heading for \chapter*{} (e.g. table of contents)
\def\@makeschapterhead#1{
  \begin{singlespace}
    \parskip 0mm
    \parindent \z@ \raggedright
    \normalfont
    \hrule \vspace{1ex}
    \vspace{-\parskip}
    \begin{center}
      \huge #1\par\nobreak
    \end{center}
    \vspace{1ex} \hrule
    \vskip 20\p@
  \end{singlespace}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% abstract
\renewcommand{\abstract}[1]{
  \setboolean{chapteronrhs}{false}
  \chapter*{\abstractname}
  \setboolean{chapteronrhs}{true}
  \thispagestyle{empty}
  #1
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% contents, lists of figures and tables

% redefine title of list of figures and table to have only
% initial uppercase letter
\renewcommand{\listfigurename}{List of figures}
\renewcommand{\listtablename}{List of tables}

% redefine \tableofcontents to change header marking
\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \markboth{}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }

% redefine \listoffigures to change header marking
\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \markboth{}{\listfigurename}}%
    \addcontentsline{toc}{subsection}{List of figures}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }


% redefine \listoftables to change header marking
\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \markboth{}{\listtablename}}%
    \addcontentsline{toc}{subsection}{List of tables}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }


\newcommand{\contentsandlists}[0]{
  \setboolean{chapteronrhs}{false}
  \begin{singlespace}
    \parskip 0mm
    \addcontentsline{toc}{subsection}{Contents}
    \tableofcontents
    \listoffigures
    \listoftables
  \end{singlespace}
  \setboolean{chapteronrhs}{true}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% quotations are to be single spaced
\renewenvironment{quotation}
               {\begin{singlespace}
                        \list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist \end{singlespace}}
\renewenvironment{quote}
               {\begin{singlespace}
                \list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist \end{singlespace}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bibliography, or more accurately references

% redefine thebibliography environment to change heading
\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \markboth{}{\bibname}}%
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

% command to insert the whole lot in one go
\newcommand{\insertreferences}[2][Bibliography]{
  \begin{singlespace}
    \parskip 0mm
    \renewcommand{\bibname}{#1}
    \bibliographystyle{authordate1}
    \addcontentsline{toc}{chapter}{\bibname}
    \markboth{}{\bibname}
    \bibliography{#2}
  \end{singlespace}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% macro to insert an eps file in a figure in one line
% e.g. \image[width=2in,height=1in]{filename}{label}{caption}
\newcommand{\image}[4][]{
  \begin{figure}[htb] \begin{center}
  \includegraphics[#1]{#2}
  \caption{#4}
  \label{fig:#3}
  \end{center}
  \end{figure}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% command to flush out floats which are pushing other (smaller) ones
% towards the end of a chapter
\newcommand{\flushfloats}{\afterpage{\clearpage}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

