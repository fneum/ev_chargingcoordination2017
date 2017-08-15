%%
%% edengths.cls - LaTeX2e thesis class
%%
%% Copyright (C) 1998 George Taylor
%% Copyright (C) 2002 Mike Nolta <mrnolta@princeton.edu>
%% Copyright (C) 2013 Mathew Topper <damm_horse@yahoo.co.uk>
%%
%% This file is part of the University of Edinburgh, Department of
%% Engineering LaTeX2e thesis template.
%% 
%% The University of Edinburgh, Department of Engineering LaTeX2e thesis
%% template is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% (at your option) any later version.
%% 
%% The University of Edinburgh, Department of Engineering LaTeX2e thesis
%% template is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with the University of Edinburgh, Department of Engineering
%% LaTeX2e thesis template.  If not, see <http://www.gnu.org/licenses/>.
%%
%%
%%   ABOUT
%%
%% This is the class file for a Latex2e template which corresponds to the
%% regulations regarding layout of a thesis submitted within the University
%% of Edinburgh School of Engineering. It is not `official', but conforms
%% as best as possible to the regulation as detailed at:
%%
%% http://www.ed.ac.uk/schools-departments/science-engineering/current-students/
%% research/submitting-thesis
%%
%% Please feel free to alter the template to your own liking, but note that
%% the template is made available under the GNU GPL and must be similarly
%% licenced should you wish to release your modified template.
%%
%%
%%   CREDITS
%%
%% This template is an amalgamtion of an existing Edinburgh University,
%% Electrical Engineering PhD Thesis class file (jthesis-v1.cls) authored by
%% George S Taylor which was released under the GNU GPL.
%% Code is included from the dmathesis class Written by M. Imran
%% for a thesis according to the university of Durham regulation, which was
%% released without copyright. It also contains ideas (possibly code) from the
%% Princeton thesis class file (PrincetonThesis.cls), authored by Mike Nolta.
%% Mathew Topper, Eoghan Maguire and Bill Edwards forsaw the need to maintain a
%% more recent latex implementation of the thesis regulations and thus, this
%% project was born. It is hoped that the template will be maintained by the
%% Edinburgh Engineering PhD community once released.
%%
%%
%%   RECORD OF REVISIONS
%%
%%     Date      Programmer         Description of change
%%     ====      ==========         =====================
%%   14/06/10    Mathew Topper      Wrote preamble.
%%                                  Declared use of report class options.
%%
%%   20/06/10                       Set an option to use fancy chapter
%%                                  headings like jthesis-v1 except using
%%                                  titlesec to design the headers
%%
%%                                  Modified the titlepage and added an
%%                                  option to include a crest.
%%
%%   21/06/10                       Had to fix interaction between reset
%%                                  of parskip and the non-starred title.
%%
%%   22/06/10                       Removed pdf option in favour of
%%                                  ifpdf package.
%%
%%                                  Added 'ae' package to improve ouput
%%                                  quality.
%%
%%                                  Fixed lenghts of TOC, LOF, LOT and
%%                                  section whitespace after the nopardent
%%                                  option is set. Had to counter the effect
%%                                  of the increased parskip.
%%
%%                                  'times' package obsolete. Replaced with
%%                                  'mathptmx'
%%
%%   23/06/10                       Removed startappendix command as to 
%%                                  get correct behaviour using include for
%%                                  appendix files, the commands must also
%%                                  be included in a file. The new file is
%%                                  called appedix/edengapp.tex
%%
%%   24/06/10                       Took cleardoublepage out of the 
%%                                  declaration and put into the makefrontmatter
%%                                  command.
%%
%%                                  Added boolean twosideflag to check if
%%                                  twoside is set. If it is set then use
%%                                  pagestyle 'twomain' which can
%%                                  set different headers (for even
%%                                  pages) at the end of makefrontmatter
%%                                  wrapper instead of main.
%%
%%   25/06/10                       Fixed parskip issue by switching to
%%                                  rapport3 class when nopardent is set.
%%                                  Had to reauthor the maketitle command
%%                                  to do this as the original definition
%%                                  was not consistent between the two
%%                                  classes. Also removed the pardentflag
%%                                  logical and added a \classfile command
%%                                  to record which class to use.
%%                                  Had to protect the sections from rapport3
%%                                  as they got messed up.
%%
%%                                  Improved title and section for possible
%%                                  12pt doublespace use.
%%
%%   03/07/10                       Changed listoffiguresandtables to have
%%                                  the addcontentsline command after the
%%                                  chapter. Also changed the headings to
%%                                  section* commands for Figures and Tables.
%%
%%                                  Removed tocbibind package in favour of
%%                                  custom toc entries using addtocontentsline
%%                                  This allows correct anchors when using
%%                                  hyperref.
%%
%%   13/10/10                       Loading the packages.tex file here now
%%                                  so that showlabels can be executed last
%%                                  (after hyperref for instance).
%%
%%   17/10/10                       Added bibhyphenpenalty which is used to 
%%                                  set hyphenpenalty just before the
%%                                  bibliography is called.
%%
%%   19/10/10                       Removed redefinition of bibliography in
%%                                  place of edbibliography. This command
%%                                  reacts to the nobib option which then
%%                                  uses nobibliograpy command from the 
%%                                  bibentry package. This package is now
%%                                  loaded by default.
%%
%%   20/10/10                       Bugfix. Added pagestyle command to
%%                                  /startbody which is needed if the
%%                                  front matter is not compiled.
%%
%%   29/10/10                       Adding hyperref option. It loads the
%%                                  pdftex option if pdflatex is used and
%%                                  dvipdfm otherwise. Thus the dvi route
%%                                  must go dvi->pdf. No postscript step.
%%                                  hypcap package is also loaded for better
%%                                  linking to figures.
%%
%%   04/04/11                       Set a qualification command to set
%%                                  a custom qualification on the front
%%                                  page.
%%
%%                                  Added a frontchapter command for adding
%%                                  chapters in the front matter that do
%%                                  not have numbers but do have TOC
%%                                  entrys. Starred version has no TOC
%%                                  entry.
%%
%%                                  Added nomencl option for running the
%%                                  nomencl package with the template.
%%                                  It redefines the way that the title is
%%                                  produced.
%%
%%   13/10/11                       Removed 11pt option from class file load
%%                                  as it overided the 10pt option if it was
%%                                  used.
%%
%%                                  Moved the loading of packages.tex to before
%%                                  the loading of the package classes so that
%%                                  the \PassOptionsToPackage command can be
%%                                  used to influence the default packages.
%%
%%   28/02/12                       Took the font options out of the titlesec
%%                                  definitions as the chapter and section 
%%                                  definitions themselves override these
%%                                  options making them redundant. I also did
%%                                  a little reorgansation to make the
%%                                  formatting definitions a little clearer.
%%                                  \defaultfont is the main control of the sans
%%                                  font option.
%%
%%   13/06/12                       Added the calc package to allow the use of +
%%                                  when calculating the title parbox height.
%%                                  Calc was missing but was called if the
%%                                  mhchem package was loaded, so it hid the
%%                                  bug.
%%
%%   19/04/13                       Added functionality for parts definitions.
%%                                  This required dissabling calls to
%%                                  MakeUppercase in the TOC when using
%%                                  hyperref.
%%
%%   03/05/13                       Preloading xcolor package using correct
%%                                  options for pdfdriver and hyperref.
%%

%% BEGIN CLASS FILE
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{edengths}[2012/02/12 University of Edinburgh Engineering thesis %
                        class]

%% LOAD PACKAGES REQUIRED FOR CLASS

%% `ifthen' provides simple boolean commands
\RequirePackage{ifthen}

%% 'ifpdf' detects whether pdflatex is being used.
\RequirePackage{ifpdf}

%% Calc package is needed for titlepage length calculation
\RequirePackage{calc}

%% DECLARE VARIABLES AND COMMANDS FOR CLASS

%% Define a command to switch between report and rapport3 classes.
%% 'rapport3' class does brutalist spacing, but the bibspacing needs
%% manually controlled as it sets this to zero.
\newcommand{\classfile}{report}

%% Define an option warning to print when unwanted options are given
\newcommand{\OptionWarning}[1]{\ClassWarning{edengths}{Option '#1' not available for edengths.}}

%% Define a command to declare the unused options in the report class, giving
%% a warning in the output.
\newcommand{\OffOption}[1]{\DeclareOption{#1}{\OptionNotUsed\OptionWarning{#1}}}

%% Define a command to declare the available options in the report class.
\newcommand{\OnOption}[1]{%
               \DeclareOption{#1}{\PassOptionsToClass{#1}{\classfile}}}

%% Make some booleans for special package options
\newboolean{sansflag}                   % Default is for sans serif fonts for 
\setboolean{sansflag}{true}             % captions and headers

\newboolean{msfontsflag}                % Use MS fonts
\setboolean{msfontsflag}{false}

\newboolean{fancychapflag}              % Use fancy chapter headerings
\setboolean{fancychapflag}{false}

\newboolean{crestflag}                  % Put a crest on the frontpage
\setboolean{crestflag}{false}

\newboolean{labelflag}                  % Print labels on the page
\setboolean{labelflag}{false}

\newboolean{twosideflag}                % Is the twoside option set?
\setboolean{twosideflag}{false}

\newboolean{bibflag}                    % Should the bibliography be printed
\setboolean{bibflag}{true}

\newboolean{hyperflag}
\setboolean{hyperflag}{false}           % Use hyperref package?

\newboolean{nomenclflag}
\setboolean{nomenclflag}{false}        % Use nomencl package?


%%%% DECLARE OPTIONS

%% LOCAL OPTIONS

%% Enable numbering of subsubsections (note: these don't appear in the contents)
\DeclareOption{subsubnos}{\addtocounter{secnumdepth}{1}}

%% Don't use the sans serif headers and captions
\DeclareOption{nosans}{\setboolean{sansflag}{false}}

%% Remove paragraph indent and add a line skip by switch class to rapport3
\DeclareOption{nopardent}{\renewcommand{\classfile}{rapport3}}

%% Use MS fonts rather than latex default
\DeclareOption{msfonts}{\setboolean{msfontsflag}{true}}

%% Use fancy chapter headings like jthesis-v1
\DeclareOption{fancychap}{\setboolean{fancychapflag}{true}}

%% Use a crest on the front page
\DeclareOption{crest}{\setboolean{crestflag}{true}}

%% Print equation labels next to equation numbers.
\DeclareOption{labels}{\setboolean{labelflag}{true}}

%% Repress the printing of the bibliography.
\DeclareOption{nobib}{\setboolean{bibflag}{false}}

%% Use hyperref
\DeclareOption{hyper}{\setboolean{hyperflag}{true}}

%% Use nomencl
\DeclareOption{nomencl}{\setboolean{nomenclflag}{true}}

%% Must have options.
\OnOption{a4paper}
\OnOption{openright}     % Start chapters on righthand side pages
%\OnOption{titlepage}     % Title should be on it's own page.

%% Pass options to the report class which are still OK
\OnOption{10pt}
\OnOption{11pt}
\OnOption{12pt}
\OnOption{draft}
\OnOption{final}
\OnOption{fleqno}
\OnOption{leqno}
\OnOption{oneside}

%% Set a boolean in addition for the twoside option
\DeclareOption{twoside}{\setboolean{twosideflag}{true}%
                        \PassOptionsToClass{twoside}{\classfile}}

%% Switch off `report' options that won't be allowed.
\OffOption{letterpaper}
\OffOption{legalpaper}
\OffOption{executivepaper}
\OffOption{a5paper}
\OffOption{b5paper}
\OffOption{landscape}
\OffOption{twocolumn}
%\OffOption{notitlepage}
\OffOption{openany}
\OffOption{openbib}

%% OTHER OPTIONS

%% Pass a warning for any options that are not known at this stage
\DeclareOption*{\OptionWarning{\CurrentOption}}

%% PROCESS OPTIONS
\ProcessOptions\relax

%% Load report class with default setup
\LoadClass[a4paper,openright,titlepage]{\classfile}

%% COLORS FIRST

%% xcolor should be loaded prior to the user inputted packages or stuff like
%% pstricks will not work apparently. xcolor gives better color match for
%% hyperlinks.
\ifthenelse{\boolean{pdf}}{%
    \newcommand{\colordriver}{pdftex}
    }{%
    \newcommand{\colordriver}{xdvi, dvipdfmx}
    }
\ifthenelse{\boolean{hyperflag}}{%
    \RequirePackage[\colordriver, dvipsnames, hyperref]{xcolor}%
   }{
    \RequirePackage[\colordriver, dvipsnames]{xcolor}%
   }

%%%% INPUT THE USER DEFINED PACKAGES (Loading here allows options to be passed
%%%%                                  to the auto loaded packages)

\InputIfFileExists{packages.tex}{\typeout{Succesful input of user packages.}}%
                                {\typeout{No user packages.tex file found.}}

%%%% LOAD MAIN PACKAGES

%% Putting some packages into the class.
%% There could be a number of options to ``turn off'' the styleised
%% parts and that the behavioural parts could be maintained in a separate file.

%% List the very must haves first...
\RequirePackage{graphicx}		        % Allows inclusion of graphics
						% in eps or jpg/pdf format

%% General formatting specified in format.tex
\RequirePackage[includehead]{geometry}          % A more modern way of setting 
                                                % the page margins.
                                                % Report options are passed
                                                % automatically.
\RequirePackage{setspace}                       % Define line spacing

%% Appendix
\RequirePackage{appendix}                       % Required for appendices

%% Bibliography
%\RequirePackage{natbib}                         % Natbib with default options
%\RequirePackage{bibentry}                       % Allow in text references and
                                                % hiding of bibliograhy.

%% FONTS 
\RequirePackage{ae}                             % Virtual fonts for T1 encoded
                                                % CMR-fonts.
\RequirePackage[T1]{fontenc}                    % T1 encoding stops some errors
                                                % for unknown fonts

\ifthenelse{\boolean{msfontsflag}}{%            % Use MS fonts if the option is
  \RequirePackage{mathptmx}%                    % given
  \RequirePackage[scaled=.92]{helvet}%
  }{%
  }

%% Provide the font options to packages for the sans serif fonts
%% This loads caption and subfig. Caption does caption formatting and
%% subfig allows subfigures.
\ifthenelse{\boolean{sansflag}}{%
   \RequirePackage[labelfont={sf,bf}, textfont=sf]{caption, subfig}%
  }{%
   \RequirePackage[labelfont=bf]{caption, subfig}%
  }

%% TITLES AND CAPTIONS

%% Titlesec does headers and title formating. The definitions for the chapter
%% headers are set below
\RequirePackage[nobottomtitles*,pagestyles,clearempty]{titlesec}%

%% Allow modifications to the able of contents and lists
\RequirePackage[titles,subfigure]{tocloft}%

%% NOMENCLATURE

%% Load the nomencl package. More work done on it lower down.
\ifthenelse{\boolean{nomenclflag}}{%
   \RequirePackage{nomencl}%
   \makenomenclature%
}{%
  % If nomencl is not loaded redefine printglossary to avoid errors.
  \def\printnomenclature{}%
}

%% HYPERLINKS

%% Load the hyperref package if the option is set. Set to the correct driver
%% dependant on the ifpdf package. Also load hypcap for better linking to 
%% figures.
\ifthenelse{\boolean{pdf}}{%
    \newcommand{\hyperdriver}{pdftex}
    }{%
    \newcommand{\hyperdriver}{dvipdfm}
    }
\ifthenelse{\boolean{hyperflag}}{%
    \AtEndOfClass{\RequirePackage[\hyperdriver]{hyperref}}%
    \AtEndOfClass{\RequirePackage[all]{hypcap}}%
   }{}

%% LABELS

%% Print labels on the page if option is given
\ifthenelse{\boolean{labelflag}}{%
   \AtEndOfClass{\RequirePackage[inner]{showlabels}}%
}{}


%%%% MORE PACKAGES CAN BE DEFINED IN packages.tex

%%%% FORMAT DEFINITIONS.

%% FONTS

%% Make a special command called defaultfont which sets sans or not. This is
%% the main control for changing the font style, only caption and subfig are
%% set otherwise, above.
\newcommand{\defaultfont}{\ifthenelse{\boolean{sansflag}}{\sffamily}{}}

%% Considering the optional fonts a lot of work has to be done to the table of
%% contents otherwise there is no action taken.
\ifthenelse{\boolean{sansflag}}%
  {%
    %% Make at symbol an internal command
    \makeatletter%
    %% Because of the font change, the page number becomes too large for the
    %% horizontal space LaTeX reserves for it by default. Without the following
    %% redefines to fix it, this would cause the Chapter entry page numbers
    %% to extend a few points into the right margin. The horror!
    \renewcommand{\@pnumwidth}{1.75em}%
    \renewcommand{\@tocrmarg}{2.75em}%
    %%
    %% Make at symbol safe again.
    \makeatother%
    %%
    %% Switch all list fonts to sans serif.
    \renewcommand{\cftpartfont}{\bfseries\sffamily}
    \renewcommand{\cftchapfont}{\bfseries\sffamily}%
    \renewcommand{\cftsecfont}{\sffamily}%
    \renewcommand{\cftsubsecfont}{\sffamily}%
    \renewcommand{\cftfigfont}{\sffamily}%
    \renewcommand{\cfttabfont}{\sffamily}%
    %%
    \renewcommand{\cftpartpagefont}{\bfseries\sffamily}%
    \renewcommand{\cftchappagefont}{\bfseries\sffamily}%
    \renewcommand{\cftsecpagefont}{\sffamily}%
    \renewcommand{\cftsubsecpagefont}{\sffamily}%
    \renewcommand{\cftfigpagefont}{\sffamily}%
    \renewcommand{\cfttabpagefont}{\sffamily}}%
  {%
    %% Otherwise, do nothing, I guess.
  }
  
%% PARTS

%% There is a particular issue with hyperef and the part definition which can be
%% fixed as follows:
\ifthenelse{\boolean{hyperflag}}{%
  \AtBeginDocument{%
    \pdfstringdefDisableCommands{%
      \let\MakeUppercase\relax%
    \typeout{Disabled uppercase in PDF strings}%
    } 
  }}{}

%% Standardise the part definitions
\titleformat{\part}[display]
   {\Huge\defaultfont\filcenter}
   {\MakeUppercase{\partname{} \thepart}}
   {0em}
   {}

%% The following code is recommended, but doesn't work. See:
%% http://tex.stackexchange.com/questions/6609/problems-with-part-labels-using-titlesec

% \makeatletter
% \let\titlesec@part\part
% \renewcommand{\part}{\@ifstar\part@star\part@nostar}
% \def\part@star#1{\NR@gettitle{#1}\titlesec@part*{#1}}
% \def\part@nostar{\@ifnextchar[\part@nostar@opt\part@nostar@nopt}
% \def\part@nostar@nopt#1{\NR@gettitle{#1}\titlesec@part{#1}}
% \def\part@nostar@opt[#1]#2{\NR@gettitle{#2}\titlesec@part[#1]{#2}}
% \makeatother

%% MAIN CHAPTER HEADINGS

%% These are produced using the titlesec package.

%% Check for fancychap flag and change chapter default if true
\ifthenelse{\boolean{fancychapflag}}{%
  \titleformat{\chapter}[display]%
    {\huge\normalfont\defaultfont\filleft\onehalfspacing%
    \titlerule[1pt]%
    \vspace{1pt}%
    \titlerule}%
    {%
    \vspace{1ex}%
    \chaptertitlename \space \thechapter}%
    {0.5ex}%
    {\bfseries\Huge}%
    [\vspace{1ex}%
    \titlerule]%
    %% Control the spacing of the numbered chapters.
    \titlespacing*{\chapter}{0pt}{0pt}{30pt}
    \titleformat{name=\chapter,numberless}[display]%
    {\huge\normalfont\defaultfont\filcenter}%
    %% Move the title to the top of the page
    {\vspace{-6ex}}%
    {0pt}%
    {\titlerule\huge}%
    [\vspace{\parskip}%
    \titlerule]
    %% Control the spacing of the un-numbered chapters.
    \titlespacing*{name=\chapter,numberless}{0pt}{0pt}{30pt}%
    }%
  {%% Fix chapter spacing to one and a half
    \titleformat{\chapter}[display]%
    {\huge\normalfont\defaultfont\bfseries\onehalfspacing}%
    {\chaptertitlename\ \thechapter}%
    {20pt}%
    {\Huge}%
  }

%% FRONT MATTER CHAPTER HEADINGS

%% Design some chapter headings for centred headings in the front matter like
%% the dmathesis style thesis. These are applied to the front matter using
%% a wrapper enviroment. Also kill the chapter numbers from \chapter, but still
%% add a TOC entry.

\newenvironment{precontent}
  {%% No chapter number, but still TOC entry.
   \setcounter{secnumdepth}{-1}%
   \renewcommand{\chaptermark}[1]{}%
   \pagestyle{plain}%
   \titlespacing*{\chapter}{0pt}{0pt}{30pt}%
   \ifthenelse{\boolean{fancychapflag}}{%
      %% 'numberless' definition above is sufficient.
    }{%
      %% Redifine the chapters to center.
      \titleformat{\chapter}[display]%
      {\normalfont\defaultfont\bfseries\filcenter}%
      {\chaptertitlename \thechapter}%
      {20pt}%
      {\Huge}%
   }}%
  {
   %% Restart numbering
   \setcounter{secnumdepth}{2}%
  }

%% SECTIONS

%% Protect the section definitions from the rapport3 class as
%% it does some weird alignment to them.
\titleformat{\section}
  {\normalfont\defaultfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\defaultfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\normalfont\defaultfont\normalsize\bfseries}{\thesubsubsection}{1em}{}


%%%% SINGLE SPACING QUOTE ENVIRONMENTS

\let\oldquote=\quote
\let\endoldquote=\endquote
\renewenvironment{quote}{%
    \begin{oldquote}%
    \begin{singlespace}%
  }{%
    \end{singlespace}%
    \end{oldquote}%
  }
\let\oldquotation=\quotation
\let\endoldquotation=\endquotation
\renewenvironment{quotation}{%
    \begin{oldquotation}%
    \begin{singlespace}%
  }{%
    \end{singlespace}%
    \end{oldquotation}%
  }


%%%% STRUCTURAL DEFINITIONS

%% TITLE PAGE

%% Define a command for the path the to crest. Default is to
%% stylefiles/EdUniCrest.pdf. Default will go to eps or to
%% pdf dependant on pdf option.
\ifthenelse{\boolean{pdf}}%
          {\newcommand{\crestdefault}{front/EdUniCrest.pdf}}%
          {\newcommand{\crestdefault}{front/EdUniCrest.eps}}

\newcommand{\crestfile}[1]{%
    \renewcommand{\crestdefault}{#1}}

%% If crestflag is true, check the path of the crest and set
%% crestflag to false if the path is wrong
\ifthenelse{\boolean{crestflag}}{%
  \IfFileExists{\crestdefault}
       {\typeout{Crest image file found.}}%
       {\setboolean{crestflag}{false}%
           \ClassWarning{edengths}{Crest image file not found.}}%
  }{% Do nothing if crestflag is false
  }

%% Add a command for defining the title of the qualification
\newcommand{\qualidefault}{\textit{Doctor of Philosophy}}%

\newcommand{\qualification}[1]{%
    \renewcommand{\qualidefault}{#1}}
    
%% Add a command for defining the name of the institution
\newcommand{\unidefault}{\textsc{The University of Edinburgh}}%

\newcommand{\university}[1]{%
    \renewcommand{\unidefault}{#1}}

%% Make a box and a height to record the height of the title
\newsavebox{\titlebox}
\newlength{\titleboxheight}

%% Allow internal use of @ symbol
\makeatletter

\renewcommand{\maketitle}{%
    %% Make the dummy title box to check the height
    \savebox{\titlebox}{%
      \parbox{0.9\linewidth}{%
        {\Huge\defaultfont\bfseries%
        \begin{onehalfspace}%
          \centering%
          \@title%
        \end{onehalfspace}%
        }%
      }%
    }%
    %% Get the height and if it's < 4cm set it to 4cm.
    \setlength{\titleboxheight}{\ht\titlebox + \dp\titlebox}%
    \ifthenelse{\lengthtest{ \titleboxheight < 4cm }}{
       \setlength{\titleboxheight}{4cm}}{}%
    \thispagestyle{empty}%
    \null\vfil%
    \begin{center}%
      {%\parskip=0pt
      %% Title in a parbox. Set its height to \titleboxheight
      \hrule%
      \vskip 1em%
      {\Huge\defaultfont\bfseries%
        \parbox[c][\titleboxheight][c]{0.9\linewidth}{%
          \begin{onehalfspace}%
            \centering%
            \@title%
          \end{onehalfspace}%
        }%
      }%
      \vskip 1em%
      \hrule%
      \vskip 1.5em%
      %% Author
      {\Large\itshape%
      \parbox[c]{0.9\linewidth}{%
        \centering%
        \@author%
      }}%
      %% University crest. Check crestflag.
      \ifthenelse{\boolean{crestflag}}{%
        \vskip 10mm%
        \centering%
          \includegraphics[width=2in]{\crestdefault}\par%
        \vskip 10mm%
      }{%
         \vskip 70mm%
      }%
      %% Qualification
      {\Large%
      \parbox[c]{10cm}{%
        \begin{onehalfspace}%
          \centering%
          \qualidefault%
        \end{onehalfspace}%
      }\\%
      %% University
      [4mm] {\unidefault} \\%
      %% Date
      [10mm]\@date
      }}%
    \end{center}\par%
    \vfil\null}

%% Make at safe again.
\makeatother

%% DEDICATION

\newcommand{\dedication}[1]{%%
   \thispagestyle{empty}
   \begin{center}%
    \vspace*{2cm}%
    \textit{\Large {#1}}%
  \end{center}%
}

%% ABSTRACT

\renewcommand{\abstract}[1]{%
  \chapter{\abstractname}%
  #1%
}

%% ACKNOWLEDGEMENTS

\newcommand{\acknowledgements}[1]{%
  \chapter{Acknowledgements}%
  #1%
}

%% DECLARATION

\newcommand{\declaration}[1]{%
  \chapter{Declaration}%
  #1
}

\newcommand{\standarddeclaration}[1]{%
  % Set up a box for the author name
  \newsavebox{\mybox}%
  \sbox{\mybox}{\bfseries{#1}}%
  %
  % Now measure its length
  \newlength{\myboxlen}%
  \settowidth{\myboxlen}{\usebox{\mybox}}%
  %
  \chapter{Declaration}%
  %
  I declare that this thesis was composed by myself, that the work contained herein is %
  my own except where explicitly stated otherwise in the text, and that this work has not %
  been submitted for any other degree or professional qualification except as specified.%
  %
  \vfill
  %
  % Add a line with the authors name underneath.
  % The line should not be less than 0.4/textwidth
  \noindent%
  \ifthenelse{\lengthtest{ \myboxlen < 0.4\textwidth }}{%
       \rule{0.4\textwidth}{.01in}\\}{%
       \rule{\myboxlen}{.01in}\\%
       }%
  %
  \usebox{\mybox}\\%
  %
  \vspace*{5cm}%
}


%% REDEFINE LOF AND LOT TO ADD LINE TO TOC

%% Maually add a line to the contents with phantomsection
%% to make better links if hyperref is used.
\let\oldlistoffigures\listoffigures
\renewcommand{\listoffigures}{
    \cleardoublepage
    \phantomsection
    \addcontentsline{toc}{chapter}{\listfigurename}
    \oldlistoffigures
}

\let\oldlistoftables\listoftables
\renewcommand{\listoftables}{
    \cleardoublepage
    \phantomsection
    \addcontentsline{toc}{chapter}{\listtablename}
    \oldlistoftables
}

%% ADD A FRONT MATTER CHAPTER COMMAND.
%% STARRED VERSION HAS NO TOC ENTRY.

%% Allow internal use of @ symbol
\makeatletter

\newcommand{\frontchapter}{\@ifstar
                     \frontchapterStar%
                     \frontchapterNoStar%
}

\newcommand{\frontchapterStar}[1]{%
    \cleardoublepage%
    \phantomsection%
    \chapter*{{#1}%
             \@mkboth{\MakeUppercase{#1}}%
                     {\MakeUppercase{#1}}%
             }%
    }

\newcommand{\frontchapterNoStar}[1]{%
    \cleardoublepage%
    \phantomsection%
    \addcontentsline{toc}{chapter}{#1}%
    \chapter*{{#1}%
             \@mkboth{\MakeUppercase{#1}}%
                     {\MakeUppercase{#1}}%
             }%
    }

%% Make at safe again.
\makeatother

%% DEFINITION OF A COMBINED FIGURE AND TABLE LIST

%% Make a name for it.
\newcommand{\listfiguretablename}{Figures and Tables}

%% Allow internal use of @ symbol
\makeatletter

\newcommand\listoffiguresandtables{%
    \if@twocolumn%
      \@restonecoltrue\onecolumn%
    \else%
      \@restonecolfalse%
    \fi%
    \frontchapter{\listfiguretablename}%
    \section*{Figures}%
     {\@starttoc{lof}}%
    \mrule%
    \section*{Tables}%
     {\@starttoc{lot}}%
    \if@restonecol\twocolumn\fi%%
    }

%% Make at safe again.
\makeatother

%% FIX THE NOMENCLATURE TITLE IF THE PACKAGE IS BEING USED.

%% Allow internal use of @ symbol
\makeatletter

\ifthenelse{\boolean{nomenclflag}}{%
  %% Redefine \theglossary command
  \renewcommand{\thenomenclature}{%
    \frontchapter{\nomname}%
    \nompreamble
    \list{}{%
      \labelwidth\nom@tempdim
      \leftmargin\labelwidth
      \advance\leftmargin\labelsep
      \itemsep\nomitemsep
      \let\makelabel\nomlabel}}%
}{}

%% Make at safe again.
\makeatother

%% DEFINE BIBLIOGRAPHY HYPHENATION PENALTY
\let\bibhyphenpenalty\hyphenpenalty

%% DEFINE EDBIBLIOGRAPHY TO ADD LINE TO TOC AND SET
%% HYPHENPENTALTY

%% Maually add a line to the contents with phantomsection
%% to make better links if hyperref is used.
%% Choose between nobib or bib dependent on nobib option.
\newcommand{\edbibliography}{%
    \cleardoublepage%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\bibname}%
    \message{Bibliography hyphenation penalty: \the\bibhyphenpenalty}%
    \hyphenpenalty=\bibhyphenpenalty%
    \ifthenelse{\boolean{bibflag}}%
    {%
      \bibliography%
    }{%
      \nobibliography%
    }
}

%% SHORTCUT WRAPPERS

%% Wrapper to produce the front matter in one go.
\newcommand{\makeprecontent}[1]{%
    %% Start the page numbering in roman
    \pagenumbering{roman}%
    \setcounter{page}{1}%
    %% Make the title page
    %\maketitle%
    %\cleardoublepage%
    \begin{precontent}%
       \input{#1}%
    \end{precontent}
    %% Make sure declaration has right page style
    %% and page numbering is sane.
    \cleardoublepage
    %% Use alternative style if twosideflag is true
    \ifthenelse{\boolean{twosideflag}}%
      {\pagestyle{twomain}}%
      {\pagestyle{main}}%
    }

%% Wrapper to call commands to start the main body text.
\newcommand{\startbody}{%
    \cleardoublepage%
    \pagenumbering{arabic}%
    \setcounter{page}{1}%
    %% Use alternative style if twosideflag is true
    \ifthenelse{\boolean{twosideflag}}%
      {\pagestyle{twomain}}%
      {\pagestyle{main}}%
    }

%% Appendix work is now done in a separate file
%% appendix/edengapp.tex

%%%% OTHER STUFF

%% Define new counter so you can have the equation
%% number 4.2.1a for example, this a gift from J.F.Blowey
%% Usage \eqlabon to start a,b numbering and \eqlaboff
%% to stop.

\newcounter{ind}
\newcommand{\eqlabon}{%
  \setcounter{ind}{\value{equation}}\addtocounter{ind}{1}%
  \setcounter{equation}{0}%
  \renewcommand{\theequation}{\arabic{chapter}%
          .\arabic{section}.\arabic{ind}\alph{equation}}}

\newcommand{\eqlaboff}{%
  \renewcommand{\theequation}{\arabic{chapter}%
          .\arabic{section}.\arabic{equation}}%
  \setcounter{equation}{\value{ind}}}

%% Thick horizontal line.
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

%% Short centred horizontal line.
\newcommand{\mrule}{%
   \begin{center}%
    \rule{0.5\linewidth}{0.2mm}%
   \end{center}}

%%%% INPUT THE USER DEFINED FORMATTING FILE

\AtEndOfClass{%
\InputIfFileExists{edengfmt.tex}{\typeout{Succesful input of formatting file.}}%
                                {\ClassError{edengths}{%
                                 Formatting file not found!%
                                }{%
                                 Oh dear! Something's gone wrong. \MessageBreak%
                                 Where is edengfmt.tex? I need it!}}%
}
