%
%% PrincetonThesis.cls - LaTeX2e thesis class
%
% Copyright (C) 2002 Mike Nolta <mrnolta@princeton.edu>
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public
% License along with this program; if not, write to the
% Free Software Foundation, Inc., 59 Temple Place - Suite 330,
% Boston, MA  02111-1307, USA.
%
%% About
%
% This is a LaTeX 2e class for Princeton University Ph.D. disserations,
% released in the hope it will be useful. Worked for me - YMMV.
%
%% Credits
%
% Put together by Mike Nolta. Much of the code was stolen from the
% University of Colorado thesis class. Other sources included ucthesis.cls,
% puthesis.sty, and the Pulsar Lab style file.
%
%% ChangeLog
%
%	1.0	2002-07-01	public release
%
%% Example
%
% In this example the main file is called thesis.tex, BibTeX is used
% to manage the bibliography, and the BibTeX file is thesis.bib. The
% abstract, acknowledgements, and chapter bodies are stored in separate
% files.
%	
%	\documentclass[phd,black]{PrincetonThesis}
%	\usepackage{epsfig}
%	
%	\title{Your Title Here}
%	\author{Your Name Here}
%	\department{Department}
%	\adviser{Your Adviser}
%	\degreemonth{January/June/November}
%	\degreeyear{Year}
%	
%	\begin{document}
%	
%	\begin{frontmatter}
%	
%	  \begin{thesisabstract}
%	  \input abstract.tex
%	  \end{thesisabstract}
%	
%	  \begin{acknowledgements}
%	  \input acknowledgements.tex
%	  \end{acknowledgements}
%	
%	\end{frontmatter}
%	
%	\cleardoublepage
%	\chapter{Chapter Name}
%	\input chapter1.tex
%	
%	...
%	
%	\appendix
%	
%	\cleardoublepage
%	\chapter{Appendix Name}
%	\input appendix1.tex
%	
%	...
%	
%	\cleardoublepage
%	\nocite{*} %% use everything in bibtex file, even if not cited
%	\bibliographystyle{abbrv} %% or your favorite style
%	\bibliography{thesis} %% assuming your bibtex file is thesis.bib
%	
%	\end{document}
%	
% To turn the file (thesis.tex) into postscript, run the following:
%
%	$ latex thesis
%	$ bibtex thesis
%	$ latex thesis
%	$ latex thesis
%	$ dvips thesis -o
%
%% Documentation
%
% \documentclass flags ::
%
%	phd
%		PhD thesis.
%
%	senior
%		Senior thesis. NB: this mode is completely untested.
%
%	black
%		Single-sided output suitable for binding. This is the default.
%
%	orange
%		Double-sided output format for unofficial copies.
%
% thesistable environment ::
%
%   A simple table environment (thesistable) simplifies the creation of
%   tables. The output format is similar to those appearing in physics
%   journals. Usage is similar to the LaTeX tabular environment:
%
%	\begin{thesistable}{(cols)}
%	(column 1 head) & ... & (column n head) \\
%	\startdata
%	... (table data) ...
%	\enddata
%	\caption{\label{...}
%	... (table caption) ...}
%	\end{thesistable}
%
%   Here, (cols) specifies the column formatting. It consists of a sequence
%   of specifiers (e.g. l, c, r), at least one for each of the columns.
%   See the LaTeX documentation on tabular for more information.
%
%% Problems
%
% - Output is too high on the page.
%
%   Note that Linux systems (which use teTeX) default to the ISO A4 paper
%   size, which is slightly longer than American letter. To set the default
%   to letter, run
%
%	$ texconfig dvips paper letter
%
%   as root.
%
%%
%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{PrincetonThesis}[2002/07/01 v1.0 Princeton Univ. thesis class]

%% OPTIONS

% options we inherit from report.cls

\DeclareOption{10pt}{\PassOptionsToClass{10pt}{report}}
\DeclareOption{11pt}{\PassOptionsToClass{11pt}{report}}
\DeclareOption{12pt}{\PassOptionsToClass{12pt}{report}}
\DeclareOption{draft}{\PassOptionsToClass{draft}{report}}
\DeclareOption{final}{\PassOptionsToClass{final}{report}}
\DeclareOption{fleqno}{\PassOptionsToClass{fleqno}{report}}
\DeclareOption{leqno}{\PassOptionsToClass{leqno}{report}}
\DeclareOption{openany}{\PassOptionsToClass{openany}{report}}
\DeclareOption{openbib}{\PassOptionsToClass{openbib}{report}}
\DeclareOption{openright}{\PassOptionsToClass{openright}{report}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{report}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{report}}

% options we don't inherit from report.cls

\DeclareOption{titlepage}{\ClassWarning{PrincetonThesis}
	{Option 'titlepage' not available for PrincetonThesis.}}
\DeclareOption{notitlepage}{\ClassWarning{PrincetonThesis}
	{Option 'notitlepage' not available for PrincetonThesis.}}

% our own options

\newif\if@PUThesisPhD
\@PUThesisPhDfalse
\DeclareOption{phd}{\@PUThesisPhDtrue}

\newif\if@PUThesisSenior
\@PUThesisSeniorfalse
\DeclareOption{senior}{\@PUThesisSeniortrue}

\newif\if@PUThesisBlack
\@PUThesisBlackfalse
\DeclareOption{black}{\PassOptionsToClass{oneside}{report}\@PUThesisBlacktrue}

\newif\if@PUThesisOrange
\@PUThesisOrangefalse
\DeclareOption{orange}{\PassOptionsToClass{twoside}{report}\@PUThesisOrangetrue}

% load report

\ProcessOptions
\LoadClass[11pt,letterpaper,oneside,onecolumn]{report}[1996/10/31]

%% USERINFO

\renewcommand*{\author}[1]{\gdef\pu@author{#1}}
\renewcommand*{\title}[1]{\gdef\pu@title{#1}}

\newcommand*{\advisor}[1]{\gdef\pu@advisor{#1}}
\newcommand*{\degreemonth}[1]{\gdef\pu@degreemonth{#1}}
\newcommand*{\degreeyear}[1]{\gdef\pu@degreeyear{#1}}
\newcommand*{\department}[1]{\gdef\pu@department{#1}}

%% PAGE HEADINGS

\if@twoside
	\def\ps@plain{%
		%\def\@evenfoot{}\def\@oddfoot{}%
		%\def\@evenhead{%
		%	\rmfamily\thepage\quad{\slshape\leftmark}\hfil\null}%
		%\def\@oddhead{%
		%	\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage}%
		\def\@evenhead{}\def\@oddhead{}%
		\def\@evenfoot{%
			\null\hfil\rmfamily\thepage\quad{\slshape\leftmark}\hfil\null}%
		\def\@oddfoot{%
			\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage\hfil\null}%
		\def\chaptermark##1{}%
		\def\sectionmark##1{}%
	}
	\def\ps@headings{%
		\def\@evenfoot{}\def\@oddfoot{}%
		\def\@evenhead{%
			\rmfamily\thepage\quad{\slshape\leftmark}\hfil\null}%
		\def\@oddhead{%
			\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage}%
		\def\chaptermark##1{\markboth {##1}{}}%
		\def\sectionmark##1{\markright %
			{\ifnum \c@secnumdepth >\z@ \thesection\ \fi ##1}}%
	}
\else
	\def\ps@plain{%
		%\def\@evenfoot{}\def\@oddfoot{}\def\@evenhead{}%
		%\def\@oddhead{%
		%	\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage}%
		\def\@evenhead{}\def\@oddhead{}\def\@evenfoot{}%
		\def\@oddfoot{%
			\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage\hfil\null}%
		\def\chaptermark##1{}%
		\def\sectionmark##1{}%
	}
	\def\ps@headings{%
		\def\@evenfoot{}\def\@oddfoot{}\def\@evenhead{}%
		\def\@oddhead{%
			\null\hfil{\slshape\rightmark}\quad\rmfamily\thepage}%
		\def\chaptermark##1{\markboth {##1}{}}%
		\def\sectionmark##1{}%
	}
\fi

% no headers on blank pages
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else%
	\hbox{}%
	\thispagestyle{empty}%
	\newpage%
	\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% SPACING

\def\setspace@size{\ifx\@currsize\normalsize\@normalsize\else\@currsize\fi}

\def\singlespace{
	\vskip \baselineskip
	\def\baselinestretch{1}
	\setspace@size
	\vskip -\baselineskip
}
\def\endsinglespace{\par}

\def\spacing#1{
	\par
	\begingroup
	\def\baselinestretch{#1}
	\setspace@size
}
\def\endspacing{
	\par
	\vskip \parskip
	\vskip \baselineskip
	\endgroup
	\vskip -\parskip
	\vskip -\baselineskip
}

%\if@twoside
%	\def\doublespacing{
%		\ifcase \@ptsize \relax
%		    \def\baselinestretch{2.004}
%		\or \def\baselinestretch{1.771}
%		\or \def\baselinestretch{1.660}
%		\fi
%		\setspace@size
%	}
%	\def\doublespace{
%		\ifcase \@ptsize \relax
%		    \spacing{2.004}
%		\or \spacing{1.771}
%		\or \spacing{1.660}
%		\fi
%	}
%\else
%	\def\doublespacing{
%		\ifcase \@ptsize \relax
%		    \def\baselinestretch{1.703}
%		\or \def\baselinestretch{1.505}
%		\or \def\baselinestretch{1.441}
%		\fi
%		\setspace@size
%	}
%	\def\doublespace{
%		\ifcase \@ptsize \relax
%		    \spacing{1.703}
%		\or \spacing{1.505}
%		\or \spacing{1.441}
%		\fi
%	}
%\fi

\def\doublespacing{
	\def\baselinestretch{1.6}
	\setspace@size
}
\def\doublespace{
	\spacing{1.6}
}
\let\enddoublespace=\endspacing

%% FLOATS
%
% single space floating material
%

\let\latex@xfloat=\@xfloat
\def\@xfloat #1[#2]{
	\latex@xfloat #1[#2]
	\def\baselinestretch{1}
	\@normalsize \normalsize
}

%% FOOTNOTE
%
% single space footnotes
%

\long\def\@footnotetext#1{
\insert\footins{
	\def\baselinestretch{1}
	\reset@font\footnotesize
	\interlinepenalty\interfootnotelinepenalty
	\splittopskip\footnotesep
	\splitmaxdepth \dp\strutbox \floatingpenalty \@MM
	\hsize\columnwidth \@parboxrestore
	\protected@edef\@currentlabel
      		{\csname p@footnote\endcsname\@thefnmark}
	\color@begingroup \@makefntext
		{\rule\z@\footnotesep\ignorespaces#1\@finalstrut\strutbox}
	\color@endgroup
}}

\long\def\@mpfootnotetext#1{
\global\setbox\@mpfootins\vbox{
	\unvbox \@mpfootins
	\def\baselinestretch {1}
	\reset@font\footnotesize
	\hsize\columnwidth \@parboxrestore
	\protected@edef\@currentlabel
		{\csname p@mpfootnote\endcsname\@thefnmark}
	\color@begingroup \@makefntext
		{\rule\z@\footnotesep\ignorespaces#1\@finalstrut\strutbox}
	\color@endgroup
}}

%% TITLE PAGE

\if@PUThesisPhD
\newcommand*{\pu@titlepage}{
	\thispagestyle{empty}
	\null\vfil
	\begin{center}
		{\LARGE\pu@title}
	\vskip 3em
		{\Large\pu@author}
	\vskip 3em
	\large\baselineskip 2em
		A Dissertation\\
		Presented to the Faculty\\
		of Princeton University\\
		in Candidacy for the Degree\\
		of Doctor of Philosophy\\
	\vskip 2em
		Recommended for Acceptance\\
		by the Department of\\
		\pu@department \\
	\vskip 2em
		\pu@degreemonth, \pu@degreeyear
	\end{center}
	\null\vfil
	\clearpage
}
\fi

\if@PUThesisSenior
\newcommand*{\pu@titlepage}{
	\thispagestyle{empty}
	\null\vfil
	\begin{center}
		{\LARGE\pu@title}
	\vskip 5em
		{\Large\pu@author}
	\vfil
		\sc a senior thesis submitted in partial fulfillment\\
		of the requirements for the degree of Bachelor of Arts in
		\pu@department at Princeton University\\
	\vskip 2em
		\pu@degreeyear
	\end{center}
	\null\vfil
	\clearpage
}
\fi

%% COPYRIGHT PAGE

\newcommand*{\pu@copyrightpage}{
	\thispagestyle{empty}
	\null\vskip 3em
	\begin{center}
		\copyright\ Copyright \pu@degreeyear\ by \pu@author.\\
		All rights reserved.
	\end{center}
	\clearpage
}

%% TABLE OF CONTENTS

\newcommand*{\pu@tocpage}{
	\addcontentsline{toc}{chapter}{\contentsname}
	\tableofcontents
	\clearpage
}

\newcommand*{\pu@lofpage}{
	\addcontentsline{toc}{chapter}{\listfigurename}
	\listoffigures
	\clearpage
}

\newcommand*{\pu@lotpage}{
	\addcontentsline{toc}{chapter}{\listtablename}
	\listoftables
	\clearpage
}

%% QUOTE, QUOTATION, VERSE, VERBATIM
%
% single space quotes
%

\let\orig@quote\quote
\let\endorig@quote\endquote
\renewenvironment*{quote}
	{\begin{orig@quote}\begin{singlespace}}
	{\end{singlespace}\end{orig@quote}}

\let\old@quotation\quotation
\let\endold@quotation\endquotation
\renewenvironment*{quotation}
	{\begin{old@quotation}\begin{singlespace}}
        {\end{singlespace}\end{old@quotation}}

\let\old@verse\verse
\let\endold@verse\endverse
\renewenvironment*{verse}
	{\begin{old@verse}\begin{singlespace}}
        {\end{singlespace}\end{old@verse}}

\let\orig@verbatim\verbatim
\let\endorig@verbatim\endverbatim
\renewenvironment*{verbatim}
	{\begin{orig@verbatim}\begin{singlespace}}
	{\end{singlespace}\end{orig@verbatim}}

%% Tables: single-spaced

\let\orig@tabular\tabular
\let\endorig@tabular\endtabular
\renewenvironment*{tabular}[1]
	{\begin{singlespace}\begin{orig@tabular}{#1}}
	{\end{orig@tabular}\end{singlespace}}

%\let\orig@table\table
%\let\endorig@table\endtable
%\renewenvironment*{table}[1]
%	{\begin{singlespace}\begin{orig@table}{#1}}
%	{\end{orig@table}\end{singlespace}}

\newcommand{\datastart}{\noalign{\smallskip}\hline\noalign{\smallskip}}
\newcommand{\dataend}{\noalign{\smallskip}\hline\end{tabular}}

\newenvironment{thesistable}[2][\normalsize]
	{\begin{table}[t]\begin{center}#1
		\begin{tabular}{#2}\hline\hline\noalign{\smallskip}}
	{\end{center}\end{table}}

\newenvironment{thesisbtable}[2][\normalsize]
	{\begin{table}[b]\begin{center}#1
		\begin{tabular}{#2}\hline\hline\noalign{\smallskip}}
	{\end{center}\end{table}}

% aastex compatibility

\let\startdata\datastart
\let\enddata\dataend
\newcommand{\colhead}[1]{#1}
\newcommand{\tablehead}[1]{#1\\}
\newcommand{\tablewidth}[1]{}
\newcommand{\tablecaption}[1]{}
\newcommand{\tablenotemark}[1]{${}^{#1}$}
\newcommand{\tablecomments}[1]{}
\newcommand{\tablenotetext}[2]{${}^{#1}${#2}}

%% BIBLIOGRAPHY

\renewcommand*{\bibname}{References}

%% ABSTRACT / ACKNOWLEDGEMENTS

%\let\orig@abstract\abstract
%\let\endorig@abstract\endabstract
%\renewenvironment*{abstract}
%	{\begin{orig@abstract}\addcontentsline{toc}{chapter}{Abstract}}
%	{\end{orig@abstract}}

\newenvironment{thesisabstract}
	{\chapter*{Abstract}\addcontentsline{toc}{chapter}{Abstract}}
	{\clearpage}

\newcommand{\ackname}{Acknowledgements}
\newenvironment{acknowledgements}
	{\chapter*{\ackname}\addcontentsline{toc}{chapter}{\ackname}}
	{\clearpage}

%% CHAPTER

%\let\orig@chapter\chapter
%\renewcommand*{\chapter}[1]{\orig@chapter{#1}}

%% INTEGRATION

\let\pu@document\document
\renewcommand*{\document}{
	\pu@document
	\pu@titlepage
	\pu@copyrightpage
}

\newenvironment{frontmatter}
{
	\cleardoublepage
	\pagestyle{plain}
	\pagenumbering{roman}
	\setcounter{page}{3}
}{
	\pu@tocpage
	\pu@lofpage
	\pu@lotpage
	\cleardoublepage
	\pagestyle{headings}
	\pagenumbering{arabic}
}

\AtBeginDocument{
	\doublespacing
	\raggedbottom
	\pagestyle{headings}

	% natbib redefines \thebibliography so we have to put this here
	\let\pu@thebibliography\thebibliography
	\let\endpu@thebibliography\endthebibliography
	\renewenvironment*{thebibliography}[1]
	{
		\begin{pu@thebibliography}{#1}
		\chaptermark{\bibname}
		\addcontentsline{toc}{chapter}{\bibname}
	}{
		\clearpage
		\end{pu@thebibliography}
	}
}

\setlength{\textheight}{8.5in}
\setlength{\textwidth}{6in}
\setlength{\oddsidemargin}{0.525in}
\setlength{\evensidemargin}{0.025in}

%% end of file
\endinput
