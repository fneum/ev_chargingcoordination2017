%
% $Id: osudissert96.cls,v 1.1 1996/10/11 17:56:24 hanes Exp $
%
% Modified:  1996/08/01 mdh to conform to 1995/1996 GS Format
% Renamed:   1996/10/11 mdh to allow for backwards compatability.
%
% Original Author: Manas Mandal
 
%% This is the LaTeX2e document class for OSU dissertations according
%% to the 1999 Graduate School specifications
%%
%% This class is based on the standard LaTeX class `report'.

\def\rcsStripFile$#1: #2,v ${#2}
\def\rcsStripDate$#1: #2 #3 ${#2}
\def\rcsStrip$#1: #2 ${#2}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{osudissert96}[\rcsStripDate$Date: 1996/10/11 17:56:24 $ 1996 OSU Thesis/Dissertation Class]
\typeout{*** Conforms to the 1996 Graduate School Format ***}
\typeout{Version \rcsStrip$Revision: 1.1 $  of \rcsStripFile $RCSfile: osudissert96.cls,v $ (\rcsStripDate$Date: 1996/10/11 17:56:24 $)}

%% Dissertation style options (other than `report' options) are
% phd (default) | dma | masters | ms | ma
% osudraft (default is off)

\newif\if@phd\@phdtrue
\newif\if@dma\@dmafalse
\newif\if@masters\@mastersfalse
\newif\if@ms\@msfalse
\newif\if@ma\@mafalse
\newif\if@osudraft\@osudraftfalse

\DeclareOption{phd}{\@phdtrue}
\DeclareOption{dma}{\@dmatrue\@phdfalse}
\DeclareOption{masters}{\@masterstrue\@phdfalse}
\DeclareOption{ms}{\@mstrue\@phdfalse}
\DeclareOption{ma}{\@matrue\@phdfalse}
\DeclareOption{osudraft}{\@osudrafttrue}

\ExecuteOptions{phd}

%% Only allowed `report' options from the full list (see page 177 of
%% LaTeX book) are shown below. Some options are disallowed because
%% they don't make sense in an OSU dissertation. Options marked
%% ``heck, why not?'' mean OSU Grad School's ``Ugly Brown Book''
%% doesn't care as far as we know, so let's allow the user some
%% creativity. If a default option is the only alternative, specifying
%% it will make no difference.

% 12pt (default)
% letterpaper (default)
% final (default) | draft
% oneside (default) | twoside (can be used for the TR form)
% openany (default) | openright
% onecolumn (default; twocolumn not allowed by Grad School. MM. 1994/08/12)
% titlepage (default)
% openbib (heck, why not?) mdh - there's a good reason. 1996/04/28
% leqno (heck, why not?)
% fleqn (heck, why not?)


%% Switch off remaining `report' options that won't be allowed.

\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
%\DeclareOption{12pt}{\OptionNotUsed}          % default
%\DeclareOption{letterpaper}{\OptionNotUsed}   % default
\DeclareOption{legalpaper}{\OptionNotUsed}
\DeclareOption{executivepaper}{\OptionNotUsed}
\DeclareOption{a4paper}{\OptionNotUsed}
\DeclareOption{a5paper}{\OptionNotUsed}
\DeclareOption{b5paper}{\OptionNotUsed}
\DeclareOption{landscape}{\OptionNotUsed}
%\DeclareOption{final}{\OptionNotUsed}
%\DeclareOption{draft}{\OptionNotUsed}
%\DeclareOption{oneside}{\OptionNotUsed}
%\DeclareOption{twoside}{\OptionNotUsed}
%\DeclareOption{openright}{\OptionNotUsed}
%\DeclareOption{openany}{\OptionNotUsed}
%\DeclareOption{onecolumn}{\OptionNotUsed}
\DeclareOption{twocolumn}{\OptionNotUsed}
%\DeclareOption{titlepage}{\OptionNotUsed}     % default
\DeclareOption{notitlepage}{\OptionNotUsed}
\DeclareOption{openbib}{\OptionNotUsed}
%\DeclareOption{leqno}{\OptionNotUsed}
%\DeclareOption{fleqn}{\OptionNotUsed}


%% The default action for any option is to pass it to the report class.
 
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
 

%% Execute the options.
 
\ProcessOptions
 

%% Load the report class in 12pt, letterpaper, titlepage (although we
%% shall redefine maketitle later), onecolumn
 
\LoadClass[12pt,letterpaper,titlepage,onecolumn]{report}



%\RequirePackage{newdissert}

%% MM. 1994/07/26.
% Requiring/using newdissert as a package causes some problems because
% the following latex209 variables
%       \footheight, \@maxsep, \@dblmaxsep
% are not available in LaTeX2e. (I carefully read classes.dtx and
% clsguide.tex, and concluded that those 3 variables won't matter.)
% Also, it is a mistake to redefine \newcounter for
%       part, chapter, section, subsection, subsubsection, paragraph
%       figure, table


\RequirePackage{osudissert96-mods}

%% MM. 1994/07/26.
% Blindly including all the newdissert style files into newdissert-mod,
% and commenting the above definitions makes it work almost as good
% as 2e in 209 compatible mode with the newdissert style.

%% MM. 1994/07/31.
% Fixed newdissert-mod. Result is exactly the same as what newdissert.sty
% produces using latex209, and that is the document that meets grad
% school rules. Fixing newdissert-mod, however, caused it to be
% unusable from 209. Ultimately, osudissert.cls should become a single
% file to be used with 2e, unless others have other ideas .....
% \documentclass[12pt,letterpage,titlepage]{report} followed by
% \usepackage{newdissert-mod} does the same thing as
% \documentclass{osudissert} which I prefer.

%% MM. 1994/08/09
% Many changes to newdissert-mod. Result is exactly the same as what
% newdissert.sty produced. Most commands are now ``renew''ed versions
% of report.cls commands.

%% MM. 1994/08/12.
% Added compatibility check to newdissert-mod so that, in compat mode,
% \documentstyle[12pt,newdissert-mod]{report} works well enough.

%% MM. 1994/08/13.
% Added all these cool initializations for degree options. Since I can't
% force exactly one option to be passed, I make phd option most important

%% MM. 1994/08/17. Cleaned up newdissert-mod. Used (re)newenvironment
%% whenever possible. Used chapter* wherever appropriate. Did leave
%% a few defs in there, esp. Al's stuff and a few variables. Oh yes,
%% fixed a few bugs that came in because I redid all the definitions.

%% MM. 1994/08/22. Changed some stuff in ms abstract.
%% Also added osudraft option.

%% MM. 1994/08/23. Changed float/text spacing yet again.

%% MM. 1995/02/11. Made Index single-space.

%% MM. 1995/02/18. Added co-advisor support to newdissert-mod.

%% MM. 1995/02/23. Fixed osudraft.

%% MM. 1995/03/08. Changed quotations to single-space. Fixed widows in toc.

% Defaults of newdissert-mod again, just for the heck of it.
\def\@degree{Doctor of Philosophy}
\def\@committee{Dissertation}
\def\@thesis{DISSERTATION}
\def\@degreeabbrev{Ph.D.}

% masters defaults to M.S.
\if@masters
  \def\@degree{Master of Science}
  \def\@degreeabbrev{M.S.}
\fi

\if@ms
  \def\@degree{Master of Science}
  \def\@degreeabbrev{M.S.}
  \@masterstrue
\fi
\if@ma
  \def\@degree{Master of Arts}
  \def\@degreeabbrev{M.A.}
  \@masterstrue
\fi

% Since I don't know how to allow only one option, I do phd first.
\if@phd
  \def\@degree{Doctor of Philosophy}
  \def\@degreeabbrev{Ph.D.}
  \def\@committee{Dissertation}
  \def\@thesis{DISSERTATION}
\else
  \if@dma
    \def\@degree{Doctor of Musical Arts}
    \def\@degreeabbrev{D.M.A.}
    \def\@committee{D.M.A.}
    \def\@thesis{D.M.A. DOCUMENT}
  \else
    \if@masters
      \def\@thesis{A Thesis}
      \def\@committee{Master's Examination}
      \thesis
    \fi
  \fi
\fi


\endinput
%% 
%% End of file `osudissert96.cls'.

