import pytest

# Fixtures - reusable test data
@pytest.fixture
def sample_metadata():
    """Sample paper metadata."""
    return {
        "title": "Test Paper",
        "authors": "John Doe",
        "abstract": "This is a test abstract."
    }


@pytest.fixture
def sample_latex_text():
    """Sample LaTeX text with sections."""
    return r"""
\documentclass{article}
\begin{document}

\begin{abstract}
This is the abstract content with sufficient length for parsing.
\end{abstract}

\section{Introduction}
This is the introduction content with some details and sufficient text length.

\section{Related Work}
This section discusses related work in the field with proper content.

\subsection{Previous Methods}
Our proposed method is described here with adequate detail.

\section{Experiments}
We conducted several experiments to validate our approach properly.

\section{Conclusion}
In conclusion, we achieved good results and contributions.

\begin{thebibliography}{9}
\bibitem{smith} Smith et al. 2020
\bibitem{jones} Jones et al. 2021
\end{thebibliography}

\end{document}
"""


@pytest.fixture
def sample_latex_with_subsections():
    """Sample LaTeX with subsections."""
    return r"""
\documentclass{article}
\begin{document}

\begin{abstract}
This is the abstract with sufficient content for proper parsing tests.
\end{abstract}

\section{Introduction}
This is the introduction content with adequate length for testing.

\section{Background}
\subsection{Previous Work}
Previous work in this area with sufficient detail for validation.

\subsection{Our Contributions}
Our approach is described here with proper explanation and context.

\section{Results}
We achieved the following results with detailed analysis included.

\section{Conclusion}
We conclude with these findings and future work directions.

\end{document}
"""


@pytest.fixture
def sample_latex_minimal():
    """Minimal LaTeX text for basic tests."""
    return r"""
\documentclass{article}
\begin{document}

\begin{abstract}
This is a minimal abstract with sufficient length for parsing validation tests.
\end{abstract}

\section{Introduction}
Introduction content here with enough text to meet minimum requirements.

\section{Conclusion}
Conclusion content with adequate length for proper section parsing tests.

\end{document}
"""