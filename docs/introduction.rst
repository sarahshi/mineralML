=========================
Introduction and Citation
=========================

Welcome to ``mineralML``: An Open-Source Machine Learning Package for Probabilistically Classifying Minerals.˘

The development of this tool is continually in progress. The manuscript is in review at Geochemistry, Geophysics, and Geosystems with the preprint shortly posted on Earth ArXiv. Please make sure you cite this tool if you use it. Software development takes time and and academia does not always recognize the effort taken, but it does recognize citations. 

``mineralML`` is currently in review, with the preprint on Earth ArXiv :cite:p:`Shietal2026`. Please refer to the manuscript for a more detailed description of the development and validation of the method. If you use this package in your work, please cite: 

.. code-block:: console

   Shi, S., Wieser, P., Gordon, C., Toth, N., Antoshechkina, P., Gleeson, M., & Lehnert, K. (2026). mineralML: Leveraging Machine Learning for Probabilistic Mineral Classification. EarthArXiv eprints, X53J2M. doi: 10.31223/X53J2M.

.. code-block:: text
        
    @article{Shietal2026,
    title = {mineralML: Leveraging Machine Learning for Probabilistic Mineral Classification},
    url = {http://dx.doi.org/10.31223/X53J2M},
    DOI = {10.31223/x53j2m},
    publisher = {California Digital Library (CDL)},
    author = {Shi, Sarah and Wieser, Penny and Gordon, Charlotte and Toth, Norbert and Antoshechkina, Paula and Gleeson,  Matthew and Lehnert, Kerstin},
    year = {2026},
    month = mar 
    }

The open-source nature of the tool allows for continuous development. We welcome the submission of high quality mineral analyses that can be added to the training dataset. You can email `sarahshi@berkeley.edu <mailto:sarahshi@berkeley.edu>`_ or post an enhancement request or report of a bug on the issue page of the `mineralML GitHub repository <https://github.com/SarahShi/mineralML>`_. 


=============
Collaborators
=============

These folks have been fundamental to the development of ``mineralML``: 

- `Sarah Shi <https://github.com/sarahshi>`_ (University of California, Berkeley, ex-LDEO)
- `Penny Wieser <https://github.com/pennywieser>`_ (University of California, Berkeley)
- `Charlotte Gordon <https://www.researchgate.net/profile/Charlotte-Gordon-5>`_ (University of California, Berkeley)
- `Norbert Toth <https://github.com/norberttoth398>`_ (University of Cambridge)
- `Paula Antoshechkina <https://github.com/magmasource>`_ (California Institute of Technology)
- `Matthew Gleeson <https://github.com/gleesonm1>`_ (University of California, Berkeley)
- `Kerstin Lehnert <https://lamont.columbia.edu/directory/kerstin-lehnert>`_ (LDEO)


========
Minerals
========

``mineralML`` is trained on a curated dataset of 128k analyses of 23 mineral groups/glass. The machine learning models classify these minerals:

- Amphibole
- Apatite
- Biotite
- Calcite
- Chlorite
- Epidote
- Feldspar (Alkali_Feldspar and Plagioclase)
- Garnet
- Glass
- Kalsilite
- Leucite
- Melilite
- Muscovite
- Nepheline
- Olivine
- Oxide (Rhombohedral_Oxides with Hematite-Ilmenite, and Spinel_Group with Magnetite-Spinel)
- Pyroxene (Clinopyroxene, Orthopyroxene, Na-Pyroxene)
- SiO2-Polymorphs (Quartz, Coesite, Stishovite, Tridymite, Cristobalite)
- Rutile
- Serpentine
- Titanite
- Tourmaline
- Zircon


=========
Chemistry
=========

``mineralML`` requires inputs of mineral chemistry, in the form of oxides. The machine learning model is trained on 11 oxides: 

*  SiO₂
*  TiO₂
*  Al₂O₃
*  FeOₜ
*  MnO
*  MgO 
*  CaO 
*  Na₂O
*  K₂O
*  Cr₂O₃
*  P₂O₅

Zircon is classified empirically with ZrO₂. 

=====
Units
=====

``mineralML`` performs all calculations using mineral compositions in oxide weight percentages. If needed, ``mm.oxide_to_element`` or ``mm.element_to_oxide`` may be helpful for converting your data into a usable format.