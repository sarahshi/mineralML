=========================
Introduction and Citation
=========================

Welcome to ``mineralML``: An Open-Source Machine Learning Package for Probabilistically Classifying Minerals.˘

The development of this tool is currently in progress, with submission planned in the near future. Please make sure you cite this tool if you use it. Software development takes time and and academia does not always recognize the effort taken, but it does recognize citations. 

The open-source nature of the tool allows for continuous development. We welcome the submission of high quality mineral analyses that can be added to the training dataset. You can email `sarahshi@berkeley.edu <mailto:sarahshi@berkeley.edu>`_ or post an enhancement request or report of a bug on the issue page of the `mineralML GitHub repository <https://github.com/SarahShi/mineralML>`_. 


=============
Collaborators
=============

These folks have been fundamental to the development of ``mineralML``: 

- `Sarah Shi <https://github.com/sarahshi>`_ (University of California, Berkeley) 
- `Penny Wieser <https://github.com/pennywieser>`_ (University of California, Berkeley)
- `Norbert Toth <https://github.com/norberttoth398>`_ (University of Cambridge)
- `Paula Antoshechkina <https://github.com/magmasource>`_ (California Institute of Technology)
- `Kerstin Lehnert <https://lamont.columbia.edu/directory/kerstin-lehnert>`_ (LDEO)


========
Minerals
========

``mineralML`` is trained on a curated dataset of 86k analyses of 17 minerals. The machine learning models classify these minerals:

- Amphibole
- Apatite
- Biotite
- Clinopyroxene
- Garnet
- Ilmenite
- K-Feldspar
- Magnetite
- Muscovite
- Olivine
- Orthopyroxene
- Plagioclase
- Quartz
- Rutile
- Spinel
- Tourmaline
- Zircon


=========
Chemistry
=========

``mineralML`` requires inputs of mineral chemistry, in the form of oxides. The machine learning models are trained on 10 oxides: 

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


=====
Units
=====

``mineralML`` performs all calculations using mineral compositions in oxide weight percentages.
