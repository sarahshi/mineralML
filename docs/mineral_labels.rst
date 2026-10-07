==============
Mineral Labels
==============

This page lists the labels ``mineralML`` returns, and how to compare them with your own mineral labels, e.g., published names in a ``Mineral`` column.


Labels returned by mineralML
============================

``mm.predict_class_prob`` returns each prediction at two levels: ``Predict_Mineral``, and a finer ``Submineral`` for feldspars, pyroxenes, and oxides. ``mm.plot_latent_space`` colors points by the neural network's own classes, which group some of these labels. The table lists the labels at each level. Labels not listed in the table, such as Olivine or Garnet, are the same at every level.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - ``Predict_Mineral``
     - ``Submineral``
     - Latent space class
   * - Alkali_Feldspar
     - Sanidine, Anorthoclase
     - Feldspar
   * - Plagioclase
     - Albite, Oligoclase, Andesine, Labradorite, Bytownite, Anorthite
     - Feldspar
   * - Feldspar_Miscibility_Gap
     - Feldspar_Miscibility_Gap
     - Feldspar
   * - Clinopyroxene
     - Augite, Diopside, Hedenbergite, Pigeonite, Wollastonite
     - Pyroxene
   * - Orthopyroxene
     - Enstatite, Ferrosilite
     - Pyroxene
   * - Na-Pyroxene
     - Jadeite, Aegirine, Omphacite, Aegirine-Augite, Ca-Mg-Fe Pyroxene
     - Pyroxene
   * - Oxide
     - Rhombohedral_Oxides
     - Rhombohedral_Oxides
   * - Oxide
     - Spinel_Group
     - Spinel_Group
   * - SiO2_Polymorph
     -
     - Not plotted (empirical rule)
   * - Carbonate
     -
     - Not plotted (empirical rule)
   * - Zircon
     -
     - Not plotted (empirical rule)

``Prediction_Score``, ``Prediction_Score_Sigma``, ``Second_Predict_Mineral``, and ``Second_Prediction_Score`` come from the neural network, so they refer to its classes (the latent space column), not to ``Predict_Mineral``. For an ``Oxide``, ``Prediction_Score`` is the score of its oxide group in ``Submineral``, not of oxides as a whole, and ``Second_Predict_Mineral`` is often the other oxide group. For example, a row could read ``Oxide`` (``Submineral`` Rhombohedral_Oxides) with a score of 0.60 and Spinel_Group second with 0.35: an oxide with a combined score of 0.95, but with less certainty about the group. Likewise, a Plagioclase or Clinopyroxene is scored as Feldspar or Pyroxene, and its second prediction uses those names, because the split within each group is calculated from stoichiometry rather than predicted by the neural network. Carbonate, SiO2_Polymorph, and Zircon are assigned by empirical rules and have no scores.


Comparing with your own labels
==============================

You will not need to alter your own labels before running predictions, as the ``Mineral`` column is not used to classify. To compare your labels with ``Predict_Mineral``, for example with scikit-learn's ``classification_report``, run both columns through ``mm.harmonize_labels`` first. Oxides are compared by group (Rhombohedral_Oxides or Spinel_Group), so pass the ``Submineral`` column as ``pred_submineral`` to replace ``Oxide`` predictions with their group:

.. code-block:: python

   from sklearn.metrics import classification_report

   published_harmonized, predict_mineral_harmonized = mm.harmonize_labels(
       df_pred['Mineral'], df_pred['Predict_Mineral'], pred_submineral=df_pred['Submineral'])
   print(classification_report(published_harmonized, predict_mineral_harmonized, zero_division=0))

``published_harmonized`` is your ``Mineral`` column and ``predict_mineral_harmonized`` is ``Predict_Mineral``, both mapped onto the labels in the table below, e.g., Hematite becomes Rhombohedral_Oxides and an ``Oxide`` prediction with ``Submineral`` Spinel_Group becomes Spinel_Group. Na-Pyroxene is compared as Clinopyroxene.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Compared as
     - Other names mapped by ``harmonize_labels``
   * - Alkali_Feldspar
     - Sanidine, Anorthoclase, K-Feldspar, Orthoclase, Microcline, Adularia, Perthite
   * - Plagioclase
     - Albite, Oligoclase, Andesine, Labradorite, Bytownite, Anorthite
   * - Clinopyroxene
     - Augite, Diopside, Hedenbergite, Pigeonite, Wollastonite, Na-Pyroxene, Jadeite, Aegirine, Aegirine-Augite, Omphacite, Ca-Mg-Fe Pyroxene, Salite, Cr-Diopside, Titanaugite, Ferroaugite, Ferrohedenbergite
   * - Orthopyroxene
     - Enstatite, Ferrosilite, Hypersthene, Ferrohypersthene
   * - Rhombohedral_Oxides
     - Hematite, Ilmenite, Titanohematite, Hemoilmenite, Ilmenite-Hematite
   * - Spinel_Group
     - Magnetite, Al-Magnetite, Hercynite, Pleonaste, Ferrian-Pleonaste, Ferrian-Picotite, Magnesioferrite, Titanomagnetite, Ti-Magnetite, Ulvospinel, Chromite, Fe-Chromite, Picotite, and any name containing "spinel"
   * - Oxide
     - Fe-Ti Oxide (parent label; see below)
   * - Amphibole
     - Tremolite, Actinolite, Ferroactinolite, Magnesiohornblende, Ferrohornblende, Tschermakite, Ferrotschermakite, Hornblende, Pargasite, Edenite, Hastingsite, Magnesiohastingsite, Kaersutite, Richterite, Ferrorichterite, Katophorite, Arfvedsonite, Riebeckite, Glaucophane, Cummingtonite, Anthophyllite
   * - Apatite
     - Fluorapatite, Chlorapatite, Hydroxylapatite
   * - Biotite
     - Phlogopite, Annite, Siderophyllite, Eastonite
   * - Garnet
     - Almandine, Pyrope, Spessartine, Grossular, Andradite
   * - Muscovite
     - Phengite, Sericite
   * - Olivine
     - Forsterite, Fayalite
   * - Titanite
     - Sphene
   * - SiO2_Polymorph
     - SiO2, Quartz, Tridymite, Cristobalite, Coesite, Stishovite, Chalcedony, Agate
   * - Carbonate
     - Calcite, Aragonite, Dolomite, Ankerite, Magnesite, Siderite, Rhodochrosite, Strontianite, Witherite

Labels not in the table, such as Chlorite or Zircon, are matched by their own name. ``harmonize_labels`` ignores case, spaces, and punctuation, so "alkali feldspar" and GEOROC's "(AL)KALIFELDSPAR" both match Alkali_Feldspar, and "TITANO-MAGNETITE" matches Titanomagnetite. The names in the table cover the mineral names in GEOROC for the minerals mineralML classifies, except "Mica", which could be Biotite or Muscovite.

If either column uses "Feldspar", "Pyroxene", or "Oxide", the children (Alkali_Feldspar and Plagioclase, Clinopyroxene and Orthopyroxene, or Rhombohedral_Oxides and Spinel_Group) are merged into that parent in both columns. For example, if ``pred_submineral`` is not passed, ``Oxide`` predictions merge both oxide groups into Oxide. Names that are not recognized are returned unchanged with a warning; rename these to a ``Predict_Mineral`` label before comparing. ``mm.confusion_matrix_df`` takes the same arguments and applies the same mapping, and drops rows with unrecognized labels, such as Feldspar_Miscibility_Gap, with a warning.
