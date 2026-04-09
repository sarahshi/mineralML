==========
Change Log
==========


Version 0.0.3.11
================
Adapted all stoichiometry calculators to return the `Predict_Mineral`, `Prediction_Score`, `Prediction_Score_Sigma`, `Second_Predict_Mineral`, and `Second_Prediction_Score` from the `predict_class_prob` function (if present), in addition to the calculated components . This provides users with the predicted mineral class and associated confidence metrics alongside the stoichiometric calculations. 

Removed last pyrolite dependency — thank you Nick Swanson-Hysell for catching one remaining line in setup.py.


Version 0.0.3.10
================
The pyrolite dependency was removed given that it was only used for a single function in the documentation. The `plot_latent_space` function now includes a random seed for reproducibility of the latent space visualization, ensuring consistent results across runs.


Version 0.0.3.9
===============
Added ``renormalize`` argument to `prep_df` for optional renormalization of oxide compositions. Increased default iterations in `predict_class_prob`. Added new functions for processing microanalytical data from Cameca, Probe for EPMA, and AZtec formats. Added stoichiometry calculations for glasses. Added glass plotting to documentation.Updated `mapping.py` to return figure objects from `plot_ctf_phases`. Patched `stoichiometry.py` for pyrolite output compatibility. 

Updated ``rtd_environment.yml`` for pyrolite dependency. Added ``openpyxl`` to ``setup.py`` dependencies.

Removed deprecated `supervised.py` module and legacy ``parametermatrix`` files. Removed unused data files and updated ``.gitignore``.

Expanded unit test suite with new and extended coverage across ``test_hybrid.py``, ``test_mapping.py``, ``test_stoichiometry.py``, ``test_core.py``, ``test_confusion_matrix.py``, ``test_synthetic_minerals.py``, and ``test_microanalysis.py``.

Committed EDS and EBSD maps from Mount Hood and Tuolumne samples. Added EBSD-EDS mapping example notebook (``mineralML_mapping_ebsd.ipynb``) and updated documentation, examples, and README with mapping figure. Updated Google Colab implementation.


Version 0.0.3.8
===============
Refactored `predict_class_prob` to use centralized hybrid checkpoint loading. Removed duplicate manual model reconstruction from prediction. Aligned prediction and reconstruction paths to use the same loaded wrapper. Improved handling when oxide columns or `ZrO2` are absent. Avoided in-place modification of input DataFrames when computing `Total`. Clamped Monte Carlo variance to zero before square root to avoid invalid standard deviations.

Added `convert_fe_to_feot` utility for converting FeO, Fe2O3, and Fe2O3t columns to a single FeOt column.
`prep_df` now accepts ``convert_Fe`` to automatically convert iron columns to FeOt instead of raising an error. Added ``drop_empty_rows`` and ``min_oxide_count`` to optionally remove rows with fewer than *n* non-zero oxide columns. Added ``verbose`` flag that prints a summary of rows processed, dropped, and coerced.
`norm_data` now accepts ``verbose`` flag and prints the number of rows normalized.
`plot_latent_space` now resolves ``Oxide`` labels via a ``submineral_column`` fallback (e.g., Oxide → Magnetite → Spinel). Labels that are empirical groupings not present in training classes (Carbonate, SiO2_Polymorph, Zircon) are skipped with a descriptive warning.

Deprecated `load_minclass_nn`; use `load_mineral_classes` instead.
Deprecated `predict_class_prob_nnwr`; use `predict_class_prob` instead.
Deprecated `plot_z2_overlay`; use `plot_latent_space` instead.


Version 0.0.3.6 and 0.0.3.7
===========================
Clean up code docstrings and update RTD documentation.


Version 0.0.3.5
===============
Add map renormalization and additional epoxy filtering. When processing EDS oxide maps, pixel compositions can optionally be renormalized to 100 wt.% after loading, improving classification consistency across pixels with variable analytical totals. Additional filtering identifies and removes pixels of epoxy with very low totals, preventing misclassification of mounting material as mineral phases.


Version 0.0.3.4
===============
Fix totals issue to include ZrO2. Previously, ZrO2 was excluded from the oxide total calculation, causing analyses with reported ZrO2 (e.g., zircon) to be classified as carbonate given low totals. ZrO2 is now properly included in the total logic.


Version 0.0.3.3
===============
Add ``Predict_Probability_Sigma`` from forward passes. Prediction uncertainty is now estimated via Monte Carlo dropout forward passes through the neural network, returning a standard deviation (sigma) on prediction scores for each analysis. This provides a per-analysis measure of model confidence beyond the single-pass prediction score. Correct ``mineral_classes_nn_v0030`` to include ``Spinel_Group``. General repository cleanup removes deprecated code. 


Version 0.0.3.2
===============
Fix totals issue — compute from ``OXIDES`` rather than ``oxides``. The oxide total calculation was referencing a local or incorrectly scoped variable rather than the module-level constant ``OXIDES``, which defines the full set of input oxides used by the neural network.


Version 0.0.3.1
===============
Update NaN row handling in ``mm.prep_df_nn``. Update ``hybrid.py`` for the correct number of input oxides, ensuring consistency between the hybrid prediction model and the expected feature count.


Version 0.0.3.0
===============
Introduce new amphibole, garnet, and pyroxene training data to the model. These mineral groups have expanded compositions and coverage. SiO2 polymorphs (e.g., quartz, cristobalite, tridymite) and carbonates (e.g., calcite, dolomite) are separated out from the main neural network and are instead classified via empirical compositional rules, since their oxide signatures are sufficiently distinctive. Includes updates to the nearest-weighted-residual (``nnwr``) prediction pathway and associated unit tests.


Version 0.0.2.6
===============
Fix ``mm.plot_z2_overlay`` for overly granular labels. Previously, z2 overlay plots displayed specific submineral labels (e.g., individual pyroxene or feldspar species) rather than the broader group labels (Pyroxene, Feldspar). Labels are now mapped to their parent mineral groups for clearer visualization.


Version 0.0.2.5
===============
Added functionality in pyroxene component plotting (Wo-En-Fs ternary) to plot the full ternary diagram, rather than the limited quadrilateral. 


Version 0.0.2.4
===============
Added functionality in Harker plotting, providing new Harker-style variation diagrams for exploring oxide-oxide relationships across mineral data.


Version 0.0.2.3
===============
Added functionality in ``mapping.py``. Includes increased optionality for map generation, support for parsing ``.ctf`` files (EBSD crystal orientation data), and updates to z2 plotting to support pre-classification visualization. Update GEOROC dataset with citations.


Version 0.0.2.2
===============
Fix latent space plotting, add legend for all minerals. The latent space visualization now returns the full legend by default, showing all mineral groups.


Version 0.0.2.1
===============
Add latent space overlay functionality, enabling users to project their data onto the autoencoder's latent space for visual comparison against the training distribution. Also adds ``reset_index`` call in ``mm.prep_df_nn`` for cleaner DataFrame handling.


Version 0.0.2.0
===============
Built hybrid model functionality with a sequential, transfer learning neural network architecture. This release introduces a two-stage prediction model that refines initial classifications. Updated mapping functions and resolved compatibility issues with NumPy 2 and scikit-image. Updated ``supervised.py``, ``unsupervised.py``, and ``stoichiometry.py`` for the new architecture. Bug fixes and unit test updates.


Version 0.0.1.9
===============
Update mineralML calibration for specific amphiboles (tremolite and actinolite). Updates to unique mapping functions (``unique_mapping_nn``) and unit test scaler adjustments.


Version 0.0.1.8
===============
Fix small bits of `mm.plot_component_composite`.


Version 0.0.1.7
===============
Provide new mapping functions with `mm.plot_component_composite` to allow for internal compositional variability to be calculated and plotted for different mineral groups - most commonly examined olivine, clinopyroxene, orthopyroxene, and plagioclase feldspar, for now. 


Version 0.0.1.6
===============
Provide submineral in addition to primary prediction in `mm.predict_class_prob_nn`. 


Version 0.0.1.5
===============
Push updates to mapping to account for different input units (wt% vs. ox%), return all outputs by default. 

Modify feldspar line bounds to provide 50% tolerance for An and Or, increased from DHZ 0.05 to 0.075.


Version 0.0.1.4
===============
Removed .set_index() for SampleID in `mm.prep_nn_df` to mitigate confusion.


Version 0.0.1.3
===============
New version of mineralML with sodic pyroxene model added and refined. Follows the Morimoto 1988 and Harlow 1998 pyrxoene classifications. Updated mineral balancing functions to better sample solid solution space. Improved mapping functions. 


Version 0.0.1.2
===============
Optimize supervised.py prediction for speed. 


Version 0.0.1.1
===============
Remove vectorization to reduce RAM usage. 


Version 0.0.1
=============
New version of mineralML calibrated for 28 mineral [groups]: Amphibole, Apatite, Biotite, Calcite, Chlorite, Epidote, Feldspar (KFeldspar and Plagioclase with inbuilt classification), Garnet, Glass, Kalsilite, Leucite, Melilite, Muscovite, Nepheline, Olivine, Pyroxene (Clinopyroxene and Orthopyroxene), Quartz, Rhombohedral_Oxides (Hematite and Ilmenite), Rutile, Serpentine, Spinels (Magnetite and Spinel), Titanite, Tourmaline, Zircon. Adds the 11th input of P2O5 concentration for accurate classification of apatites (needed for comparison against calcite). These mineral groups are classified, and the subgroups or classes are empirically classified after the neural network takes a first pass. 

Presents stoichiometry calculators for all the aforementioned mineral [groups], along with classification. All components of interest are returned (e.g. XFo in Olivine, An Ab Or in Feldspar, En Fs Wo in Pyroxene, etc.) and classifications for solid solution minerals are presented, based on DHZ. 

Introduces a SolidSolutionGenerator for creating synthetic mineral data, to augment the training dataset where natural data are lacking. The synthetic mineral generator utilizes mineral formulas and stoichiometry as input and performs mixing between endmembers to create these synthetic data. It accounts for elemental site occupancy, ensures charge neutrality, and maintains oxygen basis to simulate natural minerals. This hopefully will facilitate future training of the neural networks, if mineral groups need introduction and insufficient natural data exist. 


Version 0.0.0.7
===============
Update prep_nn_df function to coerce strings in data and to return warnings. First release of mineralML calibrated for 17 mineral [groups]: Amphibole, Apatite, Biotite, Clinopyroxene, Garnet, Ilmenite, K-Feldspar, Magnetite, Muscovite, Olivine, Orthopyroxene, Plagioclase, Quartz, Rutile, Spinel, Tourmaline, Zircon. The neural network previously predicted for each of these individual minerals. This has been altered to classify mineral groups together. 


Version 0.0.0.6
===============
Update prep_nn_df function to remove mineral filter, create missing columns (whilst returning UserWarning).


Version 0.0.0.5
===============
Fix confusion_matrix_df and prep_nn_df functions.


Version 0.0.0.4
===============
Remove plt.show(). 


Version 0.0.0.3
===============
Update standard scaler for neural network and autoencoder. 


Version 0.0.0.2
===============
Test solution to LaTeX solution — remove plot saving. 


Version 0.0.0.1
===============
Removed imblearn dependency given timeout and necessity only during training. 


Version 0.0.0.0
===============
First version on PyPi. 

