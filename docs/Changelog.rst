==========
Change Log
==========


Version 0.0.1.9
===============
Update mineralML calibration for specific amphiboles (tremolite and actinolite).


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

