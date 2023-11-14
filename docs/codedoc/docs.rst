PyIRoGlass Documentation
========================


Data Imports 
============

.. autofunction:: PyIRoGlass.Load_SampleCSV

.. autofunction:: PyIRoGlass.Load_PC

.. autofunction:: PyIRoGlass.Load_Wavenumber

.. autofunction:: PyIRoGlass.Load_ChemistryThickness


Building-blocks functions for fitting baselines and peaks
=========================================================

.. autofunction:: PyIRoGlass.Gauss

.. autofunction:: PyIRoGlass.Linear

.. autofunction:: PyIRoGlass.Carbonate

.. autofunction:: PyIRoGlass.als_baseline

.. autoclass:: PyIRoGlass.WhittakerSmoother
   :members:

.. autofunction:: PyIRoGlass.NearIR_Process

.. autofunction:: PyIRoGlass.MidIR_Process

.. autofunction:: PyIRoGlass.MCMC

.. autofunction:: PyIRoGlass.Run_All_Spectra


Functions for calculating concentrations
========================================


.. autofunction:: PyIRoGlass.Beer_Lambert

.. autofunction:: PyIRoGlass.Beer_Lambert_Error

.. autofunction:: PyIRoGlass.Concentration_Output



Functions for calculating density, molar absorptivity
=====================================================


.. autofunction:: PyIRoGlass.Density_Calculation

.. autofunction:: PyIRoGlass.Epsilon_Calculation


Functions for plotting MCMC results
===================================


.. autofunction:: PyIRoGlass.modelfit

.. autofunction:: PyIRoGlass.trace


Functions for determining thickness from reflectance FTIR spectra
=================================================================


.. autofunction:: PyIRoGlass.PeakID

.. autofunction:: PyIRoGlass.Thickness_Calc

.. autofunction:: PyIRoGlass.Thickness_Process

.. autofunction:: PyIRoGlass.Reflectance_Index


Functions for molar absorptivity inversions
===========================================


.. autofunction:: PyIRoGlass.Inversion

.. autofunction:: PyIRoGlass.Least_Squares

.. autofunction:: PyIRoGlass.Calculate_Calibration_Error

.. autofunction:: PyIRoGlass.Calculate_Epsilon

.. autofunction:: PyIRoGlass.Calculate_SEE

.. autofunction:: PyIRoGlass.Calculate_R2

.. autofunction:: PyIRoGlass.Calculate_RMSE

.. autofunction:: PyIRoGlass.Inversion_Fit_Errors
