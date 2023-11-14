=======================
Installation & Updating
=======================

Installation
============

First, obtain Python3 (tested on versions≥3.7). If you haven't used python before, we recomend installing it through `anaconda3 <https://www.anaconda.com/products/individual>`_.

``MIN_ML`` can be installed using pip in one line. If you are using a terminal, enter:

.. code-block:: python

   pip install MIN_ML

If you are using Jupyter Notebooks (on Google Colab or Binder) or Jupyter Lab, you can also install it by entering the following code into a notebook cell (note the !):

.. code-block:: bash

   !pip install MIN_ML

You then need to import ``MIN_ML`` into the script you are running code in. In all the examples, we import ``MIN_ML`` as mm:

.. code-block:: python

   import MIN_ML as mm

This means any time you want to call a function from ``MIN_ML``, you do mm.function_name.



Updating
========

To upgrade to the most recent version of ``MIN_ML``, type the following into terminal:

.. code-block:: python

   pip install MIN_ML --upgrade

Or in your Jupyter environment:

.. code-block:: bash

   !pip install MIN_ML --upgrade


For maximum reproducability, you should state which version of ``MIN_ML`` you are using. If you have imported ``MIN_ML`` as mm, you can find this using:

.. code-block:: python

    mm.__version__