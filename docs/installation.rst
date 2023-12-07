=======================
Installation & Updating
=======================

Installation
============

First, obtain Python3 (tested on versions≥3.7). If you haven't used python before, we recomend installing it through `anaconda3 <https://www.anaconda.com/products/individual>`_.

``mineralML`` can be installed using pip in one line. If you are using a terminal, enter:

.. code-block:: python

   pip install mineralML

If you are using Jupyter Notebooks (on Google Colab or Binder) or Jupyter Lab, you can also install it by entering the following code into a notebook cell (note the !):

.. code-block:: bash

   !pip install mineralML

You then need to import ``mineralML`` into the script you are running code in. In all the examples, we import ``mineralML`` as mm:

.. code-block:: python

   import mineralML as mm

This means any time you want to call a function from ``mineralML``, you do mm.function_name.



Updating
========

To upgrade to the most recent version of ``mineralML``, type the following into terminal:

.. code-block:: python

   pip install mineralML --upgrade

Or in your Jupyter environment:

.. code-block:: bash

   !pip install mineralML --upgrade


For maximum reproducability, you should state which version of ``mineralML`` you are using. If you have imported ``mineralML`` as mm, you can find this using:

.. code-block:: python

    mm.__version__