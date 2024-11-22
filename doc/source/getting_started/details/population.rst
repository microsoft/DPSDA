Population
==========

API reference: :doc:`/api/pe.population`.

:py:class:`pe.population.population.Population` is responsible for generating the initial synthetic samples and the new synthetic samples for each **Private Evolution** iteration. It has the following key methods:

* :py:meth:`pe.population.population.Population.initial`: Generates the initial synthetic samples.
* :py:meth:`pe.population.population.Population.next`: Generates the synthetic samples for the next **Private Evolution** iteration.

Available Populations
---------------------

:py:class:`pe.population.pe_population.PEPopulation` is currently the only implementation of :py:class:`pe.population.population.Population`. It supports the key population algorthms from existing **Private Evolution** papers (https://github.com/fjxmlzn/private-evolution-papers).

