Population
==========

API reference: :doc:`/api/pe.population`.

:py:class:`pe.population.Population` is responsible for generating the initial synthetic samples and the new synthetic samples for each **Private Evolution** iteration. It has the following key methods:

* :py:meth:`pe.population.Population.initial`: Generates the initial synthetic samples.
* :py:meth:`pe.population.Population.next`: Generates the synthetic samples for the next **Private Evolution** iteration.

Available Populations
---------------------

* :py:class:`pe.population.PEPopulation` is the default implementation of :py:class:`pe.population.Population`. It supports the key population algorithms from existing **Private Evolution** papers (https://github.com/fjxmlzn/private-evolution-papers).
* :py:class:`pe.population.CompositePopulation`, proposed in `Tab-PE <https://arxiv.org/abs/2606.08259>`__, can be used when different **Private Evolution** iterations should use different population algorithms or settings. It takes a list of :py:class:`pe.population.Population` objects, uses the first population to generate the initial synthetic samples, and then uses the population at each iteration index to generate the next synthetic samples. This is useful for combining multiple population strategies within a single **Private Evolution** run.
