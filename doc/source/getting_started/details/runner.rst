Runner
======

API reference: :doc:`/api/pe.runner`.

:py:class:`pe.runner.PE` manages the main **Private Evolution** algorithm by calling the other components discussed before. It has the following key methods:

* :py:meth:`pe.runner.PE.run`: Runs the **Private Evolution** algorithm.
* :py:meth:`pe.runner.PE.evaluate`: Evaluates the synthetic samples using the :py:class:`pe.callback.Callback` modules.
