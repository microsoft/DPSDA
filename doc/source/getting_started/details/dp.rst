DP
===

API reference: :doc:`/api/pe.dp`.

:py:class:`pe.dp.DP` is responsible for implementing the differential privacy mechanism. It has the following key methods:

* :py:meth:`pe.dp.DP.set_epsilon_and_delta`: Set the privacy budget for the differential privacy mechanism.
* :py:meth:`pe.dp.DP.add_noise`: Add noise to the histogram values to achieve differential privacy.

Available Differential Privacy Mechanisms
-----------------------------------------

Currently, the following differential privacy mechanisms are implemented:

* :py:class:`pe.dp.Gaussian`: The Gaussian mechanism, which adds Gaussian noise to the histogram values.