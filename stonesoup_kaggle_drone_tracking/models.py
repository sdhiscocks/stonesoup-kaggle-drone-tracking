from typing import Union

import numpy as np
from stonesoup.base import Property
from stonesoup.functions import cart2pol
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVectors, StateVector


class CartesianToBearingRangeDiana(NonLinearGaussianMeasurement):
    translation_offset: StateVector = Property(
        default=None,
        doc="A 2x1 array specifying the origin offset in terms of :math:`x,y` coordinates.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * len(self.mapping))

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset
        xyz = np.array([state.state_vector[self.mapping[0], :] - self.translation_offset[0, 0],
                        state.state_vector[self.mapping[1], :] - self.translation_offset[1, 0],
                        [0] * state.state_vector.shape[1]
                        ])

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Covert to polar
        rho, phi = cart2pol(*xyz_rot[:2, :])
        phi %= np.pi
        bearings = [Bearing(i) for i in phi]
        return StateVectors([bearings, rho]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0)], [0.]]) + out
        return out
