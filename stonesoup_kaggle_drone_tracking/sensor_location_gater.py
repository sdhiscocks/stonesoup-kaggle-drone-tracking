# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.models.measurement.nonlinear import CombinedReversibleGaussianMeasurementModel
from stonesoup.gater.base import Gater
from stonesoup.base import Property


class SensorLocationGater(Gater):

    hypothesiser: DistanceHypothesiser = Property(
        doc='hypothesiser to use when far enough away from sensors')
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))
    min_distance_from_sensor: float = Property(default=80)

    def hypothesise(self, track, detections, timestamp, *args, **kwargs):
        for detection in detections:
            measurement_model = detection.measurement_model
            if isinstance(measurement_model, CombinedReversibleGaussianMeasurementModel):
                measurement_model = measurement_model.model_list[0]
            sensor_location = measurement_model.translation_offset[:, 0][:len(self.pos_mapping)]

            track_location = track.state_vector[self.pos_mapping, 0]
            difference = track_location - sensor_location
            euclidean_dist = np.sqrt(difference[0] ** 2 + difference[1] ** 2)

            if euclidean_dist <= self.min_distance_from_sensor:
                return self.hypothesiser.hypothesise(track, set(), timestamp)
        return self.hypothesiser.hypothesise(track, detections, timestamp)
