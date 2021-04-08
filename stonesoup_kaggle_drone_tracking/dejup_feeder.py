# -*- coding: utf-8 -*-

from stonesoup.feeder.base import DetectionFeeder
from stonesoup.buffered_generator import BufferedGenerator


class Uniqueificator(DetectionFeeder):

    @BufferedGenerator.generator_method
    def data_gen(self):
        unique_values = set()
        for time, detections in self.reader:
            unique_detections = set()

            for detection in detections:

                sv = [float(val) for val in detection.state_vector]
                sv = tuple(sv)
                timestamp = detection.timestamp
                value = (sv, timestamp)
                if value in unique_values:
                    continue
                unique_values.add(value)
                unique_detections.add(detection)
            if not unique_detections:
                continue
            yield time, unique_detections
