import csv
import os
from typing import Tuple

import dateutil
import numpy as np
from pymap3d import geodetic2enu
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.functions import cart2sphere, sphere2cart
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange, \
    CartesianToBearingRange, Cartesian2DToBearing, CombinedReversibleGaussianMeasurementModel
from stonesoup.reader import DetectionReader, GroundTruthReader
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.state import StateVector

from .models import CartesianToBearingRangeDiana

lat0, lon0, alt0 = 51.519137, 5.857951, 0


class ARCUSReader(DetectionReader):
    folder: str = Property(doc="Folder where scenario file is.")
    ndim_state: int = Property(default=6)
    pos_mapping: Tuple[int, int, int] = Property(default=(0, 2, 4))
    vel_mapping: Tuple[int, int, int] = Property(default=(1, 3, 5))
    pos_noise_diag: Tuple[float, float, float] = Property(
        default=((np.pi/4) ** 2, np.radians(1) ** 2, 25 ** 2))
    vel_noise_diag: Tuple[float, float, float] = Property(default=(1, 1, 1))
    min_reflection: float = Property(default=-25)
    max_reflection: float = Property(default=-10)

    lat, lon, alt = 51.52147, 5.87056833, 31

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        position_model = CartesianToElevationBearingRange(
            self.ndim_state, self.pos_mapping, np.diag(self.pos_noise_diag),
            translation_offset=StateVector([*geodetic2enu(self.lat, self.lon, self.alt,
                                                          lat0, lon0, alt0)]))

        velocity_model = LinearGaussian(
            self.ndim_state, self.vel_mapping, np.diag(self.vel_noise_diag))

        self.model = CombinedReversibleGaussianMeasurementModel([position_model, velocity_model])

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(os.path.join(self.folder, 'ARCUS_scenario.csv'), newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['ArcusTracksTrack_Timestamp']:
                    continue

                timestamp = dateutil.parser.parse(row['ArcusTracksTrack_Timestamp'], ignoretz=True)
                lat = float(row['ArcusTracksTrackPosition_Latitude'])
                lon = float(row['ArcusTracksTrackPosition_Longitude'])
                alt = float(row['ArcusTracksTrackPosition_Altitude'])
                azimuth = np.radians(90 - float(row['ArcusTracksTrackVelocity_Azimuth']))
                elevation = np.radians(float(row['ArcusTracksTrackVelocity_Elevation']))
                speed = float(row['ArcusTracksTrackVelocity_Speed'])

                metadata = {
                    'classification': row['ArcusTracksTrack_Classification'],
                    'sensor': 'Arcus',
                    'reflection': float(row['ArcusTracksTrack_Reflection']),
                    'score': float(row['ArcusTracksTrack_Score']),
                }

                if not self.min_reflection < metadata['reflection'] < self.max_reflection:
                    continue

                easting, northing, *_ = geodetic2enu(lat, lon, alt, self.lat, self.lon, self.alt)
                rho, phi, theta = cart2sphere(easting, northing, alt)
                dx, dy, dz = sphere2cart(speed, azimuth, elevation)

                yield timestamp, {Detection(
                    [Elevation(theta), Bearing(phi), rho, dx, dy, dz], timestamp=timestamp,
                    metadata=metadata, measurement_model=self.model)}


class ALVIRAReader(DetectionReader):
    folder: str = Property(doc="Folder where scenario file is.")
    ndim_state: int = Property(default=6)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))
    vel_mapping: Tuple[int, int] = Property(default=(1, 3))
    pos_noise_diag: Tuple[float, float] = Property(
        default=(np.radians(1) ** 2, 25 ** 2))
    vel_noise_diag: Tuple[float, float] = Property(default=(1, 1))
    min_reflection: float = Property(default=-np.inf)
    max_reflection: float = Property(default=35)

    lat, lon, alt = 51.52126391, 5.85862734, 31

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        position_model = CartesianToBearingRange(
            self.ndim_state, self.pos_mapping, np.diag(self.pos_noise_diag),
            translation_offset=StateVector([*geodetic2enu(self.lat, self.lon, self.alt,
                                                          lat0, lon0, alt0)]))
        velocity_model = LinearGaussian(
            self.ndim_state, self.vel_mapping, np.diag(self.vel_noise_diag))

        self.model = CombinedReversibleGaussianMeasurementModel([position_model, velocity_model])

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(os.path.join(self.folder, 'ALVIRA_scenario.csv'), newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['AlviraTracksTrack_Timestamp']:
                    continue

                timestamp = dateutil.parser.parse(row['AlviraTracksTrack_Timestamp'], ignoretz=True)
                lat = float(row['AlviraTracksTrackPosition_Latitude'])
                lon = float(row['AlviraTracksTrackPosition_Longitude'])
                alt = float(row['AlviraTracksTrackPosition_Altitude'])
                azimuth = np.radians(90 - float(row['AlviraTracksTrackVelocity_Azimuth']))
                elevation = np.radians(float(row['AlviraTracksTrackVelocity_Elevation']))
                speed = float(row['AlviraTracksTrackVelocity_Speed'])

                metadata = {
                    'classification': row['AlviraTracksTrack_Classification'],
                    'sensor': 'Alvira',
                    'reflection': float(row['AlviraTracksTrack_Reflection']),
                    'score': float(row['AlviraTracksTrack_Score']),
                }

                if not self.min_reflection < metadata['reflection'] < self.max_reflection:
                    continue

                easting, northing, *_ = geodetic2enu(lat, lon, alt, self.lat, self.lon, self.alt)
                rho, phi, _ = cart2sphere(easting, northing, alt)
                dx, dy, dz = sphere2cart(speed, azimuth, elevation)

                yield timestamp, {Detection(
                    [Bearing(phi), rho, dx, dy], timestamp=timestamp,
                    metadata=metadata, measurement_model=self.model)}


class DroneGroundTruthReader(GroundTruthReader):
    folder: str = Property(doc="")
    groundtruth_regex: str = Property(doc="RegEx to get all Ground truth Files")

    @staticmethod
    def single_ground_truth_reader(filepath, isset=True):
        truth = GroundTruthPath()
        with open(filepath, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                alt = float(row['altitude(m)'])
                time = dateutil.parser.parse(row['datetime(utc)'])
                if row['Planename'] != "":
                    planename = row['Planename']
                x, y, z = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
                truth.append(GroundTruthState(
                    [x, 0, y, 0, z, 0],
                    timestamp=time,
                    metadata={"id": planename}))
            if isset:
                truth = {truth}
        return truth

    @classmethod
    def multiple_ground_truth_reader(cls, filepaths):
        truths = set()
        for filepath in filepaths:
            truths.add(cls.single_ground_truth_reader(filepath, isset=False))
        return truths

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        gt_files = [
            name for name in os.listdir(self.folder) if name.startswith(self.groundtruth_regex)]
        truths = self.multiple_ground_truth_reader([
            os.path.join(self.folder, gt_file) for gt_file in gt_files])
        yield None, truths


class DIANAReader(DetectionReader):
    folder: str = Property(doc="Folder where scenario file is.")
    ndim_state: int = Property(default=6)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))
    pos_noise_diag: Tuple[float, float] = Property(default=(np.radians(3) ** 2, 1000))
    filter_controllers: bool = Property(default=True)

    lat, lon, alt = 51.519137, 5.857951, 31

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CartesianToBearingRangeDiana(
            self.ndim_state, self.pos_mapping, np.diag(self.pos_noise_diag),
            translation_offset=StateVector([*geodetic2enu(self.lat, self.lon, self.alt,
                                                          lat0, lon0, alt0)]),
        )

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(os.path.join(self.folder, 'DIANA_scenario.csv'), newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['DianaTargetsTargetSignal_bearing_deg']:
                    continue
                timestamp = dateutil.parser.parse(row['datetime(utc)'])
                phi = np.pi/2 - np.deg2rad(float(row['DianaTargetsTargetSignal_bearing_deg']))
                rho = float(row['DianaTargetsTargetSignal_range_m'])

                identity = row['DianaTargetsTargetClassification_model']
                if identity == 'Parrot-ANAFI':
                    identity = "Parrot"
                elif identity == "DJI-MAVIC-PRO-PLATINUM":
                    identity = "aladrian-MAVIC PRO"
                elif identity == 'DJI-MAVIC-2-PRO':
                    identity = "djiuser_97p9AXasssb6-Mavic2"
                elif "DJI-Phantom-4" in identity:
                    identity = "kcdgc-P4 Professional V2.0"
                metadata = {
                    'identity': identity,
                    'type': row['DianaTargetsTargetClassification_type'],
                    'sensor': 'Diana',
                    'band': row['DianaTargets_band'],
                    'targetID': row['DianaTarget_ID'],
                    'classification_score': float(row['DianaTargetsTargetClassification_score'])
                }

                if self.filter_controllers and metadata['type'].lower() == 'controller':
                    continue

                yield timestamp, {Detection([Bearing(phi), rho], timestamp, self.model,
                                            metadata=metadata)}


class VENUSReader(DetectionReader):
    folder: str = Property(doc="Folder where scenario file is.")
    ndim_state: int = Property(default=6)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))
    pos_noise_diag: Tuple[float] = Property(default=(np.radians(3) ** 2, ))

    lat, lon, alt = 51.5192716, 5.8579155, 31
    orientation = np.deg2rad(-20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Cartesian2DToBearing(
            self.ndim_state, self.pos_mapping, np.diag(np.atleast_1d(self.pos_noise_diag)),
            rotation_offset=StateVector([0, 0, self.orientation]),
            translation_offset=StateVector([*geodetic2enu(self.lat, self.lon, self.alt,
                                                          lat0, lon0, alt0)])
            )

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(os.path.join(self.folder, 'VENUS_scenario.csv'), newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                timestamp = dateutil.parser.parse(row['datetime(utc)'])
                identity = row['VenusTrigger_VenusName']
                if identity == '':
                    identity = None
                elif "Mavic" in identity:
                    if "Pro" in identity:
                        identity = "aladrian-MAVIC PRO"
                    else:
                        identity = "djiuser_97p9AXasssb6-Mavic2"
                elif "Phantom" in identity:
                    identity = "kcdgc-P4 Professional V2.0"

                bearstr = row['VenusTrigger_Azimuth']
                if bearstr:
                    bearing = Bearing(np.radians(float(row['VenusTrigger_Azimuth'])))
                else:
                    # No bearing
                    continue

                metadata = {
                    'identity': identity,
                    'sensor': 'Venus',
                    'frequency': row['VenusTrigger_Frequency'],
                    'radioID': row["VenusTrigger_RadioId"],
                }
                yield timestamp, {Detection([bearing], timestamp, self.model, metadata=metadata)}
