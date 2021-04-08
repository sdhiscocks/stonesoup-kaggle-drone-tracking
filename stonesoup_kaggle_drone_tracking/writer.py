import csv
import datetime
from collections import Counter
from math import ceil
from typing import Sequence

from pymap3d import enu2geodetic
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.track import Track
from stonesoup.types.update import Update

from . import lat0, lon0, alt0


def generate_timestamps(start_time, end_time):
    total_seconds = (end_time - start_time).total_seconds()
    return [start_time + datetime.timedelta(seconds=n) for n in range(ceil(total_seconds))]


scenario_timestamps = {
    'Scenario_1_2_a': generate_timestamps(datetime.datetime(2020, 9, 29, 13, 0, 8),
                                          datetime.datetime(2020, 9, 29, 13, 9, 17)),
    'Scenario_1_4_a': generate_timestamps(datetime.datetime(2020, 9, 29, 12, 29, 0),
                                          datetime.datetime(2020, 9, 29, 12, 43, 6)),
    'Scenario_3_1': generate_timestamps(datetime.datetime(2020, 9, 30, 12, 0, 40),
                                        datetime.datetime(2020, 9, 30, 12, 18, 0)),
    'Scenario_3_1_a': generate_timestamps(datetime.datetime(2020, 9, 30, 12, 27, 38),
                                          datetime.datetime(2020, 9, 30, 12, 37, 54)),
    'Scenario_3_3': generate_timestamps(datetime.datetime(2020, 9, 30, 12, 44, 16),
                                        datetime.datetime(2020, 9, 30, 13, 1, 12)),
    'Scenario_3_4_text': generate_timestamps(datetime.datetime(2020, 9, 30, 13, 4, 6),
                                             datetime.datetime(2020, 9, 30, 13, 14, 3)),
    'Scenario_Parrot_d': generate_timestamps(datetime.datetime(2020, 9, 29, 14, 12, 18),
                                             datetime.datetime(2020, 9, 29, 14, 25, 12)),
}


def get_identification(track):
    counts = Counter()
    for data in track:
        if isinstance(data, Update):
            identity = data.hypothesis.measurement.metadata.get("identity")
            counts[identity] += 1
    del counts[None]
    delete = []

    # Protocol rather than drone type
    if 'DJI OcuSync' in counts:
        for key in counts:
            if 'mavic' in key.lower() or 'p4 pro' in key.lower():
                counts[key] += 0.25 * counts['DJI OcuSync']
        delete.append('DJI OcuSync')

    for key in counts.keys():
        if key in {"Unknown"}:
            delete.append(key)

    for key in delete:
        del counts[key]

    if counts.most_common(1):
        return counts.most_common(1)[0][0]
    return ''


def get_classification(track):
    counts = Counter()
    for data in track:
        if isinstance(data, Update):
            identity = data.hypothesis.measurement.metadata.get("classification")
            counts[identity] += 1
    del counts[None]
    delete = []
    for key in counts.keys():
        if key in {"UNKNOWN", "OTHER"}:
            delete.append(key)

    for key in delete:
        del counts[key]

    if counts.most_common(1):
        return counts.most_common(1)[0][0]


def interpolate_tracks(tracks: set, timestamps: Sequence, min_track_points: int = 1):
    timestamps = sorted(timestamps)  # sort timestamps in case out of order
    tracks_interp = set()

    for track in tracks:

        new_track = Track()
        state_iter = iter(track)
        aft = bef = next(state_iter)
        updates = set()
        if isinstance(aft, Update):
            updates.add(aft)

        for timestamp in timestamps:
            if timestamp < track[0].timestamp:
                continue
            elif timestamp > track[-1].timestamp:
                break

            while aft.timestamp < timestamp:
                bef = aft
                while bef.timestamp == aft.timestamp:
                    aft = next(state_iter)
                    if isinstance(aft, Update) and aft.timestamp <= timestamp:
                        updates.add(aft)

            if aft.timestamp == timestamp:
                sv = aft.state_vector
            else:
                bef_sv = bef.state_vector
                aft_sv = aft.state_vector

                frac = (timestamp - bef.timestamp) / (aft.timestamp - bef.timestamp)

                sv = bef_sv + ((aft_sv - bef_sv) * frac)

            index = track.index(bef)
            metadata = track.metadatas[index].copy()

            sensors = {update.hypothesis.measurement.metadata.get('sensor')
                       for update in updates}
            metadata['sensors'] = sensors

            # Using ground truth state here, just to use meta data
            new_state = GroundTruthState(sv, timestamp=timestamp, metadata=metadata)
            new_track.append(new_state)

            updates = set()
            if aft.timestamp > timestamp and isinstance(aft, Update):
                updates.add(aft)

        new_track.metadata['mean_classification'] = get_classification(track)
        new_track.metadata['mean_identification'] = get_identification(track)
        if len(new_track) >= min_track_points:
            tracks_interp.add(new_track)
    return tracks_interp


def write_output_csv(csvfile, tracks, timestamps, mapping=(0, 2, 4), min_track_points=1):

    interp_tracks = interpolate_tracks(tracks, timestamps, min_track_points)

    fieldnames = ["TrackDateTimeUTC",
                  "TrackID",
                  "TrackPositionLatitude",
                  "TrackPositionLongitude",
                  "TrackPositionAltitude",
                  "TrackClassification",
                  "TrackIdentification",
                  "TrackSource"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for timestamp in timestamps:

        t_tracks = {
            track
            for track in interp_tracks
            if track[0].timestamp < timestamp < track[-1].timestamp
        }

        for track in t_tracks:

            track_state = track[timestamp]

            sv = track_state.state_vector

            e, n, u = sv[mapping, 0]
            lat, lon, alt = enu2geodetic(e, n, u, lat0, lon0, alt0)

            # Add by interpolation currently
            classification = track.metadata['mean_classification']
            identification = track.metadata['mean_identification']

            sensors = track_state.metadata.get('sensors')
            sensors = ';'.join(sensors)

            values = [
                timestamp, track.id, lat, lon, alt, classification, identification, sensors]
            row = {key: value for key, value in zip(fieldnames, values)}
            writer.writerow(row)
