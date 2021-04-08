from stonesoup.base import Property
from stonesoup.initiator import Initiator


class AlviraInitiator(Initiator):

    initiator: Initiator = Property()

    def initiate(self, detections, timestamp, **kwargs):
        for detection in detections:
            if detection.metadata['sensor'] == 'Alvira':
                return self.initiator.initiate(detections, timestamp, **kwargs)
        return self.initiator.initiate(set(), timestamp, **kwargs)


class AlviraAndArcusInitiator(Initiator):

    initiator: Initiator = Property()

    def initiate(self, detections, timestamp, **kwargs):
        for detection in detections:
            if detection.metadata['sensor'] in ('Alvira', 'Arcus') \
                    and detection.metadata['classification'] in ('DRONE', 'SUSPECTED_DRONE'):
                return self.initiator.initiate(detections, timestamp, **kwargs)
        return self.initiator.initiate(set(), timestamp, **kwargs)
