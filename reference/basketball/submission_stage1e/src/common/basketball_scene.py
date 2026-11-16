"""CPU pixel evidence, independent of predicted person roles.

Floor/marking cues are geometric/color measurements, not a semantic court model.
No track ID, source timestamp, manual label or camera-specific polygon is used.
"""
import cv2
import numpy as np


class SceneEvidence:
    def __init__(self, config):
        self.c = config
        self.previous = None
        self.pending = False
        self.last_frame = None

    def observe(self, frame, frame_id, observations=()):
        small = cv2.resize(cv2.resize(frame, (640,360), interpolation=cv2.INTER_AREA), (320,180), interpolation=cv2.INTER_AREA)
        # Exclude broadcast scorebar/logos from scene-change measurements.
        image = small[18:153, 16:304]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1], None,
                           [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        distance = residual = 0.; response = 1.; discontinuity = False
        gap = self.last_frame is not None and frame_id - self.last_frame > self.c['pixel_max_gap']
        if self.previous is not None:
            prev, ph = self.previous
            distance = float(cv2.compareHist(ph, hist, cv2.HISTCMP_BHATTACHARYYA))
            shift, response = cv2.phaseCorrelate(prev, gray)
            transform = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            warped = cv2.warpAffine(prev, transform, (gray.shape[1], gray.shape[0]))
            valid = cv2.warpAffine(np.ones_like(prev), transform,
                                   (gray.shape[1], gray.shape[0])) > .99
            residual = float(np.mean(np.abs(warped[valid] - gray[valid]))) if valid.mean() > .5 else 1.
            # Phase correlation is undefined on flat images. An unchanged
            # frame remains unchanged even when the estimated shift is bogus.
            residual = min(residual, float(np.mean(np.abs(prev-gray))))
            discontinuity = (distance >= self.c['cut_hist_distance'] and
                             residual >= self.c['cut_aligned_residual'])
        # Two observations: isolate suspected cuts immediately; confirm only
        # when the new view persists. A sparse sample gap is a reset, NOT proof
        # that a broadcast cut occurred between those samples.
        settled = (distance <= self.c['cut_settle_hist'] and residual <= self.c['cut_settle_residual'])
        confirmed = bool(self.pending and settled and not gap)
        self.pending = bool(discontinuity or (self.pending and not settled))
        quarantine = bool(self.pending or gap)
        self.previous = (gray, hist); self.last_frame = int(frame_id)

        self.result = dict(scene_cut=confirmed, scene_quarantine=quarantine,
            pixel_sample_gap=bool(gap), scene_visual_available=True,
            scene_hist_distance=round(distance,5), scene_aligned_residual=round(residual,5),
            scene_phase_response=round(float(response),5))
        self.result.update(self.floor_evidence(frame, observations))
        return dict(self.result)

    def floor_evidence(self, frame, observations):
        self.appearance_frame = cv2.resize(frame, (640,360), interpolation=cv2.INTER_AREA)
        small = cv2.resize(cv2.resize(frame, (640,360), interpolation=cv2.INTER_AREA), (320,180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Relaxed floor color is only a proposal; connected geometry, long
        # white markings and person-specific foot support must also agree.
        wood = (((h >= 5) & (h <= 38) & (s <= 170) & (v >= 65)) |
                ((s <= 38) & (v >= 110))).astype(np.uint8)
        wood[:int(.45*180)] = 0
        # The scorebar obscures the central footer, not the whole court width.
        wood[int(.90*180):, int(.20*320):int(.80*320)] = 0
        wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # A painted key can separate the shoes from wood. Only adjacent lower
        # blue pixels are admitted, not the large blue arena wall above.
        paint = ((h >= 90) & (h <= 130) & (s >= 45) & (v >= 45)).astype(np.uint8)
        paint[:int(.70*180)] = 0
        paint[int(.90*180):, int(.20*320):int(.80*320)] = 0
        near_wood = cv2.dilate(wood, np.ones((11,11),np.uint8))
        wood = np.maximum(wood, paint * near_wood)
        for o in observations:
            x1,y1,x2,y2=o['bbox'];w,h=o['width'],o['height']
            wood[max(0,int(y1/h*180)-1):min(180,int(y2/h*180)+2),
                 max(0,int(x1/w*320)-1):min(320,int(x2/w*320)+2)] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(wood)
        self.floor = np.zeros_like(wood)
        area = 0.; span = 0.
        # People occlude and split the floor. Aggregate only sizable observed
        # lower fragments; never fill a person's box with invented floor.
        for idx in range(1,count):
            x,y,bw,bh,n=stats[idx]
            if n/wood.size >= self.c['floor_fragment_min_area'] and y+bh >= .78*180:
                self.floor[labels == idx] = 1
        area = float(self.floor.mean())
        xs=np.flatnonzero(self.floor.any(axis=0))
        span=float((xs[-1]-xs[0]+1)/320) if len(xs) else 0.
        # Include painted-area boundaries as well as white markings. A white
        # only mask misses white lines on pale, tinted floors.
        edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 40, 100)
        near_floor = cv2.dilate(self.floor, np.ones((9,9),np.uint8))
        edges[near_floor == 0] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 22, minLineLength=35, maxLineGap=8)
        valid_lines = []
        for line in ([] if lines is None else lines[:,0]):
            x1,y1,x2,y2 = map(int,line)
            # Vertical walls/rails alone cannot support a court decision.
            if abs(x2-x1) > 35 and abs(y2-y1) < abs(x2-x1):
                valid_lines.append([x1,y1,x2,y2])
        support = (area >= self.c['floor_min_area'] and span >= self.c['floor_min_span'] and
                   len(valid_lines) >= self.c['min_marking_lines'])
        return dict(independent_floor_area=round(area,5),
            independent_floor_span=round(span,5), independent_marking_lines=len(valid_lines),
            independent_match_scene=bool(support),
            scene_reason='floor_and_markings' if support else 'independent_scene_support_missing')

    def person(self, observation):
        x1,y1,x2,y2 = observation['bbox']; w,h = observation['width'],observation['height']
        x = int((x1+x2)/2/w*320); y = int(y2/h*180)
        # Foot pixels themselves can be shoes/paint; inspect floor on both
        # sides of the feet, rather than using the person's shirt as support.
        regions = [self.floor[max(0,y-6):min(180,y+4), max(0,int(edge)-9):min(320,int(edge)+9)]
                   for edge in (x1/w*320, x2/w*320)]
        support = max((float(r.mean()) for r in regions if r.size), default=0.)
        waist = self.appearance_frame[max(0,int((y1+.42*(y2-y1))/h*360)):min(360,int((y1+.65*(y2-y1))/h*360)),
            max(0,int((x1+.30*(x2-x1))/w*640)):min(640,int((x1+.70*(x2-x1))/w*640))]
        white=purple=yellow=0.
        if waist.size:
            hue,sat,val=cv2.split(cv2.cvtColor(waist,cv2.COLOR_BGR2HSV))
            white=float(np.mean((sat<=65)&(val>=175)))
            purple=float(np.mean((hue>=115)&(hue<=165)&(sat>=65)))
            yellow=float(np.mean((hue>=15)&(hue<=40)&(sat>=90)&(val>=75)))
        return dict(waist_white=round(white,4),waist_purple=round(purple,4),waist_yellow=round(yellow,4),
                    independent_foot_floor_fraction=round(support,4),
                    independent_person_support=bool(support >= self.c['foot_floor_fraction']))
