from __future__ import annotations
from typing import List, Optional, Dict
import yaml
import pykwalify.core

from filter_simulator.common import FrameList, Frame, Position


class ScenarioData:
    def __init__(self):
        self.meta: Optional[MetaInformation] = None
        self.ds: Optional[FrameList] = None
        self.fas: Optional[FrameList] = None
        self.mds: Optional[FrameList] = None
        self.gtts: Optional[List[GroundTruthTrack]] = None
        self.tds: Optional[List[float]] = None
    # end def main

    @ staticmethod
    def _read_detections(detections_frames: List[Optional[List[Dict]]], frame_list: FrameList):
        for frame in detections_frames:
            if frame is not None:
                f: Frame = Frame()
                for detection in frame:
                    f.add_detection(Position(detection.get("x"), detection.get("y")))
                # end for
                frame_list.add_frame(f)

            else:
                frame_list.add_empty_frame()
            # end if
        # end for
    # end def

    def read_file(self, fn: str) -> ScenarioData:
        with open(fn) as f:

            docs = [doc for doc in yaml.load_all(f, Loader=yaml.FullLoader)]
            doc = docs[0]

            for key, value in doc.items():

                if key == "meta-information":
                    self.meta = MetaInformation()

                    if "version" in value:
                        self.meta.version = value.get("version")

                    if "coordinate-system" in value:
                        self.meta.coordinate_system = value.get("coordinate-system")

                    if "number-steps" in value:
                        self.meta.number_steps = value.get("number-steps")

                    if "time-delta" in value:
                        self.meta.time_delta = value.get("time-delta")

                elif key == "detections":
                    self.ds = FrameList()
                    self._read_detections(value, self.ds)

                elif key == "missed-detections":
                    self.mds = FrameList()
                    self._read_detections(value, self.mds)

                elif key == "false-alarms":
                    self.fas = FrameList()
                    self._read_detections(value, self.fas)

                elif key == "ground-truth-tracks":
                    self.gtts = []

                    for gtt in value:
                        _gtt: GroundTruthTrack = GroundTruthTrack()
                        _gtt.begin_step = gtt.get("begin-step")
                        pts = gtt.get("points")

                        if pts is not None:
                            for point in pts:
                                _gtt.points.append(Position(point.get("x"), point.get("y")))
                            # end for
                        # end if
                        self.gtts.append(_gtt)
                    # end for

                elif key == "time-deltas":
                    self.tds = []

                    for time_delta in value:
                        self.tds.append(time_delta)
                    # end for
                # end if
            # end for
        # end with

        return self
    # end def

    @staticmethod
    def _write_detections(frame_list: FrameList) -> Optional[list]:
        if frame_list is not None and len(frame_list) > 0:
            d = list()  # List of frames

            for frame in frame_list:
                d.append(list())  # List of detections in current frame

                for detection in frame:
                    d[-1].append(dict())  # Add detection to current frame
                    d[-1][-1]["x"] = float(detection.x)
                    d[-1][-1]["y"] = float(detection.y)
                # end for
            # end for
            return d
        else:
            return None
        # end if
    # end def

    def write_file(self, fn: str):
        d = dict()

        # meta-information
        ##################
        block_name = "meta-information"
        d[block_name] = dict()
        meta_information = d[block_name]

        if self.meta.version is not None:
            meta_information["version"] = self.meta.version

        if self.meta.number_steps is not None:
            meta_information["number-steps"] = self.meta.number_steps

        if self.meta.coordinate_system is not None:
            meta_information["coordinate-system"] = self.meta.coordinate_system

        if self.meta.time_delta is not None:
            meta_information["time-delta"] = self.meta.time_delta

        # detections
        ############
        block_name = "detections"
        res = self._write_detections(self.ds)
        if res is not None:
            d[block_name] = res

        # missed detections
        ###################
        block_name = "missed-detections"
        res = self._write_detections(self.mds)
        if res is not None:
            d[block_name] = res

        # false alarms
        ##############
        block_name = "false-alarms"
        res = self._write_detections(self.fas)
        if res is not None:
            d[block_name] = res

        # ground truth tracks
        #####################
        if self.gtts is not None and len(self.gtts) > 0:
            block_name = "ground-truth-tracks"
            d[block_name] = list()
            gtts = d[block_name]

            for gtt in self.gtts:
                _gtt = dict()
                _gtt["begin-step"] = gtt.begin_step
                _points = list()

                for point in gtt.points:
                    _point = dict()
                    _point["x"] = float(point.x)
                    _point["y"] = float(point.y)
                    _points.append(_point)
                # end for
                _gtt["points"] = _points
                gtts.append(_gtt)
            # end for
        # end if

        # time deltas
        #############
        if self.tds is not None and len(self.tds) > 0:
            block_name = "time-deltas"
            d[block_name] = list()
            time_deltas = d[block_name]

            for td in self.tds:
                time_deltas.append(td)
        # end if

        # write to file
        ###############
        with open(fn, 'w') as f:
            yaml.dump(d, f, sort_keys=False)
        # end with
    # end def

    def cross_check(self) -> bool:
        # meta-information
        ##################

        # version
        if self.meta.version is None:
            print("'meta-information.version' needs to be set.")
            return False
        # end if

        version_list = ["1.0"]
        if self.meta.version not in version_list:
            print(f"'meta-information.version'' ({self.meta.version}) needs to be in {version_list}.")
            return False
        # end if

        # coordinate-system
        if self.meta.coordinate_system is None:
            print("'meta-information.coordinate-system' needs to be set.")
            return False
        # end if

        coordinate_system_list = ["ENU", "WGS84"]
        if self.meta.coordinate_system not in coordinate_system_list:
            print(f"'meta-information.coordinate-system'' ({self.meta.coordinate_system}) needs to be in {coordinate_system_list}.")
            return False
        # end if

        # number-steps
        if self.meta.version is None:
            print("'meta-information.number-steps' needs to be set.")
            return False
        # end if

        if self.meta.number_steps <= 0:
            print(f"'meta-information.number-steps' ({self.meta.number_steps}) needs to be > 0.")
            return False
        # end if

        # time-delta # XXX - das hier noch irgendwie mit den time-deltas weiter unten verwursten - und wie mit den kommandozeilen-parametern verwursten?
        if self.meta.time_delta <= 0:
            print(f"'meta-information.time-delta' ({self.meta.time_delta}) needs to be > 0.")
            return False
        # end if

        # detections
        ############

        # frames > length
        if len(self.ds) != self.meta.number_steps:
            print(f"'detections' needs exactly as many frames (has {len(self.ds)}) as specified in 'meta-information.number-steps' ({self.meta.number_steps}).")
            return False
        # end if

        # detections -> length
        if self.ds.get_number_of_detections() == 0:
            print("'detections' needs any detections (in total).")
            return False
        # end if

        # false alarms
        ##############
        if self.fas is not None and len(self.fas) != self.meta.number_steps:
            print(f"'false-alarms' needs exactly as many frames (has {len(self.fas)}) as specified in 'meta-information.number-steps' ({self.meta.number_steps}).")
            return False
        # end if

        # missed-detections
        ###################
        if self.mds is not None and len(self.mds) != self.meta.number_steps:
            print(f"'missed-detections' needs exactly as many frames (has {len(self.mds)}) as specified in 'meta-information.number-steps' ({self.meta.number_steps}).")
            return False
        # end if

        # ground-truth-tracks
        #####################
        if self.gtts is not None:
            for g, gtt in enumerate(self.gtts):
                # begin-step
                if gtt.begin_step is None:
                    print(f"'ground-truth-tracks[{g}].begin-step' needs to be set.")
                    return False

                if gtt.begin_step < 0:
                    print(f"'ground-truth-tracks[{g}].begin-step' ({self.meta.time_delta}) needs to be >= 0.")
                    return False

                # points > length
                if len(gtt.points) == 0:
                    print(f"'ground-truth-tracks[{g}].points' needs any points.")

                # points > index last point
                if gtt.begin_step + len(gtt.points) > self.meta.number_steps:
                    print(f"'ground-truth-tracks[{g}].begin-step' + length('ground-truth-tracks[{g}].points') ({gtt.begin_step + len(gtt.points)}) needs to be <= ({self.meta.number_steps}).")
            # end for
        # end if

        # time-deltas
        #############
        if self.tds is not None and len(self.tds) != self.meta.number_steps:
            print(f"'time-deltas' needs exactly as many numbers ({len(self.tds)}) as specified in 'meta-information.number-steps' ({self.meta.number_steps}).")
            return False
        # end if

        # cross-checks
        ##############
        ##############

        # false-alarms, ground-truth-tracks
        if self.mds is not None and self.gtts is not None:
            for f, frame in enumerate(self.mds):
                # Collect all GTT points for the current step
                gtt_points = []

                for gtt in self.gtts:
                    index = f - gtt.begin_step

                    if 0 <= index < len(gtt.points):
                        gtt_points.append(gtt.points[index])
                # end for

                for d, detection in enumerate(frame):
                    # > det in gtt
                    if not any([detection == point for point in gtt_points]):
                        print(f"'missed-detection[{d}]' needs to be in any ground-truth-track at the corresponding time-step ({f}).")
                        return False
                # end for
            # end for
        # end if

        return True
    # end def
# end class


class MetaInformation:
    def __init__(self):
        self.version = None
        self.coordinate_system = None
        self.number_steps = None
        self.time_delta = None
    # end def
# end class


class DetectionFrame:
    def __init__(self):
        self.dets: List[Position] = []
    # end def
# end class


class GroundTruthTrack:
    def __init__(self):
        self.begin_step = None
        self.points: List[Position] = []
    # end def

    def __str__(self):
        return f"GroundTruthTrack with {len(self.points)} point{'s' if len(self.points) != 1 else ''} starting at index {self.begin_step}."
# end class


def main():
    import logging
    logging.disable(logging.CRITICAL)

    c = pykwalify.core.Core(source_file="test.yaml", schema_files=["test_schema_v0.1.yaml"])

    try:
        c.validate(raise_exception=True)

    except pykwalify.core.SchemaError as e:
        print(e)
    # end try

    d = ScenarioData()
    d.read_file("test.yaml")
    d.write_file("test_out.yaml")
    d.cross_check()
    pass
# end def main


if __name__ == "__main__":
    main()
