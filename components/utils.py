class Track(object):
    """ Class representing a biigle track annotation.
        Attributes: filename: filename of the annotated video (str)
                    label_id: biigle id of the label linked to this annotation (int)
                    shape_id: biigle id of the shape used to annotate (int)
                    video_size: size of the video in pixels (width, height) (tuple of float)
                    keyframes: dict of (time, annot coords) pair values (time is a float, coords is a list of floats)
    """
    def __init__(self, filename, label_id, shape_id, video_size):
        self.filename = filename
        self.label_id = label_id
        self.shape_id = shape_id
        self.video_size = video_size
        self.keyframes = {}

    def add_keyframe(self, time, coords):
        self.keyframes.update({time:coords})
