class FacialIndexError(Exception):
    pass

class ProfileFoundNoFaceError(FacialIndexError):
    def __init__(self, profile_signature):
        super().__init__(
            f'Profile {profile_signature} found no face with current backend. \
                Try setting enforce_detection to False or changing the detector_backend'
        )

class ImagePreprocessFaceError(FacialIndexError):
    def __init__(self, image_signature):
        super().__init__(
            f'Image {image_signature} found no face while preprocessing with current backend. \
                Try setting enforce_detection to False or changing the detector_backend'
        )