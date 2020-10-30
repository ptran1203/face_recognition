from face_recognition import face_locations

FACE_SCALE_THRES = (20, 20)


def extract_box(image, single=True):
    """
    return <start_Y>, <start_X>, <end_Y>, <end_X>
    """
    boxs = face_locations(image)

    if len(boxs) == 0:
        return None

    if single:
        return boxs[0]

    return boxs


def _get_face(image, start_y, start_x, end_y, end_x):
    min_x, max_x = min(start_x, end_x), max(start_x, end_x)
    min_y, max_y = min(start_y, end_y), max(start_y, end_y)
    face = image[min_y:max_y, min_x:max_x].copy()
    fh, fw = face.shape[:2]
    # ensure the face width and height are sufficiently large
    if fw < FACE_SCALE_THRES[0] or fh < FACE_SCALE_THRES[1]:
        return None

    return face


def extract_multi_faces(image):
    h, w = image.shape[:2]

    try:
        locations = extract_box(image, single=False)
    except Exception as e:
        return None, None

    faces = []
    bboxs = []
    if locations is not None and len(locations):
        for location in locations:
            (start_y, start_x, end_y, end_x) = location
            face = _get_face(image, start_y, start_x, end_y, end_x)
            if face is not None:
                faces.append(face)
                bboxs.append((start_x, start_y, end_x - start_x, end_y - start_y))

    return faces, bboxs


def extract_face(image, return_bbox=False):
    h, w = image.shape[:2]

    try:
        (start_y, start_x, end_y, end_x) = extract_box(image)
    except Exception as e:
        return None

    face = _get_face(image, start_y, start_x, end_y, end_x)

    if return_bbox:
        return face, (start_x, start_y, end_x - start_x, end_y - start_y)

    return face
