from face_recognition import face_locations

def extract_box(image, single = True):
    """
    return <start_Y>, <start_X>, <end_Y>, <end_X>
    """
    boxs = face_locations(image)

    if len(boxs) == 0:
        return None

    if single:
        return boxs[0]

    return boxs

def extract_face(image, return_bbox=False, face_scale_thres = (20, 20)):
    h, w = image.shape[:2]

    try:
        (startY, startX, endY, endX) = extract_box(image)
    except Exception as e:
        print(e)
        return None

    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    # extract the face ROI and grab the ROI dimensions
    (fH, fW) = face.shape[:2]
    # ensure the face width and height are sufficiently large
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None

    if return_bbox:
        return face, (startX, startY, endX - startX, endY - startY)

    return face
