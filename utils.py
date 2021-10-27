

def x1y1wh2rxywh(input, imageShape):
    ''' Take x1,y1, width, height,  and return [0,1] scaled centroid coordinates and size '''
    x1, y1, w, h = input
    imgY, imgX = imageShape[0:2]
    x = (x1 + w/2)/imgX
    y = (y1 + h/2)/imgY
    w = w/imgX
    h = h/imgY

    return(x, y, w, h)


def rxywh2x1y1wh(input, imageShape):
    ''' Take [0,1] scaled centroid x,y, width, height,  and return x1,y1 coordinates and size '''
    x, y, w, h = input
    imgY, imgX = imageShape[0:2]
    x1 = int((x - w/2)*imgX)
    y1 = int((y - h/2)*imgY)
    w = int(w*imgX)
    h = int(h*imgY)

    return(x1, y1, w, h)


def x1y1x2y22rxywh(input, imageShape):
    ''' Take x1,y1, x2, y2,  and return [0,1] scaled centroid coordinates and size '''
    x1, y1, x2, y2 = input
    w = x2-x1
    h = y2-y1
    imgY, imgX = imageShape[0:2]
    x = (x1 + w/2)/imgX
    y = (y1 + h/2)/imgY
    w = w/imgX
    h = h/imgY

    return(x, y, w, h)


def yolo2xywh(input, imageShape):
    ''' Take yolo results and return and return x1,y1 coordinates and size '''
    return rxywh2x1y1wh(yolo2rxywh(input), imageShape)


def yolo2rxywh(input):
    ''' Take yolo results and return and return [0,1] scaled centroid coordinates and size '''
    (x, y), (w, h), _, _ = input

    return(float(x), float(y), float(w), float(h))
