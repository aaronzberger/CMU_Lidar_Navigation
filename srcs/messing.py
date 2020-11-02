import math
from shapely.geometry import LineString
import numpy as np

def distance(x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    return math.sqrt(dx**2 + dy**2)

def angle_to_line(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    
    a = (dy*(200-y1)+dx*(200-x1))/det
    closestX, closestY = x1+a*dx, y1+a*dy
    
    return math.atan2(closestY - 200, closestX - 200)

def line_to_point(x, y, a, b, c):
    return (abs(a * x + b * y + c)) / (math.sqrt(a * a + b * b))

def line_to_line(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.distance(line2)
    
def segment_to_standard(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    a = y1 - y2
    b = x2 - x1
    c = (x1-x2)*y1 + (y2-y1)*x1
    
    return a, b, c

def intersect(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.intersects(line2)
    

def lines_are_close(l1, l2, dist, theta_deg):
    a1, b1, c1 = segment_to_standard(l1)
    line1Dist = line_to_point(200, 200, a1, b1, c1)
    line1Angle = angle_to_line(l1)
    
    a2, b2, c2 = segment_to_standard(l1)
    line2Dist = line_to_point(200, 200, a2, b2, c2)
    line2Angle = angle_to_line(l2)
    
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    x3, y3 = l2[0][0], l2[0][1]
    x4, y4 = l2[0][2], l2[0][3]
    
    shortestDistance = line_to_point(x1, y1, a2, b2, c2)
    shortestDistance = line_to_point(x2, y2, a2, b2, c2) if line_to_point(x2, y2, a2, b2, c2) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x3, y3, a1, b1, c1) if line_to_point(x3, y3, a1, b1, c1) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x4, y4, a1, b1, c1) if line_to_point(x4, y4, a1, b1, c1) <= shortestDistance else shortestDistance

    return intersect(l1, l2) or shortestDistance < dist or (abs(line1Dist - line2Dist) < dist and abs(line1Angle - line2Angle) < math.radians(theta_deg))


def best_line(l1, l2):
    # Just pick the longest line
    line1Length = math.sqrt(((l1[0][2] - l1[0][0]) ** 2) + ((l1[0][3] - l1[0][1]) ** 2))
    line2Length = math.sqrt(((l2[0][2] - l2[0][0]) ** 2) + ((l2[0][3] - l2[0][1]) ** 2))

    return line1Length >= line2Length

def distance_to_origin(l1):
    a, b, c = segment_to_standard(l1)
    return line_to_point(200, 200, a, b, c)



lines = [
    [[47, 51, 57, 230]],
    [[50, 275, 52, 399]]
]

tracker = np.ones(len(lines))


newLines = []
for i in range(0, len(lines)):
    if tracker[i] == 1:
        group = []
        group.append(lines[i])
        for j in range(1, len(lines)):
            if tracker[j] == 1:
                for line in group:
                    if lines_are_close(line, lines[j], 20, 10):
                        group.append(lines[j])
                        tracker[j] = 0
                        break
        tracker[i] = 0
        newLines.append(group)
        


print("Groups of Lines: (%s)\n" % len(newLines))
for group in newLines:
    print("Group:")
    for line in group:
        print("%s Distance: %s Angle: %s" % (line, distance_to_origin(line), angle_to_line(line)))
    print("\n")
    
a, b, c = segment_to_standard(lines[0])
print(line_to_point(52, 399, a, b, c))