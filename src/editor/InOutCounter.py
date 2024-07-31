import cv2

class InOutCounter:
    def __init__(self,inLine,outLine):
        self.inLine = inLine
        self.outLine = outLine
        self.inLineStart = inLine[0]
        self.inLineEnd = inLine[1]
        self.outLineStart = outLine[0]
        self.outLineEnd = outLine[1]
        self.midpoint = {}
        self.inMemory = {}
        self.outMemory = {}
        self.count = {}
        pass

    def draw_lines(self,img,prev_midpoint,curr_midpoint):
        cv2.line(img, self.inLineStart, self.inLineEnd, (255, 255, 255), 2)
        cv2.line(img, self.outLineStart, self.outLineEnd, (0, 0, 0), 2)
        cv2.line(img, prev_midpoint, curr_midpoint, (0, 0, 0), 2)
        return

    def intersection(self,line1,line2):
        #Extract the coordinates of the lines
        A,B = line1
        C,D = line2

        #Calculate the direction vectors
        vector1 = (B[0] - A[0], B[1] - A[1])
        vector2 = (D[0] - C[0], D[1] - C[1])

        #Calculate the determinant of direction vectors
        det = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        #Calculate the intersection point
        if det != 0:
            t = ((C[0] - A[0])*(D[1] - C[1]) - (C[1] - A[1])*(D[0] - C[0]))/det
            u = ((C[0] - A[0])*(B[1] - A[1]) - (C[1] - A[1])*(B[0] - A[0]))/det

            #Check if the intersection point is within the line segments
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
            else:
                return False
        return False

    def trackObject(self,img,midpoint,id):
        #Check if object has been recorded before
        if id not in self.midpoint:
            self.inMemory[id] = False
            self.outMemory[id] = False
            self.count[id] = 0
            self.midpoint[id] = []
            self.midpoint[id].append(midpoint)
        
        #Get previous midpoint
        prev_midpoint = self.midpoint[id][0]
        curr_midpoint = midpoint
        self.draw_lines(img,prev_midpoint,curr_midpoint)

        #Replace previous midpoint with current midpoint
        self.midpoint[id][0] = curr_midpoint

        #Check if object has entered before
        if self.inMemory[id] == False:
            #Check if object has crossed the inLine
            print(f"Checking for object {id}")
            if self.intersection(self.inLine,(prev_midpoint,curr_midpoint)):
                self.inMemory[id] = True
                print(f"Object {id} has entered")
        else:
            #Object has entered, check for exit
            print(f"Checking for entered object {id}")
            if self.intersection(self.outLine,(prev_midpoint,curr_midpoint)):
                self.inMemory[id] = False
                print(f"Object {id} has exited")
                self.count[id] += 1
        return self.count[id]


