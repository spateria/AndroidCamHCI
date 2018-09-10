# -*- coding: utf-8 -*-

import sys
import re
import os.path
import time
import math
import cv2
import numpy as np
import threading

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *



### Get the path of current file directory ########
PATH = os.path.dirname(os.path.realpath(__file__))
temp = ""
for s in PATH:
    if s == '\\':
        temp += '/'
    else:
        temp += s
temp += '/'
PATH = temp



###### Gloabl Variables and Objects #############
class Globals():
    itr = 0
    
    #### primitives ####
    faces = []
    vertices = []
    normals = []
    
    #### Figure Manipulation ####
    Xtr = -0.05 # * -(max(vertices[:][0]) + min(vertices[:][0]))/2.0   ## X- origin
    Ytr = -0.005 # * -(max(vertices[:][1]) + min(vertices[:][1]))/2.0   ## Y- origin 
    Ztr = -0.2 
    Xrot = 0  ## rotate along X-axis
    Yrot = 0 
    Zrot = 0
    
    
    #### Interaction ####
    isActive = 0 #### activate control
    gest = -1    #### gesture
    
    objDepth = 0 
    objCentroid = []
    
    zeroDepth = 0
    zeroCentroid = []
    

    ### program control ###
    Quit = 0 
    


###### Constrain the angular values to max 360 degrees #######
def mod360():
    Globals.Xrot = Globals.Xrot%360
    Globals.Yrot = Globals.Yrot%360
    Globals.Zrot = Globals.Zrot%360
    
    

####### PARSER ###########
def parseMfile(prim_file):

    Globals.faces = [] 
    Globals.vertices = [] 
    Globals.normals = [] 
    
    
    #### to scale the co-ordinates to 0.0xxx ####
    def log_scale(n):
        if n == 0:
            dg = 1
        else:
            n = abs(n)
            dg = math.ceil(math.log10(n)) 
        
        sc = 1
        if dg >= 0:  ## number is greater than any 0.0xxxx
            sc = 1 / float((10**(dg+1)))
        
        return sc


    vertmax = 0.0  ### will save max vertex value for overall scaling
    
    with open(prim_file) as f:
        for line in f:
            l = line.split()
            
            if l[0] == 'Vertex':
                Globals.vertices.append([float(l[2]), float(l[3]), float(l[4])])
                mx = max(float(l[2]), float(l[3]), float(l[4]))
                if mx > vertmax:
                    vertmax = mx
                
                l[5] = re.sub('{normal=\(', '', l[5])  ### remove special characters from normals
                l[7] = re.sub('\)}', '', l[7])
                Globals.normals.append([float(l[5]), float(l[6]), float(l[7])])
    
            elif l[0] == 'Face':
                Globals.faces.append([int(l[2]), int(l[3]), int(l[4])])

            else:
                print ('invalid attribute!!!')
    
    
    
    scale = log_scale(vertmax)
    
    for i in range(len(Globals.vertices)):
        Globals.vertices[i][0] *= scale 
        Globals.vertices[i][1] *= scale 
        Globals.vertices[i][2] *= scale 



##### Define Video capture and Interaction methods #####
class Interaction():
    
    
        def doInteract(self):
        
            if Globals.isActive == 1:
                    
                if Globals.gest == 0:   #### Scale or Zoom
                                        
                    if Globals.zeroDepth == 0:
                        Globals.zeroDepth = Globals.objDepth
                    else:
                        Globals.Ztr += -( ((int((Globals.objDepth - Globals.zeroDepth)/100) * 100) / 100000.0) )  ### quantization and value scaling
                        
                        Globals.zeroDepth = Globals.objDepth   ### zoom is relative
                        
                        ### bound range
                        if Globals.Ztr > -0.1:
                            Globals.Ztr = -0.1
                        elif Globals.Ztr < -0.7:
                            Globals.Ztr = -0.7
                    
        
                if Globals.gest == 1:   #### Rotate
                    
                    if Globals.zeroCentroid == []:
                        Globals.zeroCentroid = Globals.objCentroid
                    else:
                        loc = [Globals.objCentroid[0] - Globals.zeroCentroid[0], Globals.objCentroid[1] - Globals.zeroCentroid[1]]
                        
                        Globals.Yrot += int((loc[0]*600)/2) * 2
                        Globals.Xrot += int(-(loc[1]*600)/2) * 2
                        
                        Globals.zeroCentroid[0] = Globals.objCentroid[0]
                        Globals.zeroCentroid[1] = Globals.objCentroid[1]
                
        
                
            mod360()
            
        
        #### check if control should be activated ####
        def detActivate(self, fgbg, img):
            
            actimg = fgbg.apply(img)

            if cv2.countNonZero(actimg) == 0:
                if Globals.isActive == 1:
                    Globals.isActive = 0
                    
            else:
                if Globals.isActive == 0:
                    Globals.isActive = 1

            
            
        def vidCap(self):
            
            ### read video feed
            cap = cv2.VideoCapture(0)
            
            ### foreground-backgorund mask
            fgbg = cv2.createBackgroundSubtractorMOG2()
            
            ### gesture flag
            gest = -1
            
                        
            ### check OpenCV version ##
            (version, _, _) = cv2.__version__.split('.')
                
            transition_buffer = 0   #### to avoid noisy manipulations while changin gestures
            
            
            while(cap.isOpened()):
                
                    # read image
                    ret, img = cap.read()
                
                    rows = img.shape[0]
                    cols = img.shape[1]
                        
                        
                    # convert to grayscale
                    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                
                    #### Check for CLOSING THE INTERFACE by covering the camera ####
                    if cv2.countNonZero(grey) == 0:
                        break
                        
                
                    # applying gaussian blur
                    value = (35, 35)
                    blurred = cv2.GaussianBlur(grey, value, 0)
                
                
                    #### check if control should be activated ####
                    self.detActivate(fgbg, blurred)
            
                    
                    if Globals.isActive == 1:
                            # thresholdin: Otsu's Binarization method
                            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                            ### find contours ###
                            if version == '3':
                                image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            elif version == '2':
                                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        
                
                            try:
                                # find contour with max area
                                cntr = max(contours, key = lambda x: cv2.contourArea(x))
                        
                                cvxhull = cv2.convexHull(cntr)
                                        
 
                                ############# GET GESTURE ##################
                                        
                                cx2 = 0;  cy2 = 0; ### marker for rotation
                                
                                
                                # finding convexity defects
                                hull_idx = cv2.convexHull(cntr, returnPoints = False)
                                defects = cv2.convexityDefects(cntr, hull_idx)
                                count_defects = 0
        
                                interest_points = []
                                
                                for i in range(defects.shape[0]):
                                    s,e,f,d = defects[i,0]   ## indices of the start point, end point, farthest point, and the distance to farthest point 
                
                                        
                                    start = tuple(cntr[s][0])
                                    end = tuple(cntr[e][0])
                                    far = tuple(cntr[f][0])
                                    
                                    
                                    ### Counting finger-to-finger depressions ###
                                    
                                    ### Method 1 : based on depression length
                                    '''if d > 50 and d < 150:
                                        count_defects += 1'''
                                        
                                    ### Method 2: based on traingles
                                    # find length of all sides of triangle
                                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            
                                    # angle
                                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                
                                    ### NOTE: gestures should be formed such that two adjacent fingers make acute  traingle
                                    if angle <= 90:
                                        count_defects += 1
                                        trngl_centroid = (cntr[s][0] + cntr[e][0] + cntr[f][0])/3.0
                                        interest_points.append(trngl_centroid)
                                        
                            
                            
                                ####### set gesture ##############################
                                if count_defects >= 3:
                                    gest = 0   #### used for Zoom or Scaling
                                    
                                elif count_defects >= 1 and count_defects < 3:
                                    gest = 1  #### used for rotation
                                    cx2 = int(interest_points[0][0])
                                    cy2 = int(interest_points[0][1])
                                    
                                else:
                                    gest = -1  ### do nothing
                                    #### reset ###
                                    Globals.objDepth = 0 
                                    Globals.objCentroid = []
                        
                                    Globals.zeroDepth = 0
                                    Globals.zeroCentroid = []
                                #################################################
                                    
                                    
                                
                                ### draw
                                cntr_img = np.zeros(img.shape,np.uint8)
                                cv2.drawContours(cntr_img, [cntr], 0, (0, 255, 0), 0)
                                cv2.drawContours(cntr_img, [cvxhull], 0, (0, 0, 255), 0)
                                cv2.rectangle(cntr_img, (int(0.1*cols), int(0.1*rows)), (int(0.9*cols), int(0.9*rows)), [255,255,255])
                                cv2.putText(cntr_img, 'move hand within white rectangle only', (int(0.05*cols), int(0.95*rows)), 1, 1.0, (255,255,255)) 
                                
                                if cx2 != 0 and cy2 != 0: 
                                    cv2.circle(cntr_img, (cx2, cy2), 5, [0,255,255], -1)  
                                
                                
                                gesture = ['None', 'Scale', 'Rotate']
                                show = 'active = ' + str(Globals.isActive) + ';  gesture = ' + gesture[Globals.gest + 1]
                                cv2.putText(cntr_img, show, (int(0.05*cols), int(0.05*rows)), 1, 1.0, (255,255,255))       
                                cv2.imshow('Interaction Window', cntr_img)           
                                cv2.waitKey(10) 
                            
                            except Exception as e: 
                                print(e)
                                continue                     
                                    
                                    
                            
                            if gest != Globals.gest:
                                #time.sleep(0.5)      ### don't use! this causes lag in display
                                if transition_buffer > 10:
                                    Globals.gest = gest
                                    transition_buffer = 0
                                
                                transition_buffer += 1 
                            
    
    
                    else:  #### not active
                            #### show empty window ####
                            blank = np.zeros((rows, cols), np.uint8)
                            gesture = ['None', 'Scale', 'Rotate']
                            show = 'active = ' + str(Globals.isActive) + ';  gesture = ' + gesture[Globals.gest + 1]
                            cv2.rectangle(blank, (int(0.1*cols), int(0.1*rows)), (int(0.9*cols), int(0.9*rows)), [255,255,255])
                            cv2.putText(blank, show, (int(0.05*cols), int(0.05*rows)), 1, 1.0, (255,255,255))       
                            cv2.imshow('Interaction Window', blank)           
                            cv2.waitKey(10) 
                        
                        
                        
                        
                        
                        
                    ##### INTERACTION BASED UPDATES ############    
                    if Globals.isActive == 1:
                        
                            if Globals.gest == 0:
                                Globals.objDepth = cv2.contourArea(cntr)

                            elif Globals.gest == 1:
                                cx2 = cx2/cols
                                cy2 = cy2/rows
                                if (cx2 > 0.1 and cx2 < 0.9) and (cy2 > 0.1 and cy2 < 0.9):  ### the tracking point does not fall at the edges
                                    Globals.objCentroid = [cx2, cy2]

                
                            self.doInteract()
                    
                    
            ##### (cap.isOpened())  loop running...
                
                        
                        
                        
            Globals.Quit = 1
            cap.release()
            cv2.destroyAllWindows()
            print ('exiting')
            sys.exit()



#### Define methods and Settings for Mesh Rendering  
class Render():    
    
        def __init__(self, interact):
            
            self.interact_obj = interact   #### hold an instance of the Interaction class
            
            
        def load_primitives1(self):
            
                norms = Globals.normals
                verts = Globals.vertices
                faces = Globals.faces
        
                glBegin(GL_TRIANGLES)
                
                glColor3f(1.0, 1.0, 0.0)
                
                for face in faces:
                    
                    fx = face[0]-1      
                    glNormal3f(norms[fx][0], norms[fx][1], norms[fx][2])
                    glVertex3f(verts[fx][0], verts[fx][1], verts[fx][2])
            
                    fx = face[1]-1 
                    glNormal3f(norms[fx][0], norms[fx][1], norms[fx][2])
                    glVertex3f(verts[fx][0], verts[fx][1], verts[fx][2])
            
                    fx = face[2]-1         
                    glNormal3f(norms[fx][0], norms[fx][1], norms[fx][2])
                    glVertex3f(verts[fx][0], verts[fx][1], verts[fx][2])
                    
                    
                glEnd()
        
        
        
        def init_params(self, width, height):
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClearDepth(1.0)
            
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_TEST)
            glShadeModel(GL_SMOOTH)
            
            
            #### set projection 
            glMatrixMode(GL_PROJECTION)
            gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
            
            
            #### set camera view
            glMatrixMode(GL_MODELVIEW)
            
            # initialize lighting 
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHTING)
        
            # enable material color for objects (independent of lighting)
            glColorMaterial(GL_FRONT, GL_DIFFUSE)    
            glEnable(GL_COLOR_MATERIAL)
            
        
        
        def disp (self):

            if Globals.Quit == 1:
                print ('leaving draw loop')
                glutLeaveMainLoop()
                sys.exit()
                
                
                ##fixed
            '''#self.interact_obj.doInteract()   #### get values from interaction 
                                             #### this code may be put into the video cap thread
                                             #### to prevent costly operations in the display loop
                                             #### however, video cap loop runs at a higher rate than rendering 
                                             #### and it was causing synchronization problem'''
            
        
        
            ### per loop ops
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
        
            
            ## draw axes
            glTranslatef(0.0, 0.0, -0.2) 
            glLineWidth(4.0)
            glBegin(GL_LINES)
            
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 0.0, 0.0)
            
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 1.0, 0.0)
        
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(-0.2, -0.2, 1.0)
            
            glEnd()
            
        
            glClear(GL_DEPTH_BUFFER_BIT)     #### to overlay the next figure
            
            
            glPushMatrix()
            
            ### translate
            glTranslatef(Globals.Xtr, Globals.Ytr, Globals.Ztr)  ## position the object
            
            ## rotate
            glTranslatef(-Globals.Xtr, -Globals.Ytr, 0)
            glRotatef(Globals.Xrot, 1.0,0.0,0.0)
            glRotatef(Globals.Yrot, 0.0,1.0,0.0)
            glRotatef(Globals.Zrot, 0.0,0.0,1.0)    
            glTranslatef(Globals.Xtr, Globals.Ytr, 0)
            
        
            ### load figure #####
            self.load_primitives1()
            
            glPopMatrix()       
            
            
            glutSwapBuffers()
        
            
            
            
        def setglut(self):
            
            glutInit()
        
            ### Init Display Settings #######
            width = 640*2
            height = 480*2    
            
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
            glutInitWindowSize(width, height)
            glutInitWindowPosition(50, 50)       
            glutCreateWindow(b"HCI - Mesh View")   
            
            
            ### Init rendering settings
            self.init_params(width, height)
            
            
            ### Assign display func
            glutDisplayFunc(self.disp)
            glutIdleFunc(self.disp)
            
            
            ### Start!!
            print ("Starting GLUT Loop......")
            glutMainLoop()
            print ("Exiting...")
        
            sys.exit()




def main():
    
    if sys.argv[1:] != []:   ### if file name given as commnd line arg
        prim_file = sys.argv[1:][0]
    else:
        fn = input('Choose from current directory {TYPE 1 for Bunny.m; 2 for Gargoyle.m} \nOR \nEnter full path+filename\nhere: ')
        if fn == '1':
            prim_file = PATH + 'Bunny.m'
        elif fn == '2':
            prim_file = PATH + 'Gargoyle.m'
        else:
            prim_file = fn
            
        
    if os.path.exists(prim_file):
        print ('Parsing file...', prim_file)
        parseMfile(prim_file)
    else:
        print ('file does not exist!!', '...', prim_file)
        sys.exit()
        
        
        
    interact = Interaction()
    t1 = threading.Thread(target = interact.vidCap)
    t1.start()

    
    rndr = Render(interact)
    rndr.setglut()
    
    
if __name__ == "__main__":
    main()