import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pag
import time
import sys
from imutils.video import WebcamVideoStream

class HandControl:

    # We are only detecting 1 hand and would like a medium-high confidence for detection and tracking
    def __init__(self, dconf=0.7, tconf=0.6, maxhands=1):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=dconf, max_num_hands=maxhands, min_tracking_confidence=tconf)
        self.distamt=200
        self.dist=0
        self.previous_x=0
        self.previous_y = 0
        self.filtersize=2
        self.filter_x=[]
        self.filter_y=[]
        self.click_time=0

    def drawFeed(self, image):
        # Draw the hand annotations on the image.
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            return False
        return True

    def generateHands(self, image):
        # Generates the pose estimation of the hands from the MediaPipe library
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)

    def generateMatrix(self):
        # We create a matrix of all the fingers in both the x and y axis for easier referencing later on
        fbase=[1,5,9,13,17]
        self.mcp_mx={}
        self.pip_mx = {}
        self.dip_mx = {}
        self.tip_mx = {}
        self.mcp_mx['x'] = np.array([self.results.multi_hand_landmarks[0].landmark[i].x for i in fbase])
        self.mcp_mx['y'] = np.array([self.results.multi_hand_landmarks[0].landmark[i].y for i in fbase])
        self.pip_mx['x'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+1].x for i in fbase])
        self.pip_mx['y'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+1].y for i in fbase])
        self.dip_mx['x'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+2].x for i in fbase])
        self.dip_mx['y'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+2].y for i in fbase])
        self.tip_mx['x'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+3].x for i in fbase])
        self.tip_mx['y'] = np.array([self.results.multi_hand_landmarks[0].landmark[i+3].y for i in fbase])

    def basecalc(self):
        # Calculating the inverse width of the palm to set the sensitivity of the mouse cursor
        if np.all(self.tip_mx['y']<self.pip_mx['y']):
            self.dist=((self.mcp_mx['y'][0]-self.mcp_mx['y'][4])**2+(self.mcp_mx['x'][0]-self.mcp_mx['x'][4])**2)**0.5
            self.dist=1/self.dist*self.distamt

    def movement(self):

        # The movement of the cursor based on whether only the index finger or both the index
        # and middle finger are raised. Calculated via the current coords - the previous coords and
        # put through a smoothening filter.
        current_x=self.mcp_mx['x'][1]
        current_y=self.mcp_mx['y'][1]
        move_x=current_x-self.previous_x
        move_y = current_y - self.previous_y
        fmove_x,fmove_y = self.filterMov(move_x,move_y)
        if np.all((self.tip_mx['y'][1:]<self.mcp_mx['y'][1:])==[1,0,0,0]):
            pag.move(fmove_x*self.dist/5, fmove_y*self.dist/4)
        elif np.all((self.tip_mx['y'][1:]<self.mcp_mx['y'][1:])==[1,1,0,0]):
            pag.move(fmove_x*self.dist, fmove_y*self.dist)
        else:
            self.filter_x=[]
            self.filter_y=[]
        self.previous_x=current_x
        self.previous_y=current_y

    def filterMov(self,move_x,move_y):

        # A simple smoothening filter to reduce noise
        self.filter_x.append(move_x)
        self.filter_y.append(move_y)
        if len(self.filter_x)>self.filtersize:
            self.filter_x.pop(0)
            self.filter_y.pop(0)
        return sum(self.filter_x)/len(self.filter_x), sum(self.filter_y)/len(self.filter_y)

    def clickAction(self):

        # Detects whether there was a clicking action with the index finger. Based on the speed
        # we will determine if it is a single or double click
        if np.all((self.tip_mx['y'][1:] < self.mcp_mx['y'][1:]) == [1, 0, 0, 0]):
            if self.tip_mx['y'][1] > self.dip_mx['y'][1]:
                if self.click_time==0:
                    self.click_time=self.cur_time
            elif self.click_time>0:
                if self.cur_time - self.click_time > 0.6:
                    pag.doubleClick()
                    self.click_time = 0
                elif self.cur_time-self.click_time>0.1:
                    pag.click()
                    self.click_time=0
    def endCheck(self):

        # Ends the program if only the pinky finger is raised.
        if np.all((self.tip_mx['y'][1:] < self.mcp_mx['y'][1:]) == [0, 0, 0, 1]):
            return False
        else:
            return True


if __name__ == '__main__':
    args = sys.argv

    #We will be using WebcamVideoStream to increase the speed of the program
    vs = WebcamVideoStream(src=0).start()
    hc=HandControl()

    movetime=0
    continueProg = True

    while not vs.stopped:

        # Getting the start time of this loop
        hc.cur_time=time.time()

        # Reading the next frame of the video stream
        image = vs.read()
        if not vs.grabbed:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)

        hc.generateHands(image)

        if hc.results.multi_hand_landmarks and hc.cur_time-movetime>0.05:
            hc.generateMatrix()
            hc.basecalc()
            hc.movement()
            hc.clickAction()
            continueProg = hc.endCheck()

            if 'video' in args and continueProg:
                continueProg = hc.drawFeed(image)
            if not continueProg:
                print("End Program")
                break

            #setting the last move time
            movetime=hc.cur_time

    vs.stop()
    hc.hands.close()
