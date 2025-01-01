import math
import cv2
import mediapipe as mp
import time


class FaceDectector():
    def __init__(self, staticMode=False, maxFaces=2, refineLandmark=False, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmark = refineLandmark
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLandmark,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def detectFace(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        AllfacelmCordinates = []
        if self.results.multi_face_landmarks:
            for id1, facrLms in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, facrLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec,
                                               self.drawSpec)
                facelmCordinates = []
                for id2, lm in enumerate(facrLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # cv2.putText(img,str(id2),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
                    facelmCordinates.append([id2, cx, cy])
                AllfacelmCordinates.append([id1, facelmCordinates])
        return img, AllfacelmCordinates

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 3, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    fd = FaceDectector()
    while True:
        success, img = cap.read()
        img, faces = fd.detectFace(img)

        if len(faces) != 0:
            print(faces)
            pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS:" + str(int(fps)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
