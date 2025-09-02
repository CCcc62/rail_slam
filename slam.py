import cv2
import pangolin
import numpy as np
import OpenGL.GL as gl
import os

from multiprocessing import Process, Queue
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

# 构建地图，显示角点的点云和相机的位姿
class Map:
    def __init__(self, W, H):
        self.width  = W
        self.Height = H
        self.poses  = []
        self.points = []
        self.bbox_points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def add_observation(self, pose, points,bpoints):
        self.poses.append(pose)
        for point in points:
            self.points.append(point)
        for point in bpoints:
            self.bbox_points.append(point)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', self.width, self.Height)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.width, self.Height, 420, 420, self.width//2, self.Height//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0,   0,  0,
                                     0,  -1,  0))
        self.handler = pangolin.Handler3D(self.scam)
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.width/self.Height)
        self.dcam.SetHandler(self.handler)

    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # draw keypoints
        gl.glPointSize(1)
        gl.glColor3f(0.5, 0.5, 0.5)
        pangolin.DrawPoints(self.state[1])
        
        # draw bbox_points
        gl.glPointSize(1)
        gl.glColor3f(1.0, 0.0, 1.0)
        pangolin.DrawPoints(self.state[2])
	
        
        pangolin.FinishFrame()

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)
        bbox_points = np.array(self.bbox_points)
        self.q.put((poses, points,bbox_points))

# 利用相机内参对角点的像素坐标进行归一化
def normalize(K, pts):
    Kinv = np.linalg.inv(K)
    # turn [[x,y]] -> [[x,y,1]]
    add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    return norm_pts


class Frame(object):
    idx = 0
    last_kps, last_des, last_pose = None, None, None

    def __init__(self, image):
        """
        只要一经初始化，Frame 就会把上一帧的信息传递给下一帧
        """
        Frame.idx += 1

        self.image = image
        self.idx   = Frame.idx
        self.last_kps  = Frame.last_kps
        self.last_des  = Frame.last_des
        self.last_pose = Frame.last_pose

# 提取 orb 角点
def extract_points(frame):
    orb = cv2.ORB_create()
    image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    # detection corners
    pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.025, minDistance=4)
    # extract features
    kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts]
    kps, des = orb.compute(image, kps)

    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    return kps, des

# 当前帧的角点和上一帧的进行配准
def match_points(frame,bbox,bbox2):
    bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bfmatch.knnMatch(frame.curr_des, frame.last_des, k=2)
    match_kps, idx1, idx2 = [], [], []

    for m,n in matches:
        if m.distance < 0.75*n.distance: # 0.75阈值过滤掉相似度接近的模糊匹配

            p1 = frame.curr_kps[m.queryIdx]
            p2 = frame.last_kps[m.trainIdx]

            '''for i in range(0,len(bbox[:,0])):
                if p1[0]>bbox[i,1] and p1[0]<bbox[i,1]+bbox[i,3] and p1[1]>bbox[i,2] and p1[1]<bbox[i,2]+bbox[i,4]: 
                    break
            for j in range(0,len(bbox2[:,0])):
                if p2[0]>bbox2[j,1] and p2[0]<bbox2[j,1]+bbox2[j,3] and p2[1]>bbox2[j,2] and p2[1]<bbox2[j,2]+bbox2[j,4]: 
                    break'''
            if (abs(p1[0]-p2[0])+abs(p1[1]-p2[1]))<40+80*abs(1-p1[0]/320): # 动态距离阈值，图像中心运动较小，约束更严格。边缘运动较大，放宽阈值。
                '''if p1[0]>bbox[i,1] and p1[0]<bbox[i,1]+bbox[i,3] and p1[1]>bbox[i,2] and p1[1]<bbox[i,2]+bbox[i,4]: 
                    if not(p2[0]>bbox2[j,1] and p2[0]<bbox2[j,1]+bbox2[j,3] and p2[1]>bbox2[j,2] and p2[1]<bbox2[j,2]+bbox2[j,4]): 
                        continue'''      
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                match_kps.append((p1, p2))
            '''idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            match_kps.append((p1, p2))'''
    assert len(match_kps) >= 8

    frame.curr_kps = frame.curr_kps[idx1]
    frame.last_kps = frame.last_kps[idx2]

    return match_kps

# 八点法对本质矩阵求解
def fit_essential_matrix(match_kps):
    global K
    match_kps = np.array(match_kps)

    # 使用相机内参对角点坐标归一化
    norm_curr_kps = normalize(K, match_kps[:, 0])
    norm_last_kps = normalize(K, match_kps[:, 1])

    # 求解本质矩阵和内点数据
    model, inliers = ransac((norm_last_kps, norm_curr_kps),
                            EssentialMatrixTransform,
                            min_samples=8,              # 最少需要 8 个点
                            residual_threshold=0.005,
                            max_trials=200)

    frame.curr_kps = frame.curr_kps[inliers]
    frame.last_kps = frame.last_kps[inliers]

    return model.params
'''
def check_Rt(frame,R1,R2,t1,t2):
    global K
    Rt1 = np.eye(4)
    Rt1[:3, :3] = R1
    Rt1[:3, 3] = t1
    
    Rt2 = np.eye(4)
    Rt2[:3, :3] = R2
    Rt2[:3, 3] = t1
    
    Rt3 = np.eye(4)
    Rt3[:3, :3] = R1
    Rt3[:3, 3] = t2
    
    Rt4 = np.eye(4)
    Rt4[:3, :3] = R2
    Rt4[:3, 3] = t2
    
    Rts= [Rt1,Rt2,Rt3,Rt4]
    last_pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    nGood=0
    nRt=0
    print(frame.last_pose)
    for i in range(0,4):
        curr_pose = np.dot(Rts[i], frame.last_pose)
        points4d1 = triangulate(frame.last_kps, frame.curr_kps, frame.last_pose, curr_pose)
        points4d1 = points4d1[check_points(points4d1)]
        
        if len(points4d1)>=nGood:
            nGood=len(points4d1)
            nRt=i
            
    #print(Rts[nRt])   
    return Rts[nRt]'''
    
    
# 从本质矩阵中分解出相机运动 R、t
def extract_Rt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)

    if np.linalg.det(U)  < 0: U  *= -1.0
    if np.linalg.det(Vt) < 0: Vt *= -1.0

    # 相机没有转弯，因此 R 的对角矩阵非常接近 diag([1,1,1])，表示无旋转
    R = (np.dot(np.dot(U, W), Vt))
    if np.sum(R.diagonal()) < 0: #np.linalg.det()
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]     # 相机一直向前，分量 t[2] > 0
    t /=np.linalg.norm(t)
    if t[2] < 0:
        t *= -1

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    #print(Rt)
    return Rt         # Rt 为从相机坐标系的位姿变换到世界坐标系的位姿
'''
    R1 = (np.dot(np.dot(U, W), Vt))
    if np.linalg.det(R1) < 0:
        R1 *= -1
        
    R2 = (np.dot(np.dot(U, W.T), Vt))
    if np.linalg.det(R2) < 0:
        R2 *= -1
    
    t1 = U[:, 2]
    t1 /= np.linalg.norm(t1)
    t2 = -t1
    return R1,R2,t1,t2'''
    


# 三角测量函数
def triangulate(pts1, pts2, pose1, pose2):
    global K
    pose1 = np.linalg.inv(pose1)            # 从世界坐标系变换到相机坐标系的位姿, 因此取逆
    pose2 = np.linalg.inv(pose2)
    
    pts1_cp =pts1
    pts2_cp =pts2

    pts1 = normalize(K, pts1)                 # 使用相机内参对角点坐标归一化
    pts2 = normalize(K, pts2)

    points4d = np.zeros((pts1.shape[0], 4))
    for i, (kp1, kp2) in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = kp1[0] * pose1[2] - pose1[0]
        A[1] = kp1[1] * pose1[2] - pose1[1]
        A[2] = kp2[0] * pose2[2] - pose2[0]
        A[3] = kp2[1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)         # 对 A 进行奇异值分解
        points4d[i] = vt[3]

    points4d /= points4d[:, 3:]            # 归一化变换成齐次坐标 [x, y, z, 1]
    reproject = np.dot(np.dot(K,pose2[0:3]),points4d.T)
    reproject = reproject.T
    repro_idx = []

    for i in range(0,len(reproject[:,1])):
        reproject[i,:-1] /= reproject[i,-1]
        repro_e=np.sqrt(int(pts2[i,0]-reproject[i,0])^2+int(pts2[i,1]-reproject[i,1])^2)  # 计算重投影误差
        if repro_e<32:  # 32 是一个阈值，表示重投影误差小于 32 时认为是有效的
            repro_idx.append(i)
            
    repro_idx = np.array(repro_idx)
    points4d = points4d[repro_idx,:]
    pts1_cp = pts1_cp[repro_idx,:]
    pts2_cp = pts2_cp[repro_idx,:]
    return pts1_cp,pts2_cp,points4d

# 画出角点的运动轨迹
def draw_points(frame):
    for kp1, kp2 in zip(frame.curr_kps, frame.last_kps):
        u1, v1 = int(kp1[0]), int(kp1[1])
        u2, v2 = int(kp2[0]), int(kp2[1])
        cv2.circle(frame.image, (u1, v1), color=(0,255,0), radius=2)
        cv2.line(frame.image, (u1, v1), (u2, v2), color=(255,0,0))
    return None

#在视频中画出检测框，高亮检测框内的特征点
def draw_bbox(frame,bbox):
    for i in range(len(bbox[:,0])):
        for kp1 in frame.curr_kps:
            u1, v1 = int(kp1[0]), int(kp1[1])
            if u1>bbox[i,1] and u1<bbox[i,1]+bbox[i,3] and v1>bbox[i,2] and v1<bbox[i,2]+bbox[i,4]: 
                cv2.circle(frame.image, (u1, v1), color=(255,0,255), radius=2)
        cv2.rectangle(frame.image,(int(bbox[i,1]),int(bbox[i,2])),(int(bbox[i,1]+bbox[i,3]),int(bbox[i,2]+bbox[i,4])), color=(255,0,0),thickness=1)
    return None
    
# 筛选角点
def check_points(points4d):
    # 判断3D点是否在摄像头前方
    good_points = points4d[:, -2] > 0
    # parallax、重投投影误差筛选等等 ....
    return good_points

def process_frame(frame):
    # 提取当前帧的角点和描述子特征
    frame.curr_kps, frame.curr_des = extract_points(frame)
    # 将角点位置和描述子通过类的属性传递给下一帧作为上一帧的角点信息
    Frame.last_kps, Frame.last_des = frame.curr_kps, frame.curr_des

    if frame.idx == 1:
        # 设置第一帧为初始帧，并以相机坐标系为世界坐标系
        frame.curr_pose = np.eye(4)
        points4d = [[0,0,0,1]]      # 原点为 [0, 0, 0] , 1 表示颜色
        points4d_draw =[[0,0,0,1]]
    else:
        bbox = np.loadtxt('CRH_1.txt')
        bbox_frame = bbox[bbox[:,0]==frame.idx+30,1:6]
        bbox_frame[:,1:]/=4 # 由视频分辨率和本程序规定的分辨率决定
        
        bbox_frame2 = bbox[bbox[:,0]==frame.idx+31,1:6] # bbox_frame_last
        bbox_frame2[:,1:]/=4 # 由视频分辨率和本程序规定的分辨率决定
        
        # 角点配准, 此时会用 RANSAC 过滤掉一些噪声
        match_kps = match_points(frame,bbox_frame,bbox_frame2)
        
        # 使用八点法拟合出本质矩阵
        essential_matrix = fit_essential_matrix(match_kps)
        #print("---------------- Essential Matrix ----------------")
        #print(essential_matrix)
        # 利用本质矩阵分解出相机的位姿 Rt
        Rt = extract_Rt(essential_matrix)
        #R1,R2,t1,t2=extract_Rt(essential_matrix)
        #Rt=check_Rt(frame,R1,R2,t1,t2)
        # 计算出当前帧相对于初始帧的相机位姿
        frame.curr_pose = np.dot(Rt, frame.last_pose)   
        # 三角测量获得角点的深度信息
        frame.last_kps, frame.curr_kps,points4d = triangulate(frame.last_kps, frame.curr_kps, frame.last_pose, frame.curr_pose)
        
        draw_points(frame)
        
        draw_bbox(frame,bbox_frame)
        print("frame: {}, curr_des: {}, last_des: {}, match_kps: {}, good_kps: {}".
            format(frame.idx, len(frame.curr_des), len(frame.last_des), len(match_kps),len(frame.last_kps)))
        flag=0
        points4d_idx=[]
        for i in range(len(bbox_frame[:,0])):
            temp_idx=0
            for kp1 in frame.curr_kps:
                u1, v1 = int(kp1[0]), int(kp1[1])
                if u1>=bbox_frame[i,1] and u1<=bbox_frame[i,1]+bbox_frame[i,3] and v1>=bbox_frame[i,2] and v1<=bbox_frame[i,2]+bbox_frame[i,4]:
                    if flag==0:
                        points4d_bbox=[np.hstack(([bbox_frame[i,0]],points4d[temp_idx]))]
                        flag=1
                    else:
                        points4d_bbox=np.vstack((points4d_bbox,np.hstack(([bbox_frame[i,0]],points4d[temp_idx]))))
                    points4d_idx.append(temp_idx)
                temp_idx +=1
                
        
        #将points4d拆分
        points4d_draw=points4d[points4d_idx]
        points4d =np.delete(points4d,points4d_idx,axis=0)
        
        good_pt4d = check_points(points4d)
        points4d = points4d[good_pt4d]
        
        if len(points4d_idx)>0:
            good_pt4d = check_points(points4d_bbox)
            points4d_bbox = points4d_bbox[good_pt4d]
        
            good_pt4d = check_points(points4d_draw)
            points4d_draw = points4d_draw[good_pt4d]
        
        #pose_txt.write(str(frame.idx)+'\n')
         
            np.savetxt(pcd_txt1,points4d,newline='\n')
            np.savetxt(pcd_txt2,points4d_bbox,newline='\n')
            np.savetxt(pose_txt,frame.curr_pose,newline='\n')  
    mapp.add_observation(frame.curr_pose, points4d,points4d_draw)     # 将当前的 pose 和点云放入地图中
    # 将当前帧的 pose 信息存储为下一帧的 last_pose 信息
    Frame.last_pose = frame.curr_pose
    return frame


if __name__ == "__main__":
    # camera intrinsics
    W, H = 635, 360  # 635,360; 640
    F = 656  # 656;554
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    mapp = Map(1024, 768)  # 构建地图

    cap = cv2.VideoCapture("/home/lcy/CRH_part1.mp4")

    pcd_txt1 = open("/home/lcy/pcd.txt", 'a')
    pcd_txt1.truncate(0)
    pcd_txt2 = open("/home/lcy/pcd_bbox.txt", 'a')
    pcd_txt2.truncate(0)
    pose_txt = open("/home/lcy/pose.txt", 'a')
    pose_txt.truncate(0)
    bbox = np.loadtxt('/home/lcy/CRH_1.txt')

    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            image = cv2.resize(image, (W, H))
            frame = Frame(image)
            if frame.idx % 3 == 1 and len(bbox[bbox[:, 0] == frame.idx, :]) > 0:
                frame = process_frame(frame)
            else:
                cv2.waitKey(0)
                break
        frame.image = cv2.resize(frame.image, (W * 2, H * 2))
        cv2.imshow("slam", frame.image)
        mapp.display()
        cv2.waitKey(0)
        # if frame.idx >450: break
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
