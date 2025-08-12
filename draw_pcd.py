import cv2
import pangolin
import numpy as np
import OpenGL.GL as gl
import os
import pandas as pd

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
        self.pole_points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def add_observation(self, pose, points,bpoints,ppoints):
        self.poses.append(pose)
        for point in points:
            self.points.append(point)
        for point in bpoints:
            self.bbox_points.append(point)
        for point in ppoints:
            self.pole_points.append(point)

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
        
	# draw pole_points
        gl.glPointSize(2)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawPoints(self.state[3])
        
        # draw pole_lines
        
        
        pangolin.FinishFrame()

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)
        bbox_points = np.array(self.bbox_points)
        pole_points = np.array(self.pole_points)
        self.q.put((poses, points,bbox_points,pole_points))

def remove_outlier(all_data):
    df = pd.DataFrame(all_data, columns=['col1', 'col2', 'col3'])
    df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    cols = df.columns  # 获得数据框的列名
    for col in cols:  # 循环读取每列
        df_col = df[col]  # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分
        # 判断Z-score得分是否大于1,整体来看标准差很大，所以Z-score卡得紧一点
        df_zscore[col] = z_score.abs() > 2.2

    df_drop_outlier = df[df_zscore['col1'] == False]
    all_data = df_drop_outlier.values
    
    return all_data
    
def check_pole(pole_points):#检测重叠，极相邻的视作同一根
    pole_checked=[]
    co_flag=0 #重叠的组数
    curr_num=0 #当前要标记的是第几组
    all_num=0 #涉及重叠的电线杆总数
    all_idx=np.zeros((1,len(pole_points[:,0]))) #重叠位置标记，重叠的组内数字相同，不涉及重叠的为0
    for i in range(0,len(pole_points[:,0])):
        flag=1
        for j in range(i+1,len(pole_points[:,0])): #双重for,计算所有电线杆的距离
            dis=abs(pole_points[j,0]-pole_points[i,0])+abs(pole_points[j,2]-pole_points[i,2]) #计算距离
            if dis <4: #相邻距离的阈值
                if flag: #遇到新的重叠组
                    co_flag+=1
                    curr_num+=1
                    flag=0
                if all_idx[0,j]>0: # 减去可能重复算的
                    curr_num=all_idx[0,j]
                    co_flag -=1
                all_idx[0,j]=curr_num #标记数字
                all_idx[0,i]=curr_num
                all_num+=1
                     
    for i in range(0,len(all_idx)):
        for j in range(1,co_flag+1):
            pole_checked.append(pole_points[all_idx[0]==j,-2])   #得到一个列表，列表元素是数组，考虑到重叠组间数量不同 ，不采用二维数组     
    return pole_checked,all_num,co_flag
    
def get_pole_info(pole_data): #获得每个电线杆的位置，点云归属信息
    pole_points=[]
    pole_line=[]

    for j in range(0,int(max(pole_data[:,0]))):
        pole=pole_data[pole_data[:,0]==j+1,1:-1]
        if len(pole)==0:
            continue
        else:
            pole=remove_outlier(pole)    	            
            pole_x=np.mean(pole[:,0])
            pole_y=np.mean(pole[:,1])
            pole_z=np.mean(pole[:,2])
            pole_points.append([pole_x,pole_y,pole_z,j+1,len(pole[:,0])]) #点云中心点作为pole_points【位置，归属电线杆序号，点云数量】
    pole_points=np.array(pole_points)
    coincide,_,_=check_pole(pole_points)#极相邻的视作同一根
    #print(coincide)    
    #pole_line的获取
    range_of_poles=np.array([0]*3)
    for j in range(0,len(pole_points[:,0])):
        for k in range(0,len(coincide)):
            curr_coincide=coincide[k]
            if len(np.where(pole_points[j,-2]==curr_coincide)[0])>0:
                logical_idx=pole_data[:,0]==curr_coincide[0]
                for i in range(1,len(curr_coincide)):
                    logical_idx=np.logical_or(pole_data[:,0]==curr_coincide[i],logical_idx)
                pole=pole_data[logical_idx,1:-1]
                break
        else:
            pole=pole_data[pole_data[:,0]==pole_points[j,-2],1:-1]            
        pole=remove_outlier(pole)
        pole_x=np.mean(pole[:,0])
        pole_y=np.mean(pole[:,1])
        pole_z=np.mean(pole[:,2])

        range_of_poles[0]+=np.percentile(pole[:,0],75)-np.percentile(pole[:,0],25)
        
        range_of_poles[1]+=np.percentile(pole[:,1],75)-np.percentile(pole[:,1],25)
        range_of_poles[2]+=np.percentile(pole[:,2],75)-np.percentile(pole[:,2],25)
        for k in range(0,len(pole[:,0])):
            pole_line.append([pole_x,np.clip(pole[k,1],pole_y-8,pole_y+8),pole_z])
                   	      
    print(str(range_of_poles/len(pole_points[:,0])))	            
    pole_line=np.array(pole_line)
    #重新修正pole_points，合并重叠
    for j in range(0,len(coincide)):
        curr_coincide=coincide[j]
        logical_idx=pole_points[:,-2]==curr_coincide[0]#用np.logical的方法找索引
        for i in range(1,len(curr_coincide)):
            logical_idx=np.logical_or(pole_points[:,-2]==curr_coincide[i],logical_idx)
            
        sum_points=sum(pole_points[logical_idx,-1])
        co_x=sum(pole_points[logical_idx,0]*pole_points[logical_idx,-1])
        co_y=sum(pole_points[logical_idx,1]*pole_points[logical_idx,-1])
        co_z=sum(pole_points[logical_idx,2]*pole_points[logical_idx,-1])
        #修改重复组的第一行
        pole_points[pole_points[:,-2]==curr_coincide[0],:3]=[co_x,co_y,co_z]/sum_points
        pole_points[pole_points[:,-2]==curr_coincide[0],-2]=curr_coincide[0]
        pole_points[pole_points[:,-2]==curr_coincide[0],-1]=sum_points
        for k in range(1,len(curr_coincide)):#删除其他重复的行
            pole_points=np.delete(pole_points,np.where(pole_points[:,-2]==curr_coincide[k])[0],axis=0) 
                
    print('The number of poles:'+str(len(pole_points[:,0])))
    
    
    pole_points=np.hstack((pole_points[:,:-2],np.ones((len(pole_points[:,0]),1))))
    pole_line=np.hstack((pole_line,np.ones((len(pole_line[:,0]),1))))
    return pole_points,pole_line
    
def correct_interval(pole_points):#修正间隔
    new_points=pole_points
    points_l=pole_points[pole_points[:,0]<0,:]
    #print(points_l)
    num_l=len(points_l[:,0])
    count_l=0
    near_l=min(points_l[:,2])
    far_l=max(points_l[:,2])
    inteval_l=(far_l-near_l)/num_l
    
    points_r=pole_points[pole_points[:,0]>0,:]
    #print(points_r)
    num_r=len(points_r[:,0])
    count_r=0
    near_r=min(points_r[:,2])
    far_r=max(points_r[:,2])
    inteval_r=(far_r-near_r)/num_r    
    
    for i in range(0,len(pole_points[:,0])):
        if pole_points[i,1]>0 :
            new_points[i,2]=0.5*(count_r*inteval_r+near_r)+0.5*pole_points[i,2]
            count_r+=1
        if pole_points[i,1]<0 :
            new_points[i,2]=0.5*(count_l*inteval_l+near_l)+0.5*pole_points[i,2]
            count_l+=1
            
    #print(new_points)
    return new_points    
        
'''def correct_distance(pole_points,pose_data):#修正到铁路的距离'''

def projection(pole_points,pose_data,pole_data): #计算点云的面投影
    pole_plane_points=[]
    for i in range(0,len(pole_points[:,0])):
        pole=pole_data[pole_data[:,0]==i+1,1:-1]
        pole=remove_outlier(pole)
        min_dis=60000
        B_point=[]
        for j in range(0,len(pose_data[:,0])//4): #找最近的相机位置，作为切点
            distance=sum((pose_data[4*j:4*j+3,-1]-pole_points[i,:-1].T)**2)**0.5
            if distance<min_dis:
                min_dis=distance
                B_point=pose_data[4*j:4*j+3,-1]
        #print(B_point)
        #print(pole_points[i,:-1])
        for k in range(0,len(pole[:,0])): #已知两点（直线），求其他点到过该直线的铅垂面的投影
            pole_x=np.mean(pole[:,0])
            pole_y=np.mean(pole[:,1])
            pole_z=np.mean(pole[:,2])
            
            x1=B_point[0]
            z1=B_point[2]
            x2=pole_points[i,0]
            z2=pole_points[i,2]
            x3=pole[k,0]
            z3=pole[k,2]
            
            lmbda=((x2-x3)*(x2-x1)+(z2-z3)*(z2-z1))/((x1-x2)**2+(z1-z2)**2) #由等和线推导得到的公式
            px=lmbda*(x1-x2)+x2
            pz=lmbda*(z1-z2)+z2
            
            min_dis=2 #min_dis是阈值，本行也可注释，即为自适应阈值
            proj_x=np.clip(px,pole_x-min_dis/2,pole_x+min_dis/2)
            proj_y=np.clip(pole[k,1],pole_y-8,pole_y+8)
            proj_z=np.clip(pz,pole_z-min_dis/2,pole_z+min_dis/2)
            
            pole_plane_points.append([proj_x,proj_y,proj_z])
        
    pole_plane_points=np.array(pole_plane_points)
    pole_plane_points=np.hstack((pole_plane_points,np.ones((len(pole_plane_points[:,0]),1))))#修复为齐次坐标[[x,y,z,1]]
    return pole_plane_points                
    
    
if __name__ == "__main__":
    # camera intrinsics
    W, H = 635, 360 #854,480
    F = 656 #739 #658
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    mapp = Map(1024, 768)    # 构建地图
    cap = cv2.VideoCapture("/home/syl/movie_003.mp4")
    pcd_data = np.loadtxt('/home/syl/pcd.txt')#全部三维地图点云信息
    pole_data = np.loadtxt('/home/syl/pcd_bbox.txt')#所有电线杆的三维点云信息
    pose_data =np.loadtxt('/home/syl/pose.txt')#相机位姿
    
    process_txt=open("/home/syl/processed.txt",'a')
    process_txt.truncate(0)
    
    for i in range(0,len(pose_data[:,0])//4):
    	ret,image=cap.read()
    	image=cv2.resize(image,(W,H))
    	cv2.imshow('video',image)
    	if i==0: #相机位姿一帧一帧画，点云一次性画
    	    pole_points,pole_line=get_pole_info(pole_data) #获取所有电线杆点云 、线投影
    	    '''#修正电线杆之间的间隔
    	    pole_points=correct_interval(pole_points)
    	    #修正pole_point 与相机位姿曲线的距离
    	    pole_points=correct_distance(pole_points,pose_data)'''
    	    
    	    pole_plane_points= projection(pole_points,pose_data,pole_data)# 相机曲线过每个电线杆中心点的法平面，获取点云的面投影
            
    	    mapp.add_observation(pose_data[4*i:4*(i+1),:],[[0,0,0,1]],[[0,0,0,1]],pole_line) 
    	    #mapp.add_observation(pose_data[4*i:4*(i+1),:],pcd_data,pole_data[:,1:],pole_points) 
    	    np.savetxt(process_txt,pole_points,newline='\n')
    	    #第4个参数可以是pole_points,pole_line,pole_plane_points 
    	else:
    	    mapp.add_observation(pose_data[4*i:4*(i+1),:],[[0,0,0,1]],[[0,0,0,1]],[[0,0,0,1]])
    	
    	mapp.display()
    	cv2.waitKey(1)
    else:
        cv2.waitKey(0)
    
    
    
    
    
    
