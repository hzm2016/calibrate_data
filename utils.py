# -*- coding: utf-8 -*
import matplotlib.pyplot as plt   
import numpy as np  
import math   
import os   
import copy as cp  

import ctypes   
import time    
import glob    
import scipy   
import argparse    

import tkinter as tk       
import tkinter.font as tkfont        
import socket    
import sys   
import zmq   
import ctypes    

from robot_teaching.plot_path.plot_path_main import *      


# Function to simulate the robot movement (replace this with actual data from your robot)
def simulate_robot_movement():
    # Example data for the robot's path (replace this with actual data from your robot)
    # Here, we use simple lists to represent the x and y coordinates of the robot's position over time.
    x_coords = [0, 50, 100, 150, 200]
    y_coords = [0, 25, 100, 75, 50]

    return x_coords, y_coords


def plot_word_path(
    real_data=None,    
    font_name='plannar_motion'   
):  
    font_size = 15  
    plt.figure(figsize=(5, 5))   
    axes = plt.gca()     
    plt.plot(real_data[:, 1], real_data[:, 0], linewidth=3)     

    axes.set_xlim([-0.3, 0.3])     
    axes.set_ylim([0., 0.6])    
    plt.xlabel('Time[s]', fontsize=font_size)      
    plt.ylabel('Output', fontsize=font_size)     
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)     

    axes.invert_xaxis() 

    plt.tight_layout()   
    plt.legend() 
    plt.savefig(font_name + '.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  


class RobotPathVisualizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()

        # Simulate the robot movement and get the path coordinates
        self.x_coords, self.y_coords = simulate_robot_movement()

        # Draw the initial robot position
        self.robot = self.canvas.create_oval(0, 0, 10, 10, fill='blue')
        self.update_robot_position(0, 0)  

        # Draw two 
        # Create the first dotted line
        line_1_x1, line_1_y1, line_1_x2, line_1_y2 = 50, 100, 350, 100
        self.line_1 = self.canvas.create_line(line_1_x1, line_1_y1, line_1_x2, line_1_y2, dash=(2, 2))

        # Create the second dotted line
        line_2_x1, line_2_y1, line_2_x2, line_2_y2 = 50, 150, 350, 150
        self.line_2 = self.canvas.create_line(line_2_x1, line_2_y1, line_2_x2, line_2_y2, dash=(4, 4))  


    def update_robot_position(self, x, y):  
        # Update the robot's position in real-time
        # for x, y in zip(self.x_coords, self.y_coords):
        #     self.canvas.coords(self.robot, x, y, x + 50, y + 50)
        #     self.root.update()  # Update the tkinter window
        #     self.root.after(1000)  # Pause for 1000 milliseconds (1 second)
        self.canvas.coords(self.robot, x, y, x + 50, y + 50)
        self.root.update()  # Update the tkinter window
        self.root.after(1)  # Pause for 1000 milliseconds (1 second)


def main_lines():
    root = tk.Tk()
    root.title("Dotted Lines")
    WIDTH = 700 
    HEIGHT = 700  

    canvas = tk.Canvas(root, width=700, height=700, bg='white')
    canvas.pack()

    # Create the first dotted line
    x1, y1, x2, y2 = 0, HEIGHT/2, WIDTH, HEIGHT/2
    line1 = canvas.create_line(x1, y1, x2, y2, dash=(2, 2), width=2, fill='black')

    # Create the second dotted line
    x1, y1, x2, y2 = WIDTH/2, 0, WIDTH/2, HEIGHT
    line2 = canvas.create_line(x1, y1, x2, y2, dash=(4, 4), width=2, fill='black')

    root.mainloop()


class RobotPathVisualizerFullScreen:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)  # Enable fullscreen mode

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='white')
        self.canvas.pack()

        # Simulate the robot movement and get the path coordinates
        self.x_coords, self.y_coords = simulate_robot_movement()

        # Draw the initial robot position
        self.robot = self.canvas.create_oval(0, 0, 50, 50, fill='blue')  

        self.update_robot_position()

    def update_robot_position(self):
        # Update the robot's position in real-time
        for x, y in zip(self.x_coords, self.y_coords):
            self.canvas.coords(self.robot, x, y, x + 10, y + 10)
            self.root.update()  # Update the tkinter window
            self.root.after(1000)  # Pause for 1000 milliseconds (1 second)


def main():
    root = tk.Tk()
    root.title("Robot Path Visualizer")

    # Set the default font size for the window title (optional)
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(size=2000)

    app = RobotPathVisualizerFullScreen(root)  
    root.mainloop()


def ctl_airhockey_main(): 
    up_left = np.array([0.63467476, 0.26104726, 0.12489759])     
    up_right = np.array([0.62749946, -0.2615156, 0.12834424])    
    down_left = np.array([0.29211079, 0.27437398, 0.12084902])    
    down_right = np.array([0.28580701, -0.25766315, 0.12225445])    

    # airhockey:: [0.5494716  0.01557979 0.12696136]   


def obtain_para_input(args):   
    args.ctl_mode = "zero_force"    
    ones = np.array([1.0, 1.0, 1.0])     
    if args.mode==0:      
        args.ctl_mode = "zero_force"      
        # K_p = np.array([1850.0, 1850.0, 1850.0])           
        # K_d = np.array([85.5, 85.5, 85.5])       
        
        # K_p = np.array([205.0,205.0,205.0])           
        # K_d = np.array([5.55, 5.55, 5.55])     
        # K_p = 1850.0 * ones            
        # K_d = 85.0 * ones       
        
        # K_p = 2250.0 * ones            
        # K_d = 125.5 * ones     
        
        # K_p = 800.0 * ones            
        # K_d = 50.0 * ones         
        
        K_p = 1600.0 * ones        
        K_d = 100.0 * ones       
          
        # K_i = np.array([0.0,0.0,0.0])      
        # K_v = np.array([0.008,0.008,0.008])        
        # K_p_theta = np.array([10,10,10])        
        # K_d_theta = K_p_theta * 1.0/70.0        
          
        # K_p = np.array([1100.0, 1100.0, 1100.0])         
        # K_d = np.array([20.5, 20.5, 20.5])         
        K_i = np.array([0.0,0.0,0.0])      
        K_v = np.array([0.002,0.002,0.002])    
            
        K_p_theta = np.array([55, 55, 55])    
        K_d_theta = K_p_theta * 1.0/65.0      
        
    elif args.mode==1:     
        args.ctl_mode = "motion"     
        # K_p = np.array([45,45,45])     
        # K_d = K_p * 1.0/50.0     
        K_p = np.array([40,40,40])      
        K_d = K_p * 1.0/50.0       
        # K_p = np.array([30,30,30])         
        # K_d = K_p * 1.0/50.0       
        # K_p = np.array([20,20,20])      
        # K_d = K_p * 1.0/50.0      
        # K_d = np.array([0.06,0.06,0.06])      
        K_i = np.array([0.0,0.0,0.0])     
        # K_v = np.array([0.005,0.005,0.005])      
        K_v = 0.007 * ones 
        # /////// parameters    
        # K_p = np.array([0.50,0.50,0.50])     
        # K_d = np.array([0.01,0.01,0.01])     
        # K_i = np.array([0.0,0.0,0.0])    
        # K_v = np.array([0.01,0.01,0.01])    
        # K_p_theta = np.array([10,10,10])      
        # K_d_theta = K_p_theta * 1.0/70.0       
        # K_p_theta = np.array([80,80,80])      
        # K_d_theta = K_p_theta * 1.0/220.0       
        # K_p_theta = 140 * ones  
        # K_d_theta = K_p_theta * 1.0/700.0   
        
        # K_p_theta = np.array([250, 250, 250])
        # K_d_theta = K_p_theta * 1.0/600.0   
        
        # # # # blue spring 
        # K_p_theta = np.array([70, 70, 70])    
        # K_d_theta = K_p_theta * 1.0/25.0    
        
        # # tracking very well
        # K_p_theta = np.array([95, 95, 95])     
        # K_d_theta = K_p_theta * 1.0/40.0     
        
        K_p_theta = np.array([50, 50, 50])    
        K_d_theta = K_p_theta * 1.0/40.0     

        # ///////////////////////// normal sea control    
        # // K_p_test<<30,30,30;     
        # // K_d_test<<1.0,1.0,1.0;    
        # // K_v<<0.012,0.012,0.012;     
        # // K_p_test<<35,35,35;     
        # // K_d_test<<1.2,1.2,1.2;      
        # // K_v<<0.012,0.012,0.012;      
        
        # # ///// friction calculation motion 
        fric_1 = 0.35 * ones;     
        fric_2 = 0.7 * ones;     
    elif args.mode==2:   
        args.ctl_mode = "assistive_force"    
        # K_p = np.array([3000.0,2200.0,2200.0])     
        K_p = np.array([2000.0,2000.0,1.0])    
        K_d = K_p * 1.0/60.0   
        # K_d = K_p * 1.0/25.0        
        # K_p = np.array([900.0,900.0,900.0])     
        # # K_d = K_p * 1.0/60.0   
        # K_d = K_p * 1.0/30.0       
        # K_d = np.array([0.2,0.2,0.2])        
        K_i = np.array([0.000,0.000,0.000])       
        K_v = np.array([0.008,0.008,0.008])       
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0       
    elif args.mode==3:   
        args.ctl_mode = "force_fre"    
        K_p = np.array([2200.0,2200.0,2200.0])     
        K_d = K_p * 1.0/40.0   
        # K_d = K_p * 1.0/25.0        
        # K_p = np.array([900.0,900.0,900.0])     
        # # K_d = K_p * 1.0/60.0   
        # K_d = K_p * 1.0/30.0       
        # K_d = np.array([0.2,0.2,0.2])        
        K_i = np.array([0.000,0.000,0.000])       
        K_v = np.array([0.008,0.008,0.008])       
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0       
    else:   
        K_p = np.array([45.0,45.0,45.0])      
        K_d = np.array([0.55,0.55,0.55])        
        K_i = np.array([0.0,0.0,0.0])     
        K_v = np.array([0.008,0.008,0.008])    
        
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0          
    
    # # ///// friction calculation motion    
    # fric_1 = 0.35 * ones;     
    # fric_2 = 0.7 * ones;     
    
    fric_1 = 0.0 * ones;     
    fric_2 = 0.1 * ones;     

    fric_3 = 0.55 * ones;     
    fric_4 = 0.3 * ones;     

    para_input = np.zeros((11, 3))     
    para_input[0, :] = cp.deepcopy(K_p)     
    para_input[1, :] = cp.deepcopy(K_d)     
    para_input[2, :] = cp.deepcopy(K_i)     
    para_input[3, :] = cp.deepcopy(K_v)     
    para_input[4, :] = cp.deepcopy(K_p_theta)       
    para_input[5, :] = cp.deepcopy(K_d_theta)       
    para_input[6, :] = cp.deepcopy(fric_1)       
    para_input[7, :] = cp.deepcopy(fric_2)       
    para_input[8, :] = cp.deepcopy(1/20 * ones)    
    para_input[9, :] = cp.deepcopy(fric_3)       
    para_input[10, :] = cp.deepcopy(fric_4)            
    return args, para_input   


def cal_iter_path_matrix(
    N=500  
):  
    # N = inter_force.shape[0]      
    G = np.zeros((N, N))        
    I = np.identity(N)     
    Z = np.zeros((4, N))  # for smoothness   
    Z[0, 0] = 1     
    Z[1, 1] = 1    
    Z[2, N-2] = 1       
    Z[3, N-1] = 1     
    
    Q = np.zeros((N+3,N))  
    for i in range(N): 
        Q[i, i] = 1 
        Q[i+1, i] = -3
        Q[i+2, i] = 3
        Q[i+3, i] = -1 
    R = np.zeros((N, N))    
    R = Q.T.dot(Q)    
    # print("R :", R)   
    G = (I - np.linalg.inv(R).dot(Z.T).dot(np.linalg.inv(Z.dot(np.linalg.inv(R)).dot(Z.T))).dot(Z)).dot(np.linalg.inv(R))
    G = G/np.linalg.norm(G)    
    # print("G :", G.shape)     
    # print("inter_force", inter_force)     
    
    # flag = "_baletral_" + control_mode + "_demo_" + args.flag + "_vr"   
    # mean_list = []     
    # std_list = []    
    # // ori stiff data   
    # ori_tau_data = np.tile(np.array([0.0, 0.0, 0.0]), (args.num, 1))        
    # ori_stiff_data = np.tile(np.array([0.0, 0.0, 0.0]), (args.num, 1))          
    # ori_damping_data = np.tile(np.array([0.0, 0.0, 0.0]), (args.num, 1))              
    # np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_0.txt",  ori_tau_data)      
    # np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_0.txt",  ori_stiff_data)       
    # np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_0.txt",  ori_damping_data)   
    
    # # // ori stiff data   
    # ori_stiff_data = np.tile(K_p, (args.num, 1))      
    # ori_damping_data = np.tile(K_d, (args.num, 1))   
    
    return G   


############################################
HOST, PORT = '127.0.0.1', 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))    
for x in range(0, 10000):  
    print("Step 1")   
    s.send(b'Hello')   
    print("Step 2")   
    print(str(s.recv(1000)))     
    print(x)   
 

def communication_test():   
    context = zmq.Context()    
    socket = context.socket(zmq.REQ)     

    # Connect to the server
    socket.connect("tcp://localhost:5555")

    # Send a request
    request = b"Hello from Python!"
    socket.send(request)
    print("Sent: %s" % request)  

    # Wait for the response
    response = socket.recv()
    print("Received: %s" % response.decode())

    # Clean up
    socket.close()
    context.term()
    return 1    
