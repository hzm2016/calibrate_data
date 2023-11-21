import cv2    
import glob    
import numpy as np    
import argparse    
import os    


########################## image process #########################
def obtain_single_image(img_path=None):    
    cap = cv2.VideoCapture(-1)     
    print(cap.isOpened())    # True   
    
    # read it using mjpg streamer
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    
    cap.set(cv2.CAP_PROP_FPS, 180)    

    ret, frame = cap.read()    
    print("frame shape :", frame.shape)          
    cv2.imshow("calibration :", frame)        
        
    cv2.imwrite(img_path, frame)        

    # # extract_green_circle(frame=frame, ball_color=ball_color, color_dist=color_dist)   

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break   
    cv2.waitKey(6000)   
    
    # Release the camera capture object
    cap.release()  
    

def capture_imgs(args=None):     
    args.writing_img_path = args.word_writing_path + '/' + args.font_name + '/' + 'stroke_' + str(args.index) + '/' + args.user_name + '/' 
    if not os.path.exists(args.writing_img_path):                
        os.makedirs(args.writing_img_path)        

    args.writing_img_path = args.writing_img_path + '/' + 'writing_' + str(args.img_index) + '.png'  
    # args.writing_img_path = args.writing_img_path + '/' + 'evaluation_' + str(args.img_index) + '.png'  

    obtain_single_image(img_path=args.writing_img_path)       
    
    
if __name__ == "__main__":  
    print('########## 一、开始内参标定 ##########')  
    parser = argparse.ArgumentParser()     
    parser.add_argument('--img_path', type=str, default="calibration_3", help='choose mode first !!!!')  
    
    args = parser.parse_args()   
    
    obtain_single_image(img_path="./test_main/data/" + args.img_path + ".png")    
    
    
    
    