import numpy as np  

np.set_printoptions(suppress=True)    
import cv2
import os  
import pybullet as p     
import copy as cp 

from scipy.io import savemat   


def calibrate(): 
    # Define the number of corners in the chessboard pattern
    num_corners_x = 7   
    num_corners_y = 6     

    # Define the size of each square in the chessboard pattern (in mm)
    square_size = 25    

    # Define the termination criteria for the corner detection algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Create an array to store the object points and image points from all the images
    obj_points = []   
    img_points = []   

    # Define the object points for the chessboard pattern
    objp = np.zeros((num_corners_x*num_corners_y, 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners_x,0:num_corners_y].T.reshape(-1,2)*square_size

    # Initialize the camera capture object
    cap = cv2.VideoCapture(0)

    # read it using mjpg streamer
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  

    while True:
        # Capture a frame from the camera 
        ret, frame = cap.read()   
        
        print(frame.shape)   

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the corners in the chessboard pattern
        ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

        # If corners are found, add the object points and image points to the arrays
        if ret == True:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            img_points.append(corners2)

            # Draw the corners on the image
            cv2.drawChessboardCorners(frame, (num_corners_x, num_corners_y), corners2, ret)
            print('find corners')

        # Display the image
        cv2.imshow('frame', frame)

        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera capture object
    cap.release()   

    # Destroy all windows
    cv2.destroyAllWindows()     

    # Calculate the camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)  

    # Print the camera matrix  
    print("Camera matrix:")   
    print(mtx)    
    

def cal_cart_pos(index, num_corners_x = 7, num_corners_y = 6, square_size=0.0235):      
    # # Create an array to store the object points and image points from all the images
    # obj_points = []   
    loaded_data = np.load(current_directory + '/test_main/data/calibration_' + str(index) + '.npz')  
    position = loaded_data['position']    
    ori = loaded_data['orientation']    
    print("positin :", position)     
    print("orientation :", ori)     
    
    T_ee = np.zeros((4, 4))    
    T_ee[:3, :3] = np.array(p.getMatrixFromQuaternion(np.array(ori[0]))).reshape(3, 3)      
    T_ee[:3, 3] = np.array(position[0][:3])   
    T_ee[3, 3] = 1.0       
    
    print("T_ee :\n", T_ee)  
      
    # Define the object points for the chessboard pattern
    objp = np.zeros((num_corners_x*num_corners_y, 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners_x,0:num_corners_y].T.reshape(-1,2)*square_size
    print("objp shape:", objp.shape)  
    print("objp_word shape:", objp.shape)    
    
    objp_word = np.zeros((num_corners_x*num_corners_y, 3), np.float32)  
    for i in range(num_corners_x * num_corners_y-1, -1, -1):      
        print("i :", i)     
        T = np.ones((4, 4))    
        T_t = np.zeros((4, 4))       
        T_t[:3, :3] = np.array(p.getMatrixFromQuaternion(np.array([0.0, 0.0, 0.0, 1.0]))).reshape(3, 3) 
        T_t[:3, 3] = np.array(objp[i, :3]) * np.array([1.0, -1.0, 1.0])        
        T_t[3, 3] = 1.0    
        # print("T_t :\n", T_t[:4, 3])     
        
        # T = cp.deepcopy(np.dot(T_t, T_ee))     
        # objp[i, :] = np.dot(T_ee, np.array([objp[i, 0], objp[i, 1], 0, 1]))[:3]
        objp_word[i, :2] = cp.deepcopy(np.dot(T_ee, T_t[:4, 3]))[:2]        
        # print("ee position : \n", cp.deepcopy(np.dot(T_ee, T_t)))       
        # print("ee position : \n", cp.deepcopy(np.dot(T_ee, T_t[:4, 3])))  
    
    print("objp :\n", objp_word.shape)              
    return objp, objp_word   
     
    
def cal_corners(start_index=1, use_num=3):       
    num_corners_x = 7   
    num_corners_y = 6   
     
    # # Define the size of each square in the chessboard pattern (in mm)
    # square_size = 25    
    
    # Define the termination criteria for the corner detection algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Create an array to store the object points and image points from all the images
    obj_points = []    
    img_points = []     
    for index in range(start_index, start_index+use_num):        
        objp, objp_word = cal_cart_pos(index, num_corners_x=7, num_corners_y=6, square_size=0.0235)    
        
        frame = cv2.imread('./test_main/data/calibration_' + str(index) + '.png')   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        # Find the corners in the chessboard pattern
        ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

        # If corners are found, add the object points and image points to the arrays
        if ret == True:   
            # obj_points.append(objp)    
            obj_points.append(objp_word)     
            
            # print("corners :", corners.shape)  
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)  
            print("corners2 :", corners2)     
            img_points.append(corners2)  

            # Draw the corners on the image
            cv2.drawChessboardCorners(frame, (num_corners_x, num_corners_y), corners2, ret)  
            print('find corners')  

        # Display the image
        cv2.imshow('frame', frame)  

        # Wait for a key press
        if cv2.waitKey(3000) & 0xFF == ord('q'): 
            break
        
    # Calculate the camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)  

    # Print the camera matrix   
    print("Camera matrix results ::")    
    print("ret :", ret)    
    print("mtx :", mtx)    
    print("rvecs :", rvecs)  
    print("tvecs :", tvecs)     
    
    print(np.array(img_points).shape)    
    
    # img_points = np.array([[[500, 400]]], dtype=np.float32)   
    
    # Use cv2.undistortPoints to get real-world points  
    undistorted_points = cv2.undistortPoints(img_points[0], mtx, dist, P=mtx)     
    # print("undistorted_points :", undistorted_points)  
    
    # print("obj points :", obj_points[0]) 
    # Use solvePnP to estimate the pose
    success, rotation_vector, translation_vector = cv2.solvePnP(obj_points[0], undistorted_points, mtx, dist)  

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)  
    
    # print("rotation_matrix :", rotation_matrix)  
    
    # # Specify the file path (replace 'output.mat' with your desired file name)
    # file_path = 'calibration.mat'  

    # Save the matrix to a .mat file

    # Save the rotation matrix as txt  
    np.savetxt('rotation_matrix.txt', rotation_matrix)  

    # Save the translation vector as txt
    np.savetxt('translation_vector.txt', translation_vector)

    # Save the camera matrix as txt
    np.savetxt('camera_matrix.txt', mtx)    

    # savemat(file_path, {'m': mtx, 'r': rotation_matrix, 't': translation_vector})    
    return mtx, dist, rvecs, tvecs, rotation_matrix, translation_vector   


def cal_world_pos(img_points, mtx, rotation_matrix, translation_vector):      
    print(np.array(img_points[0]).T.shape)   
    # print(cv2.invert(rotation_matrix)) 
    # print(cv2.invert(mtx))   
    print(np.dot(cv2.invert(rotation_matrix)[1], cv2.invert(mtx)[1]))   
    ### calculate s 
    right_M = np.dot(np.dot(cv2.invert(rotation_matrix)[1], cv2.invert(mtx)[1]), np.array(img_points[0]).T)     
    left_M = np.dot(cv2.invert(rotation_matrix)[1], translation_vector)     
    
    s = left_M[2] / right_M[2]   
    
    # print(s * np.dot(cv2.invert(mtx)[1], np.array(img_points[0]).T) - translation_vector)  
    world_points = np.dot(cv2.invert(rotation_matrix)[1], (s * np.dot(cv2.invert(mtx)[1], np.array(img_points[0]).T) - translation_vector))  
    
    return world_points  



if __name__ == "__main__":  
    # Load matrices back from the npz file
    
    # Get the current working directory
    current_directory = os.getcwd()  
    print("Current Working Directory:", current_directory)   
    
    # print(np.array(p.getMatrixFromQuaternion(np.array([0.903813, -0.37872, 0.192175, -0.0525499]))).reshape(3, 3))  
    
    mtx, dist, rvecs, tvecs, rotation_matrix, translation_vector = cal_corners(start_index=8, use_num=1)   
    
    # print("m_matrix :", mtx) 
         
    img_points = np.array([[[500, 200, 1]]], dtype=np.float32)     
    world_points = cal_world_pos(img_points, mtx, rotation_matrix, translation_vector)
    # # Reshape the result to get a list of 3D points
    world_points = world_points.reshape(-1, 3)  
    print("world_points :", world_points)    
    

    # # Invert the rotation and translation matrices
    # inverse_rotation_matrix = cv2.invert(rotation_matrix)
    # inverse_translation_vector = -translation_vector  # Negative of the translation vector

    # print("inverse :", inverse_rotation_matrix, inverse_translation_vector)  
    # print("img points :", img_points.reshape(1, -1, 2))  
    
    # Transform image points from camera coordinates to world coordinates
    # world_points = cv2.transform(img_points.reshape(1, -1, 2), inverse_rotation_matrix[1], inverse_translation_vector)


    ###############################################
    # T_ee = np.zeros((4, 4))   
    # T_ee[:3, :3] = np.array(p.getMatrixFromQuaternion(np.array([0.0, 0.0, 0.0, 1.0]))).reshape(3, 3)        
    # # T_ee[:3, 3] = position    
    
    # print("T_ee : \n", T_ee)

    # print(p.getMatrixFromQuaternion(np.array([0.0, 0.0, 0.0, 1.0])).resize(3,3))    
    
    # cal_cart_pos(0, num_corners_x = 7, num_corners_y = 6, square_size=0.0235)  