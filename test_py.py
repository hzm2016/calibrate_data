import zmq  

phase = zmq.Context()      
socket_phase = phase.socket(zmq.PULL)            
socket_phase.bind("tcp://*:5555")             
# position = np.array([0.0, 0.0, 0.0]) 
phase_index = 0      

while True: 
    phase_message = socket_phase.recv()    
    # received_phase = np.frombuffer(phase_message, dtype=np.float64)   
    print("received_phase :%s", phase_message)          