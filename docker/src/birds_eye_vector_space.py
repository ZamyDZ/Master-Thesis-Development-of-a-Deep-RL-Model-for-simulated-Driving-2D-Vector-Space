# Imports
import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


class Vector_space:
    '''
    TODO: Explaining what this File/Class does
    '''
    def __init__(self, number_of_anchor_points: int, road_roi, warped_image_shape,
                 warped_image_height, warped_image_width, number_of_depth_layers) -> None:
        print('vector space is initialized')
        
        # Loading the camera matrix parameters
        self.camera_parameters = self.loading_camaera_parameters()
        
        # Attributes
        self.number_anchor_points = number_of_anchor_points
        self.anchor_points = None   # TODO: update later
        
        self.road_roi = road_roi
        self.warped_image_shape = warped_image_shape
        self.warped_image_height = warped_image_height
        self.warped_image_width = warped_image_width
        self.number_of_depth_layers = number_of_depth_layers
        self.estimated_depth_layers = self.depth_layers_estimation()
        
        self.perspective_transformation_matrix = None
        self.inverse_perspective_transformation_matrix = None

    
    def loading_camaera_parameters(self):   # TODO: SHOULD I ADD A PATH UP HERE?
        '''
        Function description ...
        '''
        print('loading camera parameters')
        # TODO: CHANGE THE PATH TO USE PWD AND THEN THE FOLLOW UP DIRECTORY
        camera_parameters_path = '/home/zamy/masterthesis/donkeycar_camera_calibration/camera_parameters/camera_parameters.pkl'
        cam_parameters = open(camera_parameters_path,'rb')
        #camera_parameters = pickle.load(cam_par)
        return pickle.load(cam_parameters)

    def undistort_observation(self, image: np.array):
        '''
        Function description ...
        '''
        height, width = image.shape[:2]
        new_camera_matrix, region_of_interest = cv.getOptimalNewCameraMatrix(self.camera_parameters['mtx'],
                                                                             self.camera_parameters['dist'],
                                                                             (width,height),
                                                                             1,
                                                                             (width,height))
        undistorted_image = cv.undistort(image, 
                                         self.camera_parameters['mtx'], 
                                         self.camera_parameters['dist'],
                                         None, 
                                         new_camera_matrix)
        return undistorted_image

    def warping(self, image: np.array):
        '''
        Function description ... Perspective Transformation
        '''
        width, height = image.shape
        # defining new image shape for the perspective transformation
        image_shape = np.float32([[0,0],[width,0],[0,height],[width,height]])
        # Tranformation matrix
        self.perspective_transformation_matrix = cv.getPerspectiveTransform(self.road_roi, image_shape)
        self.inverse_perspective_transformation_matrix = np.linalg.inv(self.perspective_transformation_matrix)
        # Warping the image
        warped_image = cv.warpPerspective(image,self.perspective_transformation_matrix,(width,height))
        return warped_image

    def region_of_interest(self, image: np.array, verticies):
        '''
        Function description ... Selecting the region of interest
        '''
        mask = np.zeros_like(image)
        cv.fillPoly(mask,verticies,255)
        masked = cv.bitwise_and(image,mask)
        return masked
    
    def point_perspective_transform(invmatrix, point):
        '''
        Function description ...
        # FORMULA src: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga73673a7e8e18ec6963e3774e6a94b87
        '''  
        x = (invmatrix[0,0] * point[0] + invmatrix[0,1] * point[1] + invmatrix[0,2])/(invmatrix[2,0] * point[0] + invmatrix[2,1] * point[1] + invmatrix[2,2])
        y = (invmatrix[1,0] * point[0] + invmatrix[1,1] * point[1] + invmatrix[1,2])/(invmatrix[2,0] * point[0] + invmatrix[2,1] * point[1] + invmatrix[2,2])
        return np.array([x,y])
    
    def finding_road_lanes(self, image: np.array):
        '''
        Function description ...
        '''  
        # Canny edge detector
        warped_image_canny = cv.Canny(image, threshold1=50, threshold2=100)
        # Selection the region of interest through a mask
        warped_image_canny = self.region_of_interest(warped_image_canny,[self.warped_image_shape])
        
        # Detecting line contours
        # Getting the countours, return contours & hierarchy (nested objects, contours in controus)
        contours, _ = cv.findContours(warped_image_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # array of the contours
        contour_arr = []
        for i, cnt in enumerate(contours):        
            if len(cnt) > 300:
                perimeter = None
                perimeter = cv.arcLength(cnt,False)
                # if perimeter is not None:     # TODO: Check if necessary
                contour_arr.append([cnt,perimeter])
        
        contour_anchor_points = []
        # find the start & end point of a countour
        for i,cnt in enumerate(contour_arr):
            # get the min and max index position of the contour (they are switched because y starts at the top)
            #min_index = np.argmax(cnt[0][:,0][:,1])
            #max_index = np.argmin(cnt[0][:,0][:,1])

            # Get n destributed points of the contour
            # calculate the steps, so we get every x anchor point of the n contours
            step = int(len(cnt[0][:,0]) / self.number_anchor_points)
            distributed_anchor_points = cnt[0][:,0][::step+1]
            # adding the start and the end point
            #contour_pts = {'start':[cnt[0][max_index,0][0],cnt[0][max_index,0][1]],
            #               'end':[cnt[0][min_index,0][0],cnt[0][min_index,0][1]],
            #               'anchor':distributed_anchor_points}
            # adding the contour with the anchor points to the array
            #contour_anchor_points.append(contour_pts)
            contour_anchor_points.append(distributed_anchor_points)
        return contour_anchor_points

    def gaussian(self, x: int, mu: int, sig: int):
        '''
        Function description ...
        '''
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def binary_search_depth_layer(self, y_pos: int, buckets: list, left: int, right: int):
        '''
        Function description ...
        '''
        z_found = None
        mid = int(left + ((right - left) / 2))
        #print('mid: %s' % mid)
        # if the y_pos is represented in the middle
        if y_pos == buckets[mid][0]:
            return buckets[mid][1]
        # if the y_pos will be assigend to the last z of the buckets
        if y_pos >= buckets[mid][0] and mid == (len(buckets)-1):
            # return the z value
            return buckets[mid][1]
        # check if the y is greater than the mid and smaller then the mid+1 val, then assign z
        if y_pos >= buckets[mid][0] and y_pos < buckets[mid+1][0]:
            # return the z value
            return buckets[mid][1]
        # continue search on the left split side of the buckets
        if y_pos < buckets[mid][0]:                             ##  VLT HIER: <=
            return self.binary_search_depth_layer(y_pos, buckets, left, mid-1)
        # continue search on the right split side of the buckets
        if y_pos > buckets[mid][0]:                             ##  VLT HIER: >=
            return self.binary_search_depth_layer(y_pos, buckets, mid+1, right)
        ### TODO: MISSING DEFAULT RETURN VALUE??? RETURN NONE???
        print('SRY NO VALUE ASSIGNED: %s' % y_pos)
    
    def depth_layers_estimation(self):
        '''
        Function description ...
        '''
        steps = [i**2 for i in list(np.arange(0,18,.4))]
        # LOGICAL REPRESENTATION
        return [[int(val),int(100 - self.gaussian(val,1,100) * 100)] for val in steps]

    def assigning_estimated_depth_to_points(self, contours: list):
        '''
        Function description ...
        '''
        # Assign points to the z-layers
        contour_world_points = []
        contour_vectorspace_points = []
        for contour in contours:
            contour_points = []
            vector_points = []
            for point in contour:
                u_coordinate = point[0]
                v_coordinate = point[1]
                
                # Assigning the depth
                z_coordinate = self.binary_search_depth_layer(v_coordinate, self.estimated_depth_layers,0,len(self.estimated_depth_layers))
                
                # 3 dimensional world points
                #x = z_pos / camera_focal_x * (u_pos - camera_x_pos)
                #y = z_pos / camera_focal_y * (v_pos - camera_y_pos)
                
                # 3 dim-world point
                world_point = np.array([u_coordinate, v_coordinate, z_coordinate])
                
                # New 2 dimensional vector point
                vector_point = np.array([u_coordinate, z_coordinate])
                
                # Adding the new point to the list
                contour_points.append(world_point)
                vector_points.append(vector_point)
            contour_world_points.append(contour_points)
            # Convert the vector space points to a numpy array so this can be used by open cv drawContours
            # It need to be a 3-dimensional Numpy array, so the drawContours functions works
            # Append the vector points and make them 3 dimensional, so they can be drawn with OpenCV
            contour_vectorspace_points.append(np.array([[vector_points]]))
        return contour_vectorspace_points, contour_world_points

    def create_vector_space_image(self, contours: list):
        ''' BEISPIEL COMMENT, ABER INNERHALB DER FUNKTION !!!!!
        <mat>: ndarray, input array to pool
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <method>: str, 'max for max-pooling, 
                       'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
               n//f, n being <mat> size, f being kernel size.
               if pad, output has size ceil(n/f).

        Return <result>: pooled matrix.
        '''  
        # Create an empty vector space image
        vector_space_image = np.zeros((100,self.warped_image_width))
        # Drawing the contours onto the empty vector space image
        for contour in contours:
            cv.drawContours(vector_space_image, contour[0], -1,(255,255,255), 1)
        # Max-Pooling to reduce the image size (shape)
        dimension = (int(vector_space_image.shape[1] / 2), int(vector_space_image.shape[0] / 2))
        pooled_vector_space_image = cv.resize(vector_space_image, dimension, interpolation = cv.INTER_AREA)
        return pooled_vector_space_image
    
    def image_preprocessing(self, image: np.array):
        '''
        Function description ...
        '''
        # Undistortion 
        undistorted_image = self.undistort_observation(image)
                
        # Warping
        warped_image = self.warping(undistorted_image)
        
        # Finding road lanes
        anchor_points = self.finding_road_lanes(warped_image)
        
        # Depth estimation on the anchor points
        contour_vector_points, contour_world_points = self.assigning_estimated_depth_to_points(anchor_points)
        
        # Creating the vector space image and applying max-pooling
        vector_space_image = self.create_vector_space_image(contour_vector_points)
        
        #cv.imshow('Vector Space Image', vector_space_image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        
        #plt.figure()
        #plt.imshow(vector_space_image, cmap='gray')
        #plt.show()
        return vector_space_image        
        

def main():
    ANCHOR_POINTS = 8
    ROAD_ROI = np.array([(120,90),(200,90),(0,200),(320,200)],dtype='float32')
    WARPED_IMAGE_SHAPE = np.array([[10,320],[0,0],[200,0],[200,310]],np.int32)            # NEW IMAGE Shape after Warping !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    WARPED_IMAGE_HEIGHT = 320
    WARPED_IMAGE_WIDTH = 200
    NUMBER_OF_DEPTH_LAYERS = 50
    
    v1 = Vector_space(ANCHOR_POINTS, ROAD_ROI, WARPED_IMAGE_SHAPE, WARPED_IMAGE_HEIGHT, WARPED_IMAGE_WIDTH, NUMBER_OF_DEPTH_LAYERS)
    
    # Loading example image
    path = '/home/zamy/masterthesis/donkeycar_camera_calibration/calibration_images/frame_15.png'
    obs_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    # check class functions
    final_image = v1.image_preprocessing(obs_image)
    
    cv.imshow('Vector Space Image', final_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    
    # check class attributes
    #print('%s' % v1.camera_parameters['mtx'])
    
    pwd_path = os.getcwd()
    print('Path: %s' % pwd_path)

        
if __name__ == '__main__':
    main()
    