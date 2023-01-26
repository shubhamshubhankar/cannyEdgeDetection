import sys
import getopt
import cv2
import numpy as np
import math

if __name__ == "__main__":

    try:
        if len(sys.argv) < 4:
            print(
                "\n-----------------------------------------------------------------------------------------------------------------------\n"
                "This is a python implementation of the Canny Edge Detection. \n\n"
                "Usage given below:\n"
                "python cannyEdgeDetectorNew.py 'imageName' lowThreshold highThreshold\n\n"
                "All the arguements in this program are mandatory.\n"
                "Arguement List:\n"
                "imageName - Path of the image + name of the image (String data type)\n"
                "lowThreshold - Value of the low threshold for the doubleThresholding operation.(Integer data type)\n"
                "highThreshold - Value of the low threshold for the doubleThresholding operation.(Integer data type)\n\n"
                "Example : \n"
                "python3 cannyEdgeDetectionNew.py 3.jpg 20 50\n"
                "\n----------------------------------------------------------------------------------------------------------------------\n")
            exit(0)
        else:
            image_name = sys.argv[1]
            weak_threshold = int(sys.argv[2])
            strong_threshold = int(sys.argv[3])
        
        image = cv2.imread(image_name)

        # Convert to grayScale image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initializing the values for the Gaussian Filter on the image 
        # of size 3*3 in order to blur it.
        gaussianKernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]] 

        # Dividing the value by 16 to manually provide the values for gaussian
        # filter.
        for i in range(0, 3): 
          for j in range(0, 3):
            gaussianKernel[i][j] = gaussianKernel[i][j] / 16
        
        # Using shape function finding out the dimensions of the image.
        rows, cols = gray.shape

        # Creating a copy of the image
        blur = gray 
        
        # Saving the gray image in the current folder.
        cv2.imwrite("gray.jpg", gray)
        
        # Convolving gray image with gaussianKernel.
        for i in range(1, rows - 1):
          for j in range(1, cols - 1): 
            blur[i][j] = float(gray[i - 1][j - 1] * gaussianKernel[0][0] + \
                         gray[i - 1][j] * gaussianKernel[0][1] + \
                         gray[i - 1][j + 1] * gaussianKernel[0][2] + \
                         gray[i][j - 1] * gaussianKernel[1][0] + \
                         gray[i][j] * gaussianKernel[1][1] + \
                         gray[i][j + 1] * gaussianKernel[1][2] + \
                         gray[i + 1][j - 1] * gaussianKernel[2][0] + \
                         gray[i + 1][j] * gaussianKernel[2][1] + \
                         gray[i + 1][j + 1] * gaussianKernel[2][2]);

        # Saving the blur image in the current folder.
        cv2.imwrite("blur.jpg", blur)
        
        # All the operations now will be done on gray image.
        gray = blur
        
        # Initializing the values for the Sobel Filter on the image 
        # of size 3*3 in order to blur it.
        sobelKernelX = [[1, 0, -1], [2, 0, -2], [1, 0, -1]] 
        sobelKernelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] 
        
        # Initializing the image in order to create the result.
        x = np.zeros((rows, cols), dtype=np.float64)
        y = np.zeros((rows, cols), dtype=np.float64)


        # Convolving gray image with sobelKernelX and sobelKernelY.
        for i in range(1, rows - 1):
          for j in range(1, cols - 1): 
            x[i][j] = (gray[i - 1][j - 1] * sobelKernelX[0][0] + \
                         gray[i - 1][j] * sobelKernelX[0][1] + \
                         gray[i - 1][j + 1] * sobelKernelX[0][2] + \
                         gray[i][j - 1] * sobelKernelX[1][0] + \
                         gray[i][j] * sobelKernelX[1][1] + \
                         gray[i][j + 1] * sobelKernelX[1][2] + \
                         gray[i + 1][j - 1] * sobelKernelX[2][0] + \
                         gray[i + 1][j] * sobelKernelX[2][1] + \
                         gray[i + 1][j + 1] * sobelKernelX[2][2]);
            
            y[i][j] = (gray[i - 1][j - 1] * sobelKernelY[0][0] + \
                         gray[i - 1][j] * sobelKernelY[0][1] + \
                         gray[i - 1][j + 1] * sobelKernelY[0][2] + \
                         gray[i][j - 1] * sobelKernelY[1][0] + \
                         gray[i][j] * sobelKernelY[1][1] + \
                         gray[i][j + 1] * sobelKernelY[1][2] + \
                         gray[i + 1][j - 1] * sobelKernelY[2][0] + \
                         gray[i + 1][j] * sobelKernelY[2][1] + \
                         gray[i + 1][j + 1] * sobelKernelY[2][2]);
        
        # Saving the x and y image in the current folder.
        cv2.imwrite("x.jpg", x)
        cv2.imwrite("y.jpg", y)

        # Using addWeighted function in order to get the images with both
        # x and y components in the weighted form.
        image_xy = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)

        # Creating an image for the gradient.
        theta = np.zeros((rows, cols), dtype=np.float64)

        # Finding the out the theta(angle of the edges) with the range 0 to 360 degree.
        # Image with both horizontal and vertical Sobel kernels applied.
        for i in range(1, rows - 1):
          for j in range(1, cols - 1): 
              theta[i][j] = np.rad2deg(math.atan2(y[i][j], x[i][j]))
              #print("theta ", theta[i][j])
        
        '''
            Non Maximum Suppression
        '''

        for i in range(0, rows-1):
                for j in range(0, cols-1):
                    try :
                        # Used to take the next component and the previous component,
                        # in order to compare with the current value.
                        # The initial value is stored as 
                        nextComponent = 255
                        prevComponent = 255
                        
                    # General direction along 0 degree
                        if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] < 202.5)  or (337.5 <= theta[i,j] <= 360):
                            nextComponent = image_xy[i, j+1]
                            prevComponent = image_xy[i, j-1]
                        
                        # General direction along 45 degree
                        elif (22.5 <= theta[i,j] < 67.5) or (202.5 <= theta[i,j] < 247.5):
                            nextComponent = image_xy[i+1, j-1]
                            prevComponent = image_xy[i-1, j+1]
                        
                        # General direction along 90 degree
                        elif (67.5 <= theta[i,j] < 112.5) or ((247.5 <= theta[i,j] < 292.5)):
                            nextComponent = image_xy[i+1, j]
                            prevComponent = image_xy[i-1, j]
                        
                        # General direction along 135 degree
                        elif (112.5 <= theta[i,j] < 157.5) or (292.5 <= theta[i,j] < 337.5):
                            nextComponent = image_xy[i-1, j-1]
                            nextComponent = image_xy[i+1, j+1]

                        # If the value of the current pixel is greater than prev pixel and the
                        # next pixel in the direction of the theta, then it will save the value
                        # otherwise it will store it as 0.
                        if (image_xy[i,j] < nextComponent) and (image_xy[i,j] < prevComponent):
                            image_xy[i,j] = 0
                    
                    # If the theta read doesn't lies in the range of (0, 360) degree
                    except IndexError as e:
                        pass    

        '''
        Double Thresholding operation using 2 pixel intensity values(strong and weak).
        '''
        
        #cv2.imshow("lenna_Before", image_xy)
        #cv2.waitKey(5000)

        # Code snippet in order to see the histogram of the image.
        '''
        hist = cv2.calcHist([image_xy], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(5000)
        '''

        # Creating a matrix of size rows*cols, same as the original image.
        resultant = np.zeros((rows, cols), dtype = np.float64)

        # Here, strong edges are denoted by 2, weak by 1 and rest are moved to 0.
        for i in range(0, rows-1):
                for j in range(0, cols-1):
                    if image_xy[i, j] < weak_threshold:
                        image_xy[i, j] = 0
                        resultant[i, j] = 0
                    elif image_xy[i, j] >= weak_threshold and image_xy[i, j] < strong_threshold:
                        resultant[i, j] = 1
                    elif image_xy[i, j] > strong_threshold:
                        resultant[i, j] = 2

        '''
        Performing hysteresis in order to convert weak edges into strong edges where
        they are linked with a strong edge.
        '''
        for i in range(1, rows-1):
                for j in range(1, cols-1):
                    # If the current pixel is a weak edge.
                    if resultant[i, j] == 1:
                        # If any of the surrounding pixel is a strong edge, then
                        # the current pixel will also be considered as a strong edge.
                        if  (resultant[i-1, j-1] == 2) or (resultant[i-1, j] == 2) or (resultant[i-1, j+1] == 2) or \
                            (resultant[i, j-1] == 2) or (resultant[i-1, j+1] == 2) or \
                            (resultant[i+1, j-1] == 2) or (resultant[i+1, j] == 2) or (resultant[i+1, j+1] == 2):
                            image_xy[i, j] = image_xy[i, j]
                        else:
                            image_xy[i, j] = 0

        # Saving the image in the current folder.
        cv2.imwrite("output.jpg", image_xy)

        # Showing the Lenna image here.
        cv2.imshow("Canny Edge applied Image", image_xy)
        cv2.waitKey(5000)
    
    except getopt.error as err:
        print(str(err))

