# this file for do a process of computer vision, when need to combine several function into one pipeline
#input is path or image instance, output is score for sharpness

import cv2
import csv
import numpy as np
import time
import math
import os
import re
from scipy.signal import find_peaks
from preprocessing import imageProcessor
# from StartStreamCamera_Running import *

pixel_per_mm = 0.0015

endthread = False

# -----------------------------------------------------------------------------------------------
# ----------------------------------- Detect Well Edge ------------------------------------------
# -----------------------------------------------------------------------------------------------

def calculate_black_white_ratio(binary_image):
    '''
    Calculate the black to white (pixels) ratio in an image
    
    :param binary_image: A binary image (a threshold image)
    
    :return: The black to white (pixels) ratio
    
    '''
    
    # Make sure that the image is binary
    if len(binary_image.shape) != 2:
        raise ValueError("Ảnh đầu vào không phải là ảnh nhị phân.")

    # Counting the white pixels (value at 255) and the black pixels (value at 0)
    total_pixels = binary_image.size            # Total pixels in the image
    # print("Total pixels:", total_pixels)
    white_pixels = np.sum(binary_image == 255)  # Number of white pixels
    black_pixels = np.sum(binary_image == 0)    # Number of black pixels
    # print("White pixels = ", white_pixels)
    # print("Black pixels = ", black_pixels)

    # Calculate the ratio
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels

    # return white_pixels, black_pixels, white_ratio, black_ratio
    return black_ratio
    
def detectWellEdge(x_start, y_start, percentage):
    '''
    Find the Well Edge
    
    :param x_start: The starting x coord
    :param y_start: The starting y coord
    :param percentage: The percentage of black pixels to white pixels needed to break the loop

    :return: The final x and y coord
    '''
    
    i = 10
    count = 0
    global endthread
    step_y = 0.12                # Distance of each step in the y_axis
    z_start = 20.8               # Starting value of z

    # Moving all x, y and z axis to the starting position
    print("MOVING TO Start")
    # move_motor_request_x(x_start)
    # waiting_x()
    # move_motor_request_y(y_start)
    # waiting_y()
    # move_motor_request_z(z_start)
    # waiting_z()
    
    time.sleep(1)               # Delay before Starting
    
    y_scan = y_start            # Assigning the y_scan the starting y coord
    
    print("Starting MoveEdge Y Axis")
    
    while endthread == False:
        # Move y 1 step for each loop with a newly updated y_scan coord
        print("MoveEdge Y:", y_scan)
        # move_motor_request_y(y_scan)
        # waiting_y()
        
        # image = imageProcessor(export_image())            # REPLACE with image read
        
        file_name = f"img_{i}"
        image = imageProcessor(f"anhTestGieng/tenet_14_10_1/{file_name}.jpg")
        
        i += 1      # Index for each loop and image
        
        
        ratio = calculate_black_white_ratio(image.threshold_image)
        print("Ratio:", ratio*100, "%")
        
        if ratio > percentage :
            print(f"Reached {percentage}% !!!")
            break
        else:
            print("Continue another MoveEdge Loop")
            y_scan -= step_y
            
        if y_start - y_scan >= 100:  #mm
            y_scan = y_start
            break
        
    print("MoveEdge DONE")
    print("Final Image index: ", i)
    
    x_out = x_start
    y_out = y_scan
    return x_out, y_out

# detectWellEdge(x_starting_pos, y_starting_pos, ratio)




# -----------------------------------------------------------------------------------------------
# ----------------------------------- Focus Well Edge -------------------------------------------
# -----------------------------------------------------------------------------------------------

def filter_contour(contours, threshold=450):
    '''
    Input : contour list and threshold in arc curve length
    Output: the longest contour that satisfies the threshold
    '''
    longest_contour = None
    list_contour = []
    max_length = -10000
    
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length > threshold :
            list_contour.append(contour)
            if length > max_length:
                longest_contour = contour
                max_length = length

    print(f'Max length is {max_length}')
    
    return list_contour, longest_contour   
    
def compute_variance_contour(mom_path):
    '''
    Input is a list of image and its peak index
    Output: list of variance of all peak image'''

    keep_list = []
    variance_list  = []

    for img_name in os.listdir(mom_path):
        keep_list.append(img_name)
        image = cv2.imread(os.path.join(mom_path,img_name))
        out_image = image.copy()

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(gray, (5, 5), 0)  # Kích thước kernel là 5x5, có thể điều chỉnh tùy ý
            image = cv2.medianBlur(image, 5)
            # Thiết lập ngưỡng và áp dụng lọc ngưỡng
            threshold_value = 50  # Giá trị ngưỡng #130
            max_value = 255  # Giá trị tối đa

            ret, threshold_image = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
            # Tìm các contours trong ảnh đã lọc ngưỡng

            canny_image = cv2.Canny(threshold_image, threshold1=30, threshold2=100)
            contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours,_ = filter_contour(contours)

            for i, contour in enumerate(contours):
                # Compute the length of the contour
                length = cv2.arcLength(contour, closed=False)

                # Draw the contour on the image
                # cv2.drawContours(out_image, [contour], -1, (0, 255, 0), 2)  # Draw contours in green

                # Get the center point of the contour to position the text
                moments = cv2.moments(contour)
                if moments['m00'] != 0:  # To avoid division by zero
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = contour[0][0]  # If moments can't calculate center, use first point of contour

                # Step 4: Write the length on the image at the center of the contour
                cv2.putText(out_image, f"{length:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            contour = max(contours, key=cv2.contourArea)  # Taking the largest contour
            contour_points = contour[:, 0, :]  # Extracting x, y positions

            # Split contour points into separate x and y arrays
            x = contour_points[:, 0]
            y = contour_points[:, 1]

            # Step 2: Fit a curve (e.g., 2nd degree polynomial) to the contour points
            coefficients = np.polyfit(x, y, deg=2)  # You can change 'deg' based on the trend
            poly_fit = np.poly1d(coefficients)

            # Step 3: Compute the expected y-values for each x using the fitted polynomial
            y_fitted = poly_fit(x)

            # Step 4: Calculate the residuals (difference between actual y and fitted y)
            residuals = y - y_fitted

            # Step 5: Compute the variance of the residuals
            variance = np.var(residuals)
        except:
            variance = 10000
        variance_list.append(variance)

        cv2.drawContours(out_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f'teneImage\contour_applied\{os.path.splitext(os.path.basename(img_name))[0]}_variance_{str(round(variance,3))}.jpg',out_image)
    return variance_list,keep_list

def delete_non_peak_image(mom_path,peak_list):
    '''
    Delete image files that are not a peak in a folder log.
    
    :param mom_path: Folder containing all images saved when moving the z axis, relative.
    :param peak_list: A list of all peaks index
    
    :return: A list of remaing images.
    
    '''
    keep_image_list = []

    for i in range(len(peak_list)):
        keep_image_list.append(os.path.join(f'img_{peak_list[i]}.jpg'))

    for file_name in os.listdir(mom_path):
    # Check if the file is not in the keep_images list
        if file_name not in keep_image_list:
            file_path = os.path.join(mom_path, file_name)
            
            # If the file is an image (you can filter by extension here if necessary)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
                # print(f"Deleted: {file_name}")
            else:
                continue
    return keep_image_list

def get_last_index_num(image_file):
    '''
    Get the index of the image
    
    '''
    file_name_without_ext = os.path.splitext(os.path.basename(image_file))[0]

    # Step 2: Use regular expressions to find the last number in the string
    last_number_str = re.findall(r'\d+', file_name_without_ext)[-1]

    # Step 3: Convert the last number to an integer
    last_number = int(last_number_str)
    return last_number
    #Compute variance of all peak image, then choose the lowest variable, return the index of peak has lowest variable

def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv2.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    return mn

def templateMatching(image, template, method=cv2.TM_CCOEFF_NORMED): # Insert a Grayscale image
    '''
    Calculate the image sharpness after circling the area using Template Matching
    
    :param image: A grayscaled image
    :param template: A grayscaled template image
    :param method: The Template Matching method
    
    :return: The sharpness value of the image
    
    '''
    
    h, w = template.shape
    img2 = image.copy()
    
    # up = (440,170)
    # down = (470,200)
    # cv2.rectangle(img2, up, down, 0, -1)

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location = max_loc
    bottom_right = (location[0] + w, location[1] + h)
    
    # print(location, bottom_right)    

    img3 = image.copy()
    
    # cv2.rectangle(img3, location, bottom_right, 0, 5)
    # cv2.imshow('Match', img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cropped_image = image[location[1]:bottom_right[1], location[0]:bottom_right[0]]
    
    # _, thresholded = cv2.threshold(cropped_image, 160, 255, cv2.THRESH_BINARY)
    
    # cv2.imshow('cropped', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # error = mse(cropped_image,template)
    # return error
    
    tene = tenengrad(cropped_image)
    
    return tene

def focusWellEdge(z_start):
    '''
    Hàm focus Cạnh giếng
    Input : tọa độ z khởi tạo
    Output: tọa độ z cho ra ảnh rõ nét nhất
    
    '''
    
    global endthread
    
    tene_list = []
    allowed_distance = 0.75   # Maximum distance that the z axis is allowed to travel (by mm)
    z_step = 0.007            # Distance of each step in the z axis
    z_scan = z_start

    index = 0                 # Image index number
    
    template_image = imageProcessor("img_temp5.jpg")        # Template Image
    
    while endthread == False:
        image = imageProcessor('anh1.jpg')      # Replace with Camera read
        
        index+= 1            
        if index < 10:                          # Skip over the first 10 images
            continue
        
        tene = templateMatching(image.gray_image , template_image.gray_image)
        
        print("Tene value is: ", tene)
        
        if(z_scan >= z_start + allowed_distance):   # Break if exceeds the allowed distance
            break
        
        print(f'\nDone Job: {round((z_start-z_scan) / allowed_distance * 100, 2)} % \n')  # Print the searching progress
        
        z_scan += z_step
        # move_motor_request_z(z_scan)              # Move command
        # waitingz()
        
        tene_list.append(tene)  # Create an array for the tene value of each step
        
        with open('teneImage/log_tele.csv', mode = 'a', newline = '') as file:
            writer = csv.writer(file)

            file.seek(0,2)
            if file.tell() == 0:                                                        # Save the data in a csv file and the image in a folder
                writer.writerow(['Index','Tenebraum','Z Height','Image path'])
            image_name = f'teneImage/log_image/img_{index}.jpg'
            writer.writerow([index, tene, z_scan, image_name])

            cv2.imwrite(image_name , image)
                        
    tene_list = np.array(tene_list)
    peaks, _ = find_peaks(tene_list, distance=60, height=60)                            # Find the peaks in the tene_list graph
    
    _ = delete_non_peak_image('teneImage\log_image', peaks)                             # Delete (cut out) the images that are not peaks
    variance_list, keep_list_name = compute_variance_contour('teneImage\log_image')     # 
    print(f'\nKeep file : {keep_list_name}\n')
    index_min_variance_contour = get_last_index_num(keep_list_name[np.argmin(variance_list)])   #  Get the index number of the image with the
                                                                                                #  least variance of contours
    
    z_sharp_final = z_start + index_min_variance_contour * z_step                       # Calculate the final z coord with the sharpest image

    # move_motor_request_z(z_sharp_final)                                               # Move the camera to that z coord
    # waitingz()

    z_out = z_sharp_final

    # print(f'\n\n Da di chuyen truc Z den vi tri net la {z_out}, index peak min contour la {np.argmin(variance_list)}\n\n')

    time.sleep(3)
    return z_out , index_min_variance_contour

# focusWellEdge(z_starting_pos)




# -----------------------------------------------------------------------------------------------
# ----------------------------------- Find Center -----------------------------------------------
# -----------------------------------------------------------------------------------------------

def save_oocyte_with_circle(image, ct):
    # Lưu ảnh xấp xỉ hình tròn để kiểm tra
    # print(f'ct before {ct}')
    
    if len(ct) == 1:
        ct = ct[0]

    # Convert the contour to the correct shape (remove the unnecessary dimension)
    try:
        ct_max = ct.reshape(-1, 2)  # Reshape to (N, 2), where N is the number of points
    except:
        ct_max = ct

    # print(f'len ct after is {len(ct_max)}, ||| {ct_max}')

    image_color = image.copy()

    # Làm to khung hình
    offset_x_up = 500
    offset_y_up = 500
    offset_x_down = 3000
    offset_y_down = 3000

    h_origin, w_origin = image_color.shape[:2]
    new_size = (h_origin + offset_y_up + offset_y_down, w_origin + offset_x_up + offset_x_down, 3)

    new_img = np.zeros(new_size, dtype=np.uint8)  # Canvas image without content
    new_img[offset_y_up:h_origin + offset_y_up, offset_x_up:w_origin + offset_x_up] = image_color

    # Select points A, B, and C from the contour
    A_draw = ct_max[len(ct_max) // 10].copy()
    B_draw = ct_max[len(ct_max) // 5].copy()
    C_draw = ct_max[len(ct_max) // 2].copy()

    # Apply offset to A, B, and C
    A_draw[0] += offset_x_up
    A_draw[1] += offset_y_up

    B_draw[0] += offset_x_up
    B_draw[1] += offset_y_up

    C_draw[0] += offset_x_up
    C_draw[1] += offset_y_up

    # Calculate the center and radius using the circleApprox function
    print(f'[saveOOcyte] Ba diem A, B, C de xap xi duoc chon lan luot la: {A_draw}, {B_draw}, {C_draw}')
    center, radius = circleApprox(A_draw, B_draw, C_draw)

    # Draw the center and the circle
    cv2.circle(new_img, center, 20, (255, 255, 255), thickness=-1)  # Center

    # Draw the points A, B, C and annotate them
    cv2.circle(new_img, tuple(A_draw), 10, (255, 255, 0), thickness=-1)
    cv2.putText(new_img, 'A', (A_draw[0] + 20, A_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    cv2.circle(new_img, tuple(B_draw), 10, (255, 255, 0), thickness=-1)
    cv2.putText(new_img, 'B', (B_draw[0] + 20, B_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    cv2.circle(new_img, tuple(C_draw), 10, (255, 255, 0), thickness=-1)
    cv2.putText(new_img, 'C', (C_draw[0] + 20, C_draw[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=5, color=(255, 255, 255), thickness=5)

    # Draw the circle with the calculated radius
    cv2.circle(new_img, center, int(radius), (255, 255, 255), thickness=5)

    # Save the resulting image
    cv2.imwrite(f'teneImage/approxCircle/xapXiHinhTron_{radius:.2f}_pixel.jpg', new_img)

def circleApprox(A, B, C):
    '''
    '''
    
    #Xap xi hinh tron khi biet 3 diem, tra ve toa do tam (pixel) va ban kinh (pixel)
    xa, ya = A
    xb, yb = B
    xc, yc = C
    
    # Phương trình trung trực của AB
    D = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
    
    # Tính tọa độ của tâm đường tròn (center)
    Ux = ((xa**2 + ya**2) * (yb - yc) + (xb**2 + yb**2) * (yc - ya) + (xc**2 + yc**2) * (ya - yb)) / D
    Uy = ((xa**2 + ya**2) * (xc - xb) + (xb**2 + yb**2) * (xa - xc) + (xc**2 + yc**2) * (xb - xa)) / D
    
    # Tính bán kính (radius) bằng khoảng cách từ tâm đến điểm A
    radius = math.sqrt((Ux - xa) ** 2 + (Uy - ya) ** 2)
    
    # Trả về cả tâm (Ux, Uy) và bán kính (radius)
    return (int(Ux), int(Uy)), radius

def findCenter(x_edge, y_edge, z_edge):
    '''
    Find the Center of the Well Edge (in mm)
    
    :param x_edge: Coord of x axis after finding Well Edge
    :param y_edge: Coord of y axis after finding Well Edge
    :param z_edge: Coord of z axis after finding Well Edge

    :return x_center_mm: Coord x of the Well Center in mm
    :return y_center_mm: Coord y of the Well Center in mm
    :return radius: Radius of the circle in mm
    
    '''
    
    # z_edge += 0.25                      # Offset the z axis
    image = imageProcessor('anh1.jpg')  # Replace with image Read
    
    backup_image = image.original_image.copy()
    
    contours = image.contours
    contours, ct_max = filter_contour(contours)
    
    out_image = image.copy()
    cv2.drawContours(out_image, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(f'teneImage/final_image_z_{str(z_edge)}.jpg',out_image)
    print("Saved the sharpest Final Contour Image")
    
    if len(ct_max) == 1 :
        ct_max = ct_max[0]
        
    save_oocyte_with_circle(backup_image,contours)

    ct_max = np.squeeze(ct_max)
    A = ct_max[len(ct_max)//10]
    B = ct_max[len(ct_max)//5]
    C = ct_max[len(ct_max)//2]
    print(f'[findCenter] The three coord A, B, C approximately chosen: {A}, {B}, {C}')
    
    center_pixel ,radius = circleApprox(A,B,C)
    
    x_center_pixel, y_center_pixel = center_pixel
    print(f'\n\n Center of the Spiral (pixel): ({round(x_center_pixel,2)},{round(y_center_pixel)}), radius la :{radius} pixel\n')

    dx = (x_center_pixel - 432)* pixel_per_mm  # 432 is half a frame along the x axis, the center of the cam
    dy = (y_center_pixel - 324)* pixel_per_mm  # 324 is half a frame along the y axis, the center of the cam

    x_center_mm = x_edge + dy # Y pixel axis is reversed
    y_center_mm = y_edge - dx 
    
    # move_motor_request_x(x_center_mm)
    # waiting_x()
    # move_motor_request_y(y_center_mm)
    # waiting_y()
    
    time.sleep(2)
    
    return x_center_mm, y_center_mm, radius

# findCenter(x_from_detectWellEdge, y_from_detectWellEdge, z_from_focusWellEdge)




# -----------------------------------------------------------------------------------------------
# ----------------------------------- Detect Oocyte ---------------------------------------------
# -----------------------------------------------------------------------------------------------

def createSpiral(x_center, y_center, R, num_points=100, a=0, b=2):
    points = []
    theta = 0
    delta_theta = 2 * np.pi / num_points  # Control how fine-grained the points are
    
    while True:
        # Compute radial distance r for this theta
        r = a + b * theta
        
        # Stop if the spiral reaches or exceeds the radius of the circle
        if r > R:
            break
        
        # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x = x_center + r * np.cos(theta)
        y = y_center + r * np.sin(theta)
        
        # Append the point to the list
        points.append((x, y))
        
        # Increment theta by the step size
        theta += delta_theta
    
    return points

def detectOocyte(x_center, y_center, radius_mm):
    '''
    Find Oocyte in a Spiral Pattern
    
    :param x_center: X coord of the center of the spiral 
    :param y_center: Y coord of the center of the spiral 
    :param radius: The radius of the circle (in mm) 

    :return: 
    '''
    
    x_true = -1
    y_true = -1
    z_num =  -1
    
    A = createSpiral(x_center, y_center, radius_mm*pixel_per_mm)      # Create a Spiral Trajectory
    # print(f"Spiral Trajectory : {A}")

    for i in range(0,len(A)-1):
        if endthread == True:
            break
        
        x_scan, y_scan = A[i]
        print(f'Center of Spiral of the first Oocyte Well: {x_scan, y_scan}')

        # move_motor_request_x(x)
        # move_motor_request_y(y)
        # waitingxy()

        # current_x , current_y, current_z = readposition()
        # print(f'\n[TAKE 1] After going to the starting center position \n The current coord is: ({current_x:.5f},{current_y:.5f},{current_z:.5f})')

        image = imageProcessor("anh1.jpg")              # Replace with image read
        
        kernel = np.ones((4,4), np.uint8)
        canny_image_dilate = cv2.dilate(image.canny_image, kernel, iterations=1)
                                       
        added_image = cv2.add(canny_image_dilate, image.gray_image)
        cv2.imwrite(f'teneImage\detectOocyte\image_circle_{i}.jpg', added_image)

        radius_list = []
        
        try:
            circles = cv2.HoughCircles(added_image, cv2.HOUGH_GRADIENT, dp=1, minDist=700, param1=60, param2=20, minRadius= 50, maxRadius=100) #max 400 min 80
            circles = np.uint16(np.around(circles))     # Detect Circles and round up the Coords [(x_center,y_center,radius)]
            
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                radius_list.append(radius)
                
                cv2.circle(image.original_image, center, radius, (0, 255, 0), 2)
                
                dx = round(abs(((center[0] - 432))*pixel_per_mm),3)
                dy = round(abs(((center[1] - 324))*pixel_per_mm),3)
                # print("X distance to move horizontally: ", dx)
                # print("Y distance to move vertically  : ", dy)
                
                if center[0] > 432:
                    if center[1] < 324:
                        x_true = round(x_scan + dy, 3)
                        y_true = round(y_scan - dx, 3)

                    else:
                        x_true = round(x_scan - dy, 3)
                        y_true = round(y_scan - dx, 3)

                else:
                    if center[1]  < 324:
                        x_true = round(x_scan + dy, 3)
                        y_true = round(y_scan + dx, 3)
 
                    else:
                        x_true = round(x_scan - dy, 3)
                        y_true = round(y_scan + dx, 3)
                
            flag = True
            
            print(f'\nDetected {len(circles)} circles, with a Radius list: {radius_list}')
            
            break
            
        except:
            pass

    # if x_true != -1:
        # move_motor_request_x(x_true)
        # move_motor_request_y(y_true)
        # waitingx()
        # waitingy()
        
    return x_true, y_true

# detectOocyte(x_from_findCenter, y_from_findCenter, radius_from_findCenter)





# -----------------------------------------------------------------------------------------------
# ----------------------------------- Focus Oocyte ----------------------------------------------
# -----------------------------------------------------------------------------------------------

def focusOocyte(z_start, T):
    '''
    Focus on Oocyte
    
    :param z_start: Z Starting Coord (after focusing on Well Edge)
    :param T: Fuzzy param
    
    :return: The final sharpest z coord
    '''
    
    z_scan = z_start
    
    u1 = 0
    u2 = 0
    u3 = 0
    u4 = 0
    u5 = 0
    old_tene = 0
    heso = 1
    uz = 0
    z_out = 0
    
    while endthread == False:
        image = imageProcessor("anh1.jpg")      # Replace with image read
        tene = tenengrad(image.median_blur_image)
        
        print(f"Tenegrad value: {tene}")
        print(f"OLD Tenegrad value: {old_tene}")

        if (z_scan >= (z_start + 1.2)) | (z_scan <= (z_start - 1.2)):
            break
        
        if oldtene > (tene + 0.2*oldtene):
            if heso == 1:
                oldtene = 0
            heso = -1
            
        print("HESO : ", heso)
        if tene == 0:
            break
        if (heso == -1) & ((oldtene - tene) > (oldtene*0.3)) & (oldtene >= 3*T/5):
            break
        
        if tene > oldtene:
            z_out = z_scan
            oldtene = tene
            # print(z_out)
        
        t1 = T/5
        t2 = 2*T/5
        t3 = 3*T/5
        t4 = 4*T/5
        t5 = 5*T/5
        
        # Fuzzy Logic
        if tene <= t1:
            u1 = 1
            u2 = 0
            u3 = 0
            u4 = 0
            u5 = 0
        elif (tene > t1) & (tene <= t2):
            u1 = ((t2 - tene) /t1)
            u2 = 1- u1
            u3 = 0
            u4 = 0
            u5 = 0
        elif (tene > t2) & (tene <= t3):
            u1 = 0
            u2 = (t3 - tene)/t1
            u3 = 1 - u2
            u4 = 0
            u5 = 0
        elif (tene > t3) & (tene <= t4):
            u1 = 0
            u2 = 0
            u3 = (t4 - tene)/t1
            u4 = 1 - u3
            u5 = 0
        elif (tene > t4) & (tene <= t5):
            u1 = 0
            u2 = 0
            u3 = 0
            u4 = (t5 - tene)/t1
            u5 = 1 - u4
        elif tene > t5:
            u1 = 0
            u2 = 0
            u3 = 0
            u4 = 0
            u5 = 1

        if tene > (t5*1.02):
            break
        
        if ( u1 + u2  + u3 + u4 + u5 ) == 0:
            break
        else:
            uz = heso*round(((u1*0.05 + u2*0.00775 + u3*0.0055 + u4*0.00325 + u5*0.001  )/ ( u1 + u2  + u3 + u4 + u5 )),3)
        
        z_scan += uz
        # move_motor_request_z(z)
        # waitingz()
        
        # if z_out != 0:
            # move_motor_request_z(z_out)
            # waitingz()
        
    return z_out

# focusOocyte(z_from_focusWellEdge, T = 200)






    
    
