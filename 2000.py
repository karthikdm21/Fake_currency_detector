


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tkinter import *
from tkinter.ttk import Progressbar
import time

# Resizing the Plots
plt.rcParams["figure.figsize"] = (12, 12)
myProgress = 0.0

%store -r path

print('Path of input image: ', path)
test_img = cv2.imread(path)

# Pre-processing
test_img = cv2.resize(test_img, (1165, 455))
blur_test_img = cv2.GaussianBlur(test_img, (5, 5), 0)
gray_test_image = cv2.cvtColor(blur_test_img, cv2.COLOR_BGR2GRAY)

def preprocessing():
    plt.imshow(gray_test_image, 'gray')
    plt.title('Input image after pre-processing')
    plt.show()
    progress['value'] = 5
    ProgressWin.update_idletasks()

def calculateSSIM(template_img, query_img):
    min_w = min(template_img.shape[1], query_img.shape[1])
    min_h = min(template_img.shape[0], query_img.shape[0])
    img1 = cv2.resize(template_img, (min_w, min_h))
    img2 = cv2.resize(query_img, (min_w, min_h))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 2, 1)
    plt.imshow(img1, 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, 'gray')
    plt.show()
    score = ssim(img1, img2)
    return score

def computeORB(template_img, query_img):
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    kpts1, descs1 = orb.detectAndCompute(template_img, None)
    kpts2, descs2 = orb.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = template_img.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M) if M is not None else None
    return dst, dst_pts, kpts1, kpts2, dmatches

search_area_list = [[200, 270, 160, 330], [1050, 1500, 250, 400], [50, 400, 0, 100], [750, 1050, 0, 100], [850, 1050, 280, 380], [700, 820, 290, 370], [400, 650, 0, 100]]
feature_area_limits_list = [[10000, 14000], [9000, 15000], [17000, 21500], [19000, 28000], [17500, 23000], [6500, 9000], [10000, 16000]]
score_set_list = []
best_extracted_img_list = []
avg_ssim_list = []
NUM_OF_FEATURES = 7

def testFeature_1_2_7():
    global myProgress, score_set_list, best_extracted_img_list, avg_ssim_list
    myProgress = progress['value']
    for j in range(NUM_OF_FEATURES):
        print(f'ANALYSIS OF FEATURE {j+1}')
        score_set = []
        max_score = -1
        max_score_img = None
        for i in range(6):
            print(f'---> Template {i+1} :')
            template_path = f'Dataset/2000_Features Dataset/Feature {j+1}/{i+1}.jpg'
            template_img = cv2.imread(template_path)
            template_img_blur = cv2.GaussianBlur(template_img, (5, 5), 0)
            template_img_gray = cv2.cvtColor(template_img_blur, cv2.COLOR_BGR2GRAY)
            test_img_mask = gray_test_image.copy()
            search_area = search_area_list[j]
            test_img_mask[:, :search_area[0]] = 0
            test_img_mask[:, search_area[1]:] = 0
            test_img_mask[:search_area[2], :] = 0
            test_img_mask[search_area[3]:, :] = 0
            dst, dst_pts, kpts1, kpts2, dmatches = computeORB(template_img_gray, test_img_mask)
            if dst is None:
                print('An Error occurred - Homography matrix is of NoneType')
                continue
            query_img = test_img.copy()
            res_img1 = cv2.polylines(query_img, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
            res_img2 = cv2.drawMatches(template_img, kpts1, res_img1, kpts2, dmatches[:20], None, flags=2)
            (x, y, w, h) = cv2.boundingRect(dst)
            min_area, max_area = feature_area_limits_list[j]
            feature_area = w * h
            if feature_area < min_area or feature_area > max_area:
                (x, y, w, h) = cv2.boundingRect(dst_pts)
                feature_area = w * h
                if feature_area < min_area or feature_area > max_area:
                    print('Template Discarded- Area of extracted feature is outside permitted range!')
                    continue
            cv2.rectangle(res_img1, (x, y), (x + w, y + h), (0, 255, 0), 3)
            plt.rcParams["figure.figsize"] = (16, 16)
            plt.subplot(1, 2, 1)
            plt.imshow(res_img2)
            plt.subplot(1, 2, 2)
            plt.imshow(res_img1)
            plt.show()
            crop_img = blur_test_img[y:y + h, x:x + w]
            plt.rcParams["figure.figsize"] = (5, 5)
            score = calculateSSIM(template_img_blur, crop_img)
            score_set.append(score)
            print(f'SSIM score: {score}\n')
            if score > max_score:
                max_score = score
                max_score_img = crop_img
            myProgress += (75.0 / (NUM_OF_FEATURES * 6))
            progress['value'] = myProgress
            ProgressWin.update_idletasks()
        score_set_list.append(score_set)
        print(f'SSIM score set of Feature {j+1}: {score_set}\n')
        if score_set:
            avg_ssim_list.append(sum(score_set) / len(score_set))
            print(f'Average SSIM of Feature {j+1}: {sum(score_set) / len(score_set)}\n')
        else:
            print('No SSIM scores were found for this feature!')
            avg_ssim_list.append(0.0)
            print(f'Average SSIM of Feature {j+1}: 0\n')
        best_extracted_img_list.append([max_score_img, max_score])
    print('Final Score-set list:\n')
    for x in range(len(score_set_list)):
        print(f'Feature {x+1}: {score_set_list[x]}')
    print('\nFinal Average SSIM list for each feature:\n')
    for x in range(len(avg_ssim_list)):
        print(f'Feature {x+1}: {avg_ssim_list[x]}')

left_BL_result = []
right_BL_result = []
result_list = []
number_panel_result = []

def testFeature_8():
    plt.rcParams["figure.figsize"] = (5, 5)
    print('\nANALYSIS OF FEATURE 8 : LEFT BLEED LINES\n')
    crop = test_img[80:230, 10:30]
    img = crop.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh, 'gray')
    plt.show()
    whitePixelValue = 255
    blackPixelValue = 0
    width = thresh.shape[1]
    result = []
    num_of_cols = 0
    print('Number of black regions found in each column: ')
    for j in range(width):
        col = thresh[:, j:j+1]
        count = 0
        for i in range(len(col) - 1):
            pixel1_value = col[i][0]
            pixel2_value = col[i + 1][0]
            if pixel1_value != 0 and pixel1_value != 255:
                pixel1_value = 255
            if pixel2_value != 0 and pixel2_value != 255:
                pixel2_value = 255
            if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                count += 1
        if 0 < count < 10:
            print(count)
            result.append(count)
            num_of_cols += 1
        else:
            print(f'{count} Erroneous -> discarded')
    print(f'\nNumber of columns examined: {width}')
    print(f'Number of non-erroneous columns found: {num_of_cols}')
    average_count = sum(result) / num_of_cols if num_of_cols != 0 else -1
    print(f'\nAverage number of black regions is: {average_count}')
    global left_BL_result
    left_BL_result = [thresh, average_count]
    progress['value'] = 80
    ProgressWin.update_idletasks()

def testFeature_9():
    plt.rcParams["figure.figsize"] = (5, 5)
    print('\nANALYSIS OF FEATURE 9 : RIGHT BLEED LINES\n')
    crop = test_img[90:230, 1140:1160]
    img = crop.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh, 'gray')
    plt.show()
    whitePixelValue = 255
    blackPixelValue = 0
    width = thresh.shape[1]
    result = []
    num_of_cols = 0
    print('Number of black regions found in each column: ')
    for j in range(width):
        col = thresh[:, j:j+1]
        count = 0
        for i in range(len(col) - 1):
            pixel1_value = col[i][0]
            pixel2_value = col[i + 1][0]
            if pixel1_value != 0 and pixel1_value != 255:
                pixel1_value = 255
            if pixel2_value != 0 and pixel2_value != 255:
                pixel2_value = 255
            if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                count += 1
        if 0 < count < 10:
            print(count)
            result.append(count)
            num_of_cols += 1
        else:
            print(f'{count} Erroneous -> discarded')
    print(f'\nNumber of columns examined: {width}')
    print(f'Number of non-erroneous columns found: {num_of_cols}')
    average_count = sum(result) / num_of_cols if num_of_cols != 0 else -1
    print(f'\nAverage number of black regions is: {average_count}')
    global right_BL_result
    right_BL_result = [thresh, average_count]
    progress['value'] = 85
    ProgressWin.update_idletasks()

plt.rcParams["figure.figsize"] = (7, 7)
crop = gray_test_image[360:440, 760:1080]
crop_bgr = test_img[360:440, 760:1080]

def testFeature_10():
    plt.imshow(crop_bgr)
    plt.show()
    plt.rcParams["figure.figsize"] = (7, 7)
    print('\n\nANALYSIS OF FEATURE 10 : NUMBER PANEL \n')
    test_passed = False
    res_img_list = []
    count = 0
    num = 1
    for thresh_value in range(90, 155, 5):
        _, thresh = cv2.threshold(crop, thresh_value, 255, cv2.THRESH_BINARY)
        print(f'---> Threshold {num} with Threshold value {thresh_value} :')
        num += 1
        copy = crop_bgr.copy()
        img = cv2.bitwise_and(crop, crop, mask=thresh)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        h_img, w_img = img.shape[:2]
        bounding_rect_list = []
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if x != 0:
                bounding_rect_list.append([x, y, w, h])
        bounding_rect_list.sort()
        min_area = 150
        res_list = []
        for i in range(len(bounding_rect_list)):
            if i >= len(bounding_rect_list):
                break
            if bounding_rect_list[i][2] * bounding_rect_list[i][3] > min_area:
                res_list.append(bounding_rect_list[i])
        i = 0
        while i < len(res_list):
            [x, y, w, h] = res_list[i]
            j = i + 1
            while j < len(res_list):
                [x0, y0, w0, h0] = res_list[j]
                if (x + w) >= x0 + w0:
                    res_list.pop(j)
                else:
                    break
            i += 1
        i = 0
        while i < len(res_list):
            [x, y, w, h] = res_list[i]
            if (h_img - (y + h)) > 40:
                res_list.pop(i)
            elif h < 17:
                res_list.pop(i)
            else:
                i += 1
        for rect in res_list:
            [x, y, w, h] = rect
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if len(res_list) == 9:
            test_passed = True
            res_img_list.append(copy)
            count += 1
            print('Test Successful: 9 letters found!')
        else:
            print('Unsuccessful!')
        if count == 3:
            break
    if count == 0:
        best_img = crop_bgr
    elif count == 1:
        best_img = res_img_list[0]
    elif count == 2:
        best_img = res_img_list[1]
    else:
        best_img = res_img_list[2]
    if test_passed:
        print('Test Passed!- 9 characters were detected in the serial number panel.')
        plt.imshow(best_img)
        plt.show()
    else:
        print('Test Failed!- 9 characters were NOT detected in the serial number panel.')
    global number_panel_result
    number_panel_result = [best_img, test_passed]
    progress['value'] = 90
    ProgressWin.update_idletasks()

def testResult():
    plt.rcParams["figure.figsize"] = (3, 3)
    print('\n\nRESULT ANALYSIS\n')
    min_ssim_score_list = [0.45, 0.4, 0.45, 0.45, 0.5, 0.4, 0.5]
    global result_list
    result_list = []
    successful_features_count = 0
    for i in range(NUM_OF_FEATURES):
        avg_score = avg_ssim_list[i]
        img, max_score = best_extracted_img_list[i]
        status = False
        min_allowed_score = min_ssim_score_list[i]
        if avg_score >= min_allowed_score or max_score >= 0.79:
            status = True
            successful_features_count += 1
            print(f'Feature {i+1}: Successful')
        else:
            status = False
            print(f'Feature {iThe server is busy. Please try again later.