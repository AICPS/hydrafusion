import torch
import torch.nn as nn
import numpy as np
from ensemble_boxes import *
import copy



'''
Represents the final fusion block in HydraFusion.
Different fusion types currently implemented:
    1 = WBF, 2 = NMS, 3 = Soft-NMS
'''
class FusionBlock(nn.Module):
    def __init__(self, config, fusion_type, weights, iou_thr, skip_box_thr, sigma, alpha):
        super(FusionBlock, self).__init__()
        
        self.config = config
        self.weights = weights #output losses should be a list for losses for each branch
        # This weights variable can be tuned to favor certain branches over others
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.sigma = sigma
        self.fusion_type = fusion_type
        self.alpha = alpha

    ''' Function from the RADIATE SDK'''
    def transform(self, LidarToCamR, LidarToCamT):
        Rx = self.RX(LidarToCamR)
        Ry = self.RY(LidarToCamR)
        Rz = self.RZ(LidarToCamR)

        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]]).astype(np.float)
        R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

        LidarToCam = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                               [R[1, 0], R[1, 1], R[1, 2], 0.0],
                               [R[2, 0], R[2, 1], R[2, 2], 0.0],
                               [LidarToCamT[0], LidarToCamT[1], LidarToCamT[2], 1.0]]).T
        return LidarToCam

    ''' Function from the RADIATE SDK'''
    def RX(self, LidarToCamR):
        thetaX = np.deg2rad(LidarToCamR[0])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(thetaX), -np.sin(thetaX)],
                       [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float)
        return Rx

    ''' Function from the RADIATE SDK'''
    def RY(self, LidarToCamR):
        thetaY = np.deg2rad(LidarToCamR[1])
        Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                       [0, 1, 0],
                       [-np.sin(thetaY), 0, np.cos(thetaY)]])
        return Ry

    ''' Function from the RADIATE SDK'''
    def RZ(self, LidarToCamR):
        thetaZ = np.deg2rad(LidarToCamR[2])
        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                       [np.sin(thetaZ), np.cos(thetaZ), 0],
                       [0, 0, 1]]).astype(np.float)
        return Rz
    
    ''' Function from the RADIATE SDK'''
    """ 
    method to project the bounding boxes to the camera
        :param annotations: the annotations for the current frame
        :type annotations: list
        :param intrinsict: intrisic camera parameters
        :type intrinsict: np.array
        :param extrinsic: extrinsic parameters
        :type extrinsic: np.array
        :return: dictionary with the list of bbounding boxes with camera coordinate frames
        :rtype: dict
    """
    def project_bboxes_to_camera(self, annotations, labels, intrinsict, extrinsic):
        
        bboxes_3d = []
        heights = {'car': 1.5,
                        'bus': 3,
                        'truck': 2.5,
                        'pedestrian': 1.8,
                        'van': 2,
                        'group_of_pedestrians': 1.8,
                        'motorbike': 1.5,
                        'bicycle': 1.5,
                        'vehicle': 1.5
                        }
        class_names=['background', 'car', 'van', 'truck', 'bus', 'motorbike', 'bicycle','pedestrian','group_of_pedestrians']
        ii = 0
        for object in annotations:
            obj = {}
            bb = np.float64(object.cpu())
            idx = np.int(labels[ii].cpu())
            height = heights[class_names[idx]]
            #converts [x1,y1,x2,y2] to [x1,y1,w,h]:
            bb = np.array([ bb[0], bb[1], (bb[2]-bb[0]), (bb[3]-bb[1]) ])
            rotation = 0
            bbox_3d = self.__get_projected_bbox(bb, rotation, intrinsict, extrinsic, height)
            obj['bbox_3d'] = bbox_3d
            bboxes_3d.append(obj)
            ii = ii + 1

        return bboxes_3d

    ''' Function from the RADIATE SDK'''
    def __get_projected_bbox(self, bb, rotation, cameraMatrix, extrinsic, obj_height=2):
        """get the projected boundinb box to some camera sensor
        """
        rotation = np.deg2rad(-rotation)
        res = 0.173611 #self.config['radar_calib']['range_res']
        cx = bb[0] + bb[2] / 2
        cy = bb[1] + bb[3] / 2
        T = np.array([[cx], [cy]])
        pc = 0.2
        bb = [bb[0]+bb[2]*pc, bb[1]+bb[3]*pc, bb[2]-bb[2]*pc, bb[3]-bb[3]*pc]

        R = np.array([[np.cos(rotation), -np.sin(rotation)],
                      [np.sin(rotation), np.cos(rotation)]])

        points = np.array([[bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]],
                           [bb[0], bb[1] + bb[3]],
                           [bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]]]).T

        points = points - T
        points = np.matmul(R, points) + T
        points = points.T

        points[:, 0] = points[:, 0] - 576 #self.config['radar_calib']['range_cells']
        points[:, 1] = 576 - points[:, 1] #self.config['radar_calib']['range_cells'] - points[:, 1]
        points = points * res

        points = np.append(points, np.ones(
            (points.shape[0], 1)) * -1.7, axis=1)
        p1 = points[0, :]
        p2 = points[1, :]
        p3 = points[2, :]
        p4 = points[3, :]

        p5 = np.array([p1[0], p1[1], p1[2] + obj_height])
        p6 = np.array([p2[0], p2[1], p2[2] + obj_height])
        p7 = np.array([p3[0], p3[1], p3[2] + obj_height])
        p8 = np.array([p4[0], p4[1], p4[2] + obj_height])
        points = np.array([p1, p2, p3, p4, p1, p5, p6, p2, p6,
                           p7, p3, p7, p8, p4, p8, p5, p4, p3, p2, p6, p3, p1])

        points = np.matmul(np.append(points, np.ones(
            (points.shape[0], 1)), axis=1), extrinsic.T)

        points = (points / points[:, 3, None])[:, 0:3]

        filtered_indices = []
        for i in range(points.shape[0]):
            #if (points[i, 2] > 0 and points[i, 2] < self.config['max_range_bbox_camera']):
            if (points[i, 2] > 0 and points[i, 2] < 100):
                filtered_indices.append(i)

        points = points[filtered_indices]

        fx = cameraMatrix[0, 0]
        fy = cameraMatrix[1, 1]
        cx = cameraMatrix[0, 2]
        cy = cameraMatrix[1, 2]

        xIm = np.round((fx * points[:, 0] / points[:, 2]) + cx).astype(np.int)
        yIm = np.round((fy * points[:, 1] / points[:, 2]) + cy).astype(np.int)

        proj_bbox_3d = []
        for ii in range(1, xIm.shape[0]):
            proj_bbox_3d.append([xIm[ii], yIm[ii]])
        proj_bbox_3d = np.array(proj_bbox_3d)
        return proj_bbox_3d


    def forward(self, output_losses, output_detections, fusion_sweep=False):
        #init some parameters:
        ylim = 376; xlim = 672
        fxl = 3.379191448899105e+02; fyl=  3.386957068549526e+02
        fxr = 337.873451599077 ; fyr = 338.530902554779
        cxl =  3.417366010946575e+02; cyl= 2.007359735313929e+02
        cxr = 329.137695760749 ; cyr = 186.166590759716
        left_cam_mat = np.array([[fxl, 0, cxl],
                                    [0, fyl, cyl],
                                    [0,  0,  1]])
        right_cam_mat = np.array([[fxr, 0, cxr],
                                    [0, fyr, cyr],
                                    [0,  0,  1]])

        RadarT = np.array([0.0, 0.0, 0.0])
        RadarR = np.array([0.0, 0.0, 0.0])
        LidarT = np.array([0.6003, -0.120102, 0.250012])
        LidarR = np.array([0.0001655, 0.000213, 0.000934])
        LeftT = np.array([0.34001, -0.06988923, 0.287893])
        LeftR = np.array([1.278946, -0.530201, 0.000132])
        RightT = np.array([0.4593822, -0.0600343, 0.287433309324])
        RightR = np.array([0.8493049332, 0.37113944, 0.000076230])

        RadarToLeftT = RadarT - LeftT; RadarToRightT = RadarT - RightT
        RadarToLeftR = RadarR - LeftR; RadarToRightR = RadarR - RightR
        RadarToLeft = self.transform(RadarToLeftR, RadarToLeftT)
        RadarToRight = self.transform(RadarToRightR, RadarToRightT)

        LidarToLeftT = LidarT - LeftT; LidarToRightT = LidarT - RightT
        LidarToLeftR = LidarR - LeftR; LidarToRightR = LidarR - RightR
        LidarToLeft = self.transform(LidarToLeftR, LidarToLeftT)
        LidarToRight = self.transform(LidarToRightR, LidarToRightT)

        num_branches = len(output_detections) #get the number of branches
        good_branches = [] #will return a list of good branches with detections
        bboxes_3d = {} ; bboxes_3dl = {}
        output_detections_copy = copy.deepcopy(output_detections)
        for i in output_detections_copy:
            # Step 1: Handle case where branches have no detections: moved to later to handle radar,lidar,radar_lidar out of range issues
            if output_detections_copy[i][0]['boxes'].numel(): 
                good_branches.append(i)
            
            if i=='radar': #radar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3d = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'], output_detections_copy[i][0]['labels'],right_cam_mat, RadarToRight)
                j = 0
                for k in range(len(bboxes_3d)):
                    #Added fix for empty boxes that were out of range of camera during transformation
                    if not(bboxes_3d[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3d[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3d[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3d[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3d[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue

            if i=='lidar': #lidar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3dl = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'],output_detections_copy[i][0]['labels'],right_cam_mat, LidarToRight)
                j = 0
                for k in range(len(bboxes_3dl)):
                    if not(bboxes_3dl[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3dl[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3dl[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3dl[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3dl[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
            
            if i=='radar_lidar': #lidar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3dlr = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'], output_detections_copy[i][0]['labels'],right_cam_mat, RadarToRight)
                j = 0
                for k in range(len(bboxes_3dlr)):
                    if not(bboxes_3dlr[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3dlr[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3dlr[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3dlr[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3dlr[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue

        # Step 3: only use good detections
        good_branches2 = []
        for i in output_detections_copy:
            if output_detections_copy[i][0]['boxes'].numel():
                good_branches2.append(i)    
        good_detections = []
        for i in good_branches2:
            good_detections.append(output_detections_copy[i])
        

        # Step 4: Normalize the bbox pixel values for fusion
        #boxes_list should be a list of floats example for two branches (branch one has two objects, branch one has one):
        # Example [[ [0.00, 0.51, 0.81, 0.91], [0.10, 0.31, 0.71, 0.61], ], [ [0.04, 0.56, 0.84, 0.92],]]
        # scores_list # Example: [[0.9, 0.8], [0.3]]
        # labels_list  # Example: [[0, 1], [1]]

        #Size of the image: defined above: xlim,ylim
        
        boxes_list = []; scores_list = []; labels_list = []
        for i in good_detections:
            branch_boxes = []; branch_scores = []; branch_labels = []
            for j in np.float64(i[0]['boxes'].cpu()):
                if j[3] > ylim: j[3] = round(j[3]) #handles case where j[3] = 376.0001
                if j[2] > xlim: j[2] = round(j[2])
                branch_boxes.append([ j[0]/xlim , j[1]/ylim , j[2]/xlim , j[3]/ylim  ])

            boxes_list.append(branch_boxes)
            scores_list.append(i[0]['scores'].cpu().numpy().tolist())
            labels_list.append(i[0]['labels'].cpu().numpy().tolist())


        if not(fusion_sweep):
            if not(bool(boxes_list)): #checks if there are any bounding boxes
                fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
            elif not(bool(boxes_list[0])):
                fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
            else:
                fboxes1, fscores1, flabels1 = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr)
                fboxes2, fscores2, flabels2 = nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr)
                fboxes3, fscores3, flabels3 = soft_nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr, sigma=self.sigma, thresh=self.skip_box_thr) 

            # Step 5b: rescale up the predictions
            ffboxes1 = []
            for i in fboxes1:
                ffboxes1.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
            ffboxes2 = []
            for i in fboxes2:
                ffboxes2.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
            ffboxes3 = []
            for i in fboxes3:
                ffboxes3.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])

            # Step 6: Compile the results
            output_detections.update({'fused1':[{'boxes': torch.tensor(ffboxes1, device=self.config.device),'labels': torch.from_numpy(flabels1).to(self.config.device),'scores': torch.from_numpy(fscores1).to(self.config.device) }]})
            output_detections.update({'fused2':[{'boxes': torch.tensor(ffboxes2, device=self.config.device),'labels': torch.from_numpy(flabels2).to(self.config.device),'scores': torch.from_numpy(fscores2).to(self.config.device) }]})
            output_detections.update({'fused3':[{'boxes': torch.tensor(ffboxes3, device=self.config.device),'labels': torch.from_numpy(flabels3).to(self.config.device),'scores': torch.from_numpy(fscores3).to(self.config.device) }]})
        else: #fusion sweep
            #current default: iou_thr=0.4, skip_box_thr=0.01, sigma=0.5
            iou_thr_range = [0.4,0.5,0.6] 
            skip_box_thr_range  = [0.01 , 0.1 , 0.3]
            sigma_range = [0.1 , 0.25, 0.5]
            sweep_num = len(iou_thr_range)*len(skip_box_thr_range)*len(sigma_range)
            fff = 0
            for iou in iou_thr_range:
                for skip_b in skip_box_thr_range:
                    for sig in sigma_range: 
            
                        if not(bool(boxes_list)): #checks if there are any bounding boxes
                            fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
                        elif not(bool(boxes_list[0])):
                            fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
                        else:
                            fboxes1, fscores1, flabels1 = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou, skip_box_thr=skip_b)
                            fboxes2, fscores2, flabels2 = nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou)
                            fboxes3, fscores3, flabels3 = soft_nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou, sigma=sig, thresh=skip_b) 

                        # Step 5b: rescale up the predictions
                        ffboxes1 = []
                        for i in fboxes1:
                            ffboxes1.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
                        ffboxes2 = []
                        for i in fboxes2:
                            ffboxes2.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
                        ffboxes3 = []
                        for i in fboxes3:
                            ffboxes3.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])

                        # Step 6: Compile the results
                        fusion_key1 = 'fused1_' + 'iou_' + str(iou) +'_skip_' + str(skip_b)
                        fusion_key2 = 'fused2_' + 'iou_' + str(iou) 
                        fusion_key3 = 'fused3_' + 'iou_' + str(iou) +'_sig_'+ str(sig) + '_skip_' + str(skip_b)
                        output_detections.update({fusion_key1:[{'boxes': torch.tensor(ffboxes1, device=self.config.device),'labels': torch.from_numpy(flabels1).to(self.config.device),'scores': torch.from_numpy(fscores1).to(self.config.device) }]})
                        output_detections.update({fusion_key2:[{'boxes': torch.tensor(ffboxes2, device=self.config.device),'labels': torch.from_numpy(flabels2).to(self.config.device),'scores': torch.from_numpy(fscores2).to(self.config.device) }]})
                        output_detections.update({fusion_key3:[{'boxes': torch.tensor(ffboxes3, device=self.config.device),'labels': torch.from_numpy(flabels3).to(self.config.device),'scores': torch.from_numpy(fscores3).to(self.config.device) }]})
                        fff = fff + 1
        

        # Step 7: Compute the fused loss
        #TODO: calculate final loss
        final_loss = output_losses
        return final_loss, output_detections
