
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from numpy import dot
from scipy.linalg import inv, block_diag


class Tracker(): # class for Kalman Filter based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = [] # list to store the coordinates for a bounding box
        self.centres = [] # list to store the coordinates of previous centres
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)

        # These two are used so we can draw the appropriate bounding boxes later
        self.class_labels = -1 # Label for the class
        self.class_scores = -1 # Score for the class

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state=[]
        self.dt = 0.75   # time interval

        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])

        # Measurement matrix, assuming we can only measure the coordinates

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])


        # Initialize the state covariance
        self.L = 100.0
        self.P = np.diag(self.L*np.ones(8))


        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/2., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_ratio = 1.0/16.0
        self.R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)


    def update_R(self):
        R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)


    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates
                                     #(pixel values)


    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)



class Detector(object):
    def __init__(self):
        self.boxes = []

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0 ] *height), int(box[1 ] *width), int(box[2 ] *height), int(box[3 ] *width)]
        return np.array(box_pixel)

    def get_localization(self ,boxes ,scores, classes, image):
        """Determines the location of the classes in the image.

        Args:
            boxes: Bounding boxes detected.
            scores: Scores for the bounding boxes.
            classes: Class index.

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]
            idx_vec: Indices of the boxes that were successfully detected.

        """
        idx_vec = []
        tmp_boxes =[]


        for idx, bb in enumerate(boxes):
            dim = image.shape[0:2]
            box = self.box_normal_to_pixel(bb, dim)
            tmp_boxes.append(box)
            idx_vec.append(idx)


        self.boxes = tmp_boxes

        return self.boxes ,idx_vec


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;


def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);


def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)


def convert_to_pixel(box_yolo, img, crop_range):
    '''
    Helper function to convert (scaled) coordinates of a bounding box
    to pixel coordinates.

    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041,
    0.36866588651069609)

    crop_range: specifies the part of image to be cropped
    '''

    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape

    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w / 2.) * (xmax - xmin) + xmin)
    top = int((box.y - box.h / 2.) * (ymax - ymin) + ymin)

    width = int(box.w * (xmax - xmin))
    height = int(box.h * (ymax - ymin))

    # Deal with corner cases
    if left < 0:  left = 0
    if top < 0:   top = 0

    # Return the coordinates (in the unit of the pixels)

    box_pixel = np.array([left, top, width, height])
    return box_pixel


def convert_to_cv2bbox(bbox, img_dim=(1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])

    return (left, top, right, bottom)


def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    # box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x = 'x=' + str((left + right) / 2)
        cv2.putText(img, text_x, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y = 'y=' + str((top + bottom) / 2)
        cv2.putText(img, text_y, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img



class GlobalTracker(object):
    def __init__(self, max_age = 9):
        self.det = Detector()

        self.max_age = max_age  # no.of consecutive unmatched detection before
                     # a track is deleted

        self.min_hits = 0  # no. of consecutive matches needed to establish a track

        self.tracker_list =[] # list for trackers

        # Modified list for tracker IDs.
        self.last_track_id = 0


    def assign_detections_to_trackers(self,trackers, detections, iou_thrd = 0.3):
        '''
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        '''

        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t,trk in enumerate(trackers):
            #trk = convert_to_cv2bbox(trk)
            for d,det in enumerate(detections):
             #   det = convert_to_cv2bbox(det)
                IOU_mat[t,d] = box_iou2(trk,det)

        # Produces matches
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)

        matched_idx = linear_sum_assignment(-IOU_mat)
        matched_idx = np.asarray(matched_idx)
        matched_idx = np.transpose(matched_idx)

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if(IOU_mat[m[0],m[1]]<iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


    def pipeline(self, boxes,scores,classes,img, iou_threshold = 0.15, return_tracker_id = True):
        """Pipeline function for detection and tracking

        Args:
            boxes: Bounding boxes detected.
            scores: Scores for the bounding boxes.
            classes: Class index.
            img: Input image
            iou_threshold: Detection overlap threshold
            return_tracker_id: If enabled, will return the tracker id. Used for compatibility for older versions
            of our code.

        Returns:
            o_boxes: Tracked bounding boxes. Size (N)
            out_scores_arr: Corresponding scores. Size (N)
            out_classes_arr: Corresponding classes. Size (N)
            img: Output Image. Will be removed in a future revision.

        """
        z_box, idx_vec = self.det.get_localization(boxes,
                                    scores,
                                    classes,
                                    img)
        x_box =[]

        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)


        matched, unmatched_dets, unmatched_trks \
        = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = iou_threshold)

        # Deal with matched detections
        if matched.size >0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                tmp_trk.centres.append([
                    xx[1]+(xx[3]-xx[1])/2,
                    xx[0]+(xx[2]-xx[0])/2
                ])
                tmp_trk.hits += 1


        # Deal with unmatched detections
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.centres.append([
                    xx[1]+(xx[3]-xx[1])/2,
                    xx[0]+(xx[2]-xx[0])/2
                ])
                self.last_track_id += 1
                tmp_trk.id = self.last_track_id # assign an ID for the tracker
                tmp_trk.class_labels = classes[idx_vec[idx]] # assign the corresponding class label
                tmp_trk.class_scores = scores[idx_vec[idx]] # assign the corresponding class score
                self.tracker_list.append(tmp_trk)

        # Deal with unmatched tracks
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.centres.append([
                    xx[1]+(xx[3]-xx[1])/2,
                    xx[0]+(xx[2]-xx[0])/2
                ])


        # The list of tracks to be annotated
        out_boxes = []
        out_scores = []
        out_classes = []
        out_trk_id = []
        out_centres = []
        for trk in self.tracker_list:
            if ((trk.hits >= self.min_hits) and (trk.no_losses <=self.max_age)):
                # We need to rearrange the bounding box since
                # it currently puts it into a yolo format.
                # ymin, xmin, ymax, xmax
                bounding_boxes = trk.box
                out_boxes.append(bounding_boxes)
                out_scores.append(trk.class_scores)
                out_classes.append(trk.class_labels)
                out_trk_id.append(trk.id)
                out_centres.append(trk.centres)
        # Book keeping
        deleted_tracks = filter(lambda x: x.no_losses >self.max_age, self.tracker_list)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses<=self.max_age]


        # Before we return the result, we need to convert to an
        # array, then normalize the bounding box values between 0 and 1.
        # Convert them to arrays
        out_boxes_arr = np.asarray(out_boxes)
        out_scores_arr = np.asarray(out_scores)
        out_classes_arr = np.asarray(out_classes)
        out_trk_id_list = list(out_trk_id)


        # Normalize the box values. Copy to new array to prevent overwriting the old one.
        if out_boxes_arr.size > 0: # Check to ensure array isn't empty.
            o_boxes = np.zeros(shape=(out_boxes_arr.shape[0],out_boxes_arr.shape[1]))
            o_boxes[:,0] = out_boxes_arr[:,0] / float(img.shape[0])
            o_boxes[:,2] = out_boxes_arr[:,2] / float(img.shape[0])
            o_boxes[:,1] = out_boxes_arr[:,1] / float(img.shape[1])
            o_boxes[:,3] = out_boxes_arr[:,3] / float(img.shape[1])
        else:
            o_boxes = np.asarray([]) # Return an empty array

        # Normalize the centres values. Copy to new array to prevent overwriting the old one.
        o_centres = []
        for trk_centres in out_centres:
            out_trk_centres_arr = np.asarray(trk_centres)
            o_trk_centres = np.zeros(shape=(out_trk_centres_arr.shape[0],out_trk_centres_arr.shape[1]))
            o_trk_centres[:,0] = out_trk_centres_arr[:,0] / float(img.shape[1])
            o_trk_centres[:,1] = out_trk_centres_arr[:,1] / float(img.shape[0])
            o_centres.append(o_trk_centres)


        # Note here that we return 'o_boxes' rather than 'out_boxes'
        if return_tracker_id:
            return out_trk_id_list, o_boxes, o_centres, out_scores_arr, out_classes_arr
        else:
            return o_boxes, out_scores_arr, o_centres, out_classes_arr
