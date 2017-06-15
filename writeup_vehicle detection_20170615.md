## Vehicle Detection
### Project 5 - Self Driving Car Nanodegree

---

There were many ideas introduced for this project and I tried to implement as many as possible. The goals/workflow for this project were:

1. Define Features: Perform HOG feature extraction on a labeled training set images.
2. Define Classifier: Train a Linear SVM classifier.
3. Detect Vehicles: Use a sliding window technique combined with the classifier to determine whether the window contains a vehicle or not.
4. Remove Duplicates: Create a heatmap to remove duplicates.
5. Track Vehicles: Track vehicles and predict their future trajectory based on past centroid locations.
6. Apply to Video: Run the pipeline on each frame and detect vehicles.

[//]: # (Image References)
[image1]: ./output_images/car_original.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/output_22_0.png
[image5]: ./output_images/output_22_1.png
[image6]: ./output_images/output_22_2.png
[image7]: ./output_images/output_22_3.png
[image8]: ./output_images/output_22_4.png
[image9]: ./output_images/output_22_5.png
[video1]: ./P5_output.mp4

The rubric for this project is located [here](https://review.udacity.com/#!/rubrics/513/view).

---

### Histogram of Oriented Gradients (HOG)

#### 1, 2, 3. Explain how (and identify where in your code) you extracted HOG features from the training images, explain how you settled on your final choice of HOG parameters, and describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for extracting HOG features is located within the first major code block in the Jupyter notebook in the `FeatureTrack` class. HOG feature extraction is combined with spatial information and color channel histograms to perform feature extraction.

In order to determine the optimal HOG parameters a series of experiments were conducted using different ranges of orientations, pixels per cell, cells per block, and color spaces. After a suitable range was determined for each parameter, all possible combinations were used to train the classifier and the resulting accuracy was saved. Based on the ranges and experiments I ran, the optimal parameters were the YCbCr color space with 10 orientations, 8 pixels per cell, and 2 cells per block. 

The two images below show what HOG looks like when applied to one of the car images:

![Original car image][image1]
![HOG car image][image2]

The code for the calculation of HOG for each image is shown below:

```python
# Initialize hog features and pixels per cell
self.hog_features = []
self.pixels_per_cell = pixels_per_cell
        
# Initialize image and get height, width, and color of the image
self.image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
(self.h, self.w, self.c) = self.image.shape
        
# For each color channel run HOG and append to hog_features
for channel in range(self.c):
    self.hog_features.append(hog(self.image[:,:, channel], orientations = orientation,
                                         pixels_per_cell = (pixels_per_cell, pixels_per_cell), cells_per_block = (cells_per_block, cells_per_block),
                                         transform_sqrt=True, visualise=False, feature_vector=False))`

```

The HOG features for all color channels are concatenated.

By calculating the HOG features in this way, it allows us to get the HOG features for a specified region of the image without performing the feature extraction again. The code below shows how features are retrieved from a specified region of image given a x and y offset and size of the region. (`x_hog` and `y_hog` are the offesets and `s_hog` is the length of one side of the square region).

```python
s_hog = (s // self.pixels_per_cell) - 1
        
x_hog = max((x // self.pixels_per_cell) - 1, 0)
if x_hog + s_hog > self.hog_features.shape[2]:
    x_hog = self.hog_features.shape[2] - s_hog
else:
    x_hog

y_hog = max((y // self.pixels_per_cell) - 1, 0)

if y_hog + s_hog > self.hog_features.shape[1]:
    y_hog = self.hog_features.shape[1] - s_hog
else:
    y_hog
            
return np.ravel(self.hog_features[:, y_hog : y_hog + s_hog, x_hog : x_hog + s_hog, :, :, :])
```

Similar to what was done in the classroom, color channel histogram information was used with 32 bins and (0, 256) range. In addition the spatial information was gathered by resizing the image to 32x32 and flattening to a 1-D vector using `ravel()`

```python
# Compute the spacial vector
cv2.resize(image, size).ravel()

# Compute the histograms separately for each color channel
ch1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
ch2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
ch3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

# Concatenate the histograms into a single feature vector
return np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
```

Finally, calling `feature_vec()` returns the combined feature vector based on the HOG,color histogram, and spatial features. The whole image or specific regions (and region sizes) can be used.

In the Jupyter notebook, under the headings **Training Data, Features, and Scale**, the images are extracted and appended to arrays based on whether they are `cars` or `notcars`, the features are then extracted using `FeatureTrack`, and scaled using sklearn's `StandardScaler`.

Under **Train the Classifier** the data is split into training and validation sets using `train_test_split` and trained using a Linear SVC.

---

### Sliding Window Search

#### 1, 2. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows? Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Like in the classroom, I implemented a sliding window search. The sliding window search was implemeted using the `vehicle_detection()`, `detection_scaler()`, and `merge_car_detections()` in the `CarTrack()` class. Window size is decreased as the window moves from the bottom to the top of the image (as cars move from close to far away). In addition, only the lower part of the image is searched since there will not be any vehicles in the upper part of the image. The code below shows the various scaling and image size (part of the image to be searched) parameters and how the sliding window search is implemented.

```python
def vehicle_detection(self, image):
    scale_options = np.array([0.25, 0.5, 0.75, 0.8])
    y_axis_options = np.array([0.63, 0.6, 0.58, 0.55])

    image_detections = np.empty([0, 4], dtype=np.int64)

    for s, y in zip(scale_options, y_axis_options):
        scale_detections = self.detection_scaler(image, s, y, 64)
        image_detections = np.append(image_detections, scale_detections, axis=0)

    detections, self.heatmap = self.merge_car_detections(image_detections, image.shape, threshold=1)

    self.car_detect_history.append(detections)

def detection_scaler(self, image, scale, y, k):
    '''
    Run the classifier on an image with the specified scale

    Parameters
    ----------
    image  : Current frame
    scale  : Scale parameter
    y      : Uppermost y coord. of the windows
    k      : Window size

    Returns
    -------
    Bounding boxes for detections
    '''

    (h, w, d) = image.shape
    scaled_frame = resize((image / 255.).astype(np.float64), (int(h * scale), int(w * scale), d), preserve_range=True).astype(np.float32)
    feat_track = FeatureTrack(scaled_frame)

    (h, w, d) = scaled_frame.shape
    detections = np.empty([0, 4], dtype=np.int)
    y = int(h * y)
    s = k // 3
    x_range = np.linspace(0, w - k, (w + s) // s)

    for x in x_range.astype(np.int):
        features = feat_track.feature_vec(x, y, k)
        features = self.scaler.transform(np.array(features).reshape(1, -1))

        if self.classify.predict(features)[0] == 1:
            detections = np.append(detections, [[x, y, x + k, y + k]], axis=0)

    return (detections / scale).astype(np.int)
```

The image below shows an example of the sliding window search on one of the example images.

![Windows][image3]

In order to merge the various vehicle detections, a heatmap is generated for regions containing detected vehicles. As mentioned in the classroom, the `scipy.ndimage.measurements.label()` function was used to group the detections together and then determine a bounding box. The code below shows how this is implememted.

```python
def add_heat(self, heatmap, coordinates):
    '''
    For each detection add a 1 to that pixel

    Parameters
    ----------
    heatmap      : Heatmap array
    coordinates  : Detection coordinates

    Returns
    -------
    Heatmap of frame
    '''

    for loc in coordinates:
        heatmap[loc[1]:loc[3], loc[0]:loc[2]] += 1

    return heatmap

def merge_car_detections(self, detections, image_shape, threshold):
    '''
    Brings together and merges detections based on the heatmap and threshold.

    Parameters
    ----------
    detections : Array of detections
    shape      : Image shape
    threshold  : Heatmap threshold

    Return
    ------
    The merged regions and heatmap
    '''

    heatmap = np.zeros((image_shape[0], image_shape[1])).astype(np.float)
    heatmap = self.add_heat(heatmap, detections)

    # Apply a threshold to remove any false positives
    heatmap[heatmap < threshold] = 0
    # Clip the heatmap so that all values are either 0 or 255
    heatmap = np.clip(heatmap, 0, 255)
    # Label the features
    labels = label(heatmap)

    # Iterate through all the detected vehicles
    vehicles = np.empty([0, 4], dtype=np.int64)
    for vehicle in range(1, labels[1] + 1):
        nonzero = (labels[0] == vehicle).nonzero()
        vehicles = np.append(vehicles,[[np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])]], axis=0)

    return (vehicles, heatmap)
```
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my final video.](./P5_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This was implemented using the `add_heat()` and `merge_car_detections()` functions and described above. 

### Here are six test image frames and their corresponding heatmaps:

![Heatmap output image 1][image4]
![Heatmap output image 2][image5]
![Heatmap output image 3][image6]

![Heatmap output image 4][image7]
![Heatmap output image 5][image8]
![Heatmap output image 6][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It's clear that the approach I took is not completely accurate. This is evidenced in a few of the test images above and at various instances throughout the output video. The pipeline will likely fail when more vehicles are in the frame, especially when they overlap each other. In addition, the classifier was trained on a small subset of cars and car locations/positions. This could cause additional classification problems. It appears that the sliding window search I implemented still has too many false positives. In the future, a better method for reducing them might help. In addition, some sort of method for creating the bounding boxes around the vehicles could be optimized to better keep track as they move through the frame. 

