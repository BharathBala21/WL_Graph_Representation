# Leaf Segmentation Algorithm: Choice and Rationale

## 1. Algorithm Used

The leaf-segmentation method used in notebook 08 is a color-index thresholding pipeline:

1. Convert the input image from BGR to RGB.
2. Compute the Excess Green (ExG) index for every pixel:

   $$\mathrm{ExG} = 2\frac{G}{R+G+B} - \frac{R}{R+G+B} - \frac{B}{R+G+B}$$

3. Normalize ExG to an 8-bit range from 0 to 255.
4. Apply Otsu's global thresholding to produce a binary foreground mask.
5. Apply morphological opening and closing with a 5 x 5 elliptical kernel, each for two iterations.
6. Detect external contours and keep the contour with the largest area as the leaf.
7. Calculate the mask-area ratio and assign a quality flag:
   - `FAILED` when the mask is empty or smaller than 1% of the image.
   - `SUSPICIOUS` when the mask covers more than 98% of the image.
   - `ACCEPT` otherwise.

The accepted mask is then used for shape, color, texture, and spatial feature extraction. The implementation is in `segment_leaf` in [08_OLID_Full_Feature_Extraction.ipynb](08_OLID_Full_Feature_Extraction.ipynb).

## 2. Why ExG + Otsu Was Selected

The choice is appropriate for the project's data and objective for the following reasons.

### It uses the strongest visual distinction available

The subject is a plant leaf, and the green response is usually more informative than raw brightness. ExG increases the relative contribution of green while reducing the influence of red and blue. This makes the method more focused on leaf-like pixels than a grayscale threshold would be.

### It is unsupervised

The pipeline processes approximately 4,749 images across multiple crop and condition classes. ExG + Otsu does not require manually labelled pixel masks, a trained segmentation model, or crop-specific threshold values. Otsu estimates a threshold from each image's ExG histogram, allowing the same procedure to be applied across the dataset.

### It is computationally inexpensive

The method uses standard OpenCV array operations, thresholding, morphology, and contour extraction. This is substantially lighter than training and running a semantic-segmentation network, which was useful for a large feature-extraction pipeline.

### It preserves the intended downstream features

The project needs a foreground mask to calculate leaf geometry and masked color, texture, and regional features. A single connected leaf mask is sufficient for these measurements. Selecting the largest external contour also removes many small isolated noise regions before feature extraction.

### It is deterministic and easy to reproduce

The method has explicit operations and fixed morphology settings. It does not depend on random initialization, interactive annotations, or a separately trained model. This supports repeatable feature generation and leakage-safe downstream experiments.

## 3. Why Only This Algorithm Was Used in the Main Pipeline

The notebook was designed as a scalable full-dataset feature-extraction pipeline, not as a segmentation-algorithm comparison study. The project therefore selected one simple, reproducible baseline that matched the expected image structure and used it consistently for all records.

This does **not** mean that ExG + Otsu is universally the best segmentation algorithm. It means that it offered a reasonable balance of:

- low implementation and computational cost;
- no requirement for pixel-level labels;
- compatibility with green leaf imagery;
- consistent processing across all classes; and
- direct production of the mask required by the feature extractor.

Using several unrelated segmentation methods in the main pipeline would also create a second experimental variable. Different masks would change shape, color, texture, and spatial features, making it harder to determine whether later classifier or graph results came from the representation or from the segmentation method. A single fixed method keeps the main experiment controlled.

## 4. Why Other Algorithms Were Not Used

The following methods were not included in the main pipeline. These are methodological reasons based on the project's requirements; notebook 08 does not report a formal head-to-head benchmark against all of them.

### Fixed RGB, HSV, or grayscale thresholding

These methods are simple, but a fixed threshold is sensitive to illumination, camera exposure, shadows, and disease-related color changes. A grayscale threshold also ignores the fact that green chromaticity can distinguish a leaf from a non-green background even when brightness varies. HSV thresholding could be useful, but it would require selecting hue and saturation ranges and validating them across crop and condition classes.

### Adaptive thresholding

Adaptive thresholding can handle local illumination changes, but it may fragment a leaf when texture, veins, shadows, or lesions create local intensity differences. It also introduces neighbourhood-size and constant parameters that would need tuning and validation for this dataset.

### K-means or Gaussian-mixture colour clustering

Clustering can separate colour groups without labels, but the clusters do not inherently identify which group is the leaf. Cluster ordering and the number of clusters must be chosen, and diseased or yellowing leaves may overlap with background colours. The extra selection logic would make the pipeline less direct than ExG followed by thresholding.

### GrabCut or graph-cut segmentation

GrabCut can produce detailed masks, but it normally needs an initialization such as a bounding box, foreground/background seeds, or a reliable automatic initialization. Applying and validating that initialization across thousands of images adds complexity and can fail when the background is visually similar to the leaf.

### Watershed segmentation

Watershed is mainly useful when touching objects must be separated. The pipeline generally treats each image as containing one main leaf, so watershed would add a marker-generation problem without solving the primary foreground-background problem. It can also over-segment textured leaves.

### Active contours or level sets

These methods can refine boundaries, but they require good initialization and involve more parameters and computational cost. Their advantage is less important when the downstream features are global and contour-based rather than requiring highly precise boundary detail.

### Deep-learning segmentation models

Models such as U-Net or Mask R-CNN could learn more complex boundaries, but they require pixel-level training masks or a suitable pretrained model, additional training and validation, and more compute. They would also introduce a learned component that was outside the scope of notebook 08's handcrafted-feature pipeline.

### Manual segmentation

Manual masks could provide a strong reference standard, but they are slow, subjective, and difficult to scale to the complete dataset. They are better suited to creating a validation subset than to producing every production mask.

## 5. Quality Control and Observed Limitation

The pipeline did not blindly accept every mask. It recorded `mask_area_ratio` and rejected empty or tiny masks, while flagging masks covering almost the whole image as suspicious. In the executed notebook 08 results:

- total processed: 4,749;
- accepted: 4,387;
- failed or excluded: 362; and
- 38 numerical features were extracted per accepted image.

The later validation and bias-audit notebooks show why this quality control matters. Acceptance was not identical across crops. Notebook 09 reports the lowest acceptance for tomato at approximately 0.73 and the highest for ash gourd at 1.00, and notes that crop morphology may affect how successfully Otsu separates leaf from background. Notebook 10 was created specifically to investigate whether Otsu-based exclusion introduced systematic bias, especially for tomato and the LM condition.

Therefore, the accurate conclusion is:

> ExG + Otsu was chosen as a practical, scalable, unsupervised, and reproducible segmentation method for the main pipeline. It was not established as universally superior to all other algorithms, and its crop-dependent failure behaviour is a documented limitation.

## 6. Recommended Interpretation for the Report

The method should be described as a **fixed classical segmentation baseline**, not as an optimal segmentation solution. The main experimental contribution is the downstream feature and graph representation pipeline. A stronger future comparison would evaluate ExG + Otsu against a small labelled validation subset using metrics such as IoU, Dice score, boundary quality, acceptance rate, and downstream classification performance.

## Source Notebooks

- [08_OLID_Full_Feature_Extraction.ipynb](08_OLID_Full_Feature_Extraction.ipynb): implementation and full-dataset extraction.
- [09_OLID_Full_Feature_Validation.ipynb](09_OLID_Full_Feature_Validation.ipynb): acceptance and crop-level validation.
- [10_OLID_Segmentation_Bias_Audit.ipynb](10_OLID_Segmentation_Bias_Audit.ipynb): diagnostic audit of Otsu-related exclusions.