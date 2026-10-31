# Predictions and locked experimental plan (recorded before training)

This is an AI-assisted prediction and study design, written before any CNN was
trained or any model prediction was inspected. It is a local audit record, not
an independently registered or externally timestamped preregistration.

## Research question
For the same small CNN, does changing the relationship between a digit's shape
label and a synthetic foreground texture change its decisions under cue conflict?

## Predictions, to be retained even if wrong
1. With textures correlated with digit labels, the CNN will follow texture more
   often than a CNN trained with texture independent of digit labels.
2. Randomizing textures during training will increase conflict shape agreement
   and conditional shape bias, while preserving useful accuracy on normal digits.
3. The plain-digit baseline will follow digit shape on most conflict images.
4. All three conditions will achieve at least 80% normal test accuracy. This is
   a declared interpretation safeguard, not a threshold imposed by the assignment.

## Data and controls
Use all 1,797 images from sklearn.datasets.load_digits (UCI Optical Recognition
of Handwritten Digits, 10 classes). Split original-image indices into 60% train,
20% validation, 20% test, stratified by digit with random_state=20261031. All
derivatives of an original image remain in its split. Inspect exact duplicate
source-image hashes for leakage. No claim of writer-disjoint splits is made.

Upscale 8x8 images to 32x32 with bilinear interpolation. Preserve the unmodified
grayscale version for normal evaluation. Define a foreground mask at intensity
>0.25. Texture is a sinusoidal grating: five orientations times two frequencies,
with randomized phase. Rank-map foreground values onto the same evenly spaced
intensity sequence (0.25..0.95) for each mask. Thus paired cue-conflict images
have exactly the same mask, background, and foreground intensity histogram.
Texture spatial arrangement changes; color, pixel count, brightness, and
histogram cannot identify the replacement texture within a paired shape.

## Intervention and training
Train the identical CNN from scratch in three conditions: plain (untextured
digits), correlated (texture class equals shape label), randomized (texture
independent of shape label). In every condition, the same 25% of examples per
epoch are normal grayscale digits. The remaining 75% are flat foreground
digits in plain or textured digits in the other conditions. Randomized texture
labels are sampled independently, not guaranteed perfectly balanced in any
single epoch. All conditions use identical source images, epoch count, optimizer,
batch order, normal-example flags, texture phases, and initialization for a paired
seed; only the treatment-dependent rendering changes. Seeds: 11,22,33.
Use the fixed final epoch (50), not the best conflict/test result. Validation is
monitoring only; no hyperparameter or epoch selection from test results. Preserve
all runs and report failure if the normal-accuracy safeguard is not satisfied.

## Evaluation
Evaluate identical held-out images for every model: normal grayscale; plain
mask; aligned texture; all 9 mismatching texture labels per digit (exhaustive
cue conflict); and texture-only fixed disk. Reuse one deterministic test phase
per source-image/texture pair. The disk is a supplementary, out-of-distribution
diagnostic, not the main evidence for texture use.

For conflicts let S=predictions matching shape, T=matching texture, O=other.
Report S/N, T/N, O/N and conditional shape bias S/(S+T), not just conditional
shape bias. Report coverage (S+T)/N. Chance digit accuracy is 10%; conditional
shape bias has chance expectation 50% when the two cues are exchangeable.
Report all seeds and mean +/- sample SD across seeds. Compare randomized minus
correlated using paired seeds. For a conditional 95% CI, average model outcomes
across seeds first, then cluster-bootstrap original test images with their 9
conflicts together (2,000 draws). This CI describes test-image uncertainty
conditional on this split and these trained models, not population training or
writer uncertainty. Do not treat transformed variants as independent images.

## Limitations to report
Synthetic gratings on small binarized handwritten digits are not natural object
textures. Digit shape includes local strokes; shape agreement does not prove use
of global outline rather than local shape. The experiment measures a learned
training-data dependence in this CNN and domain, not an inherent property of all
CNNs. The normal/texture mixture, foreground masking, and fixed-disk diagnostic
produce distribution shifts. One split and three training seeds do not establish
writer-level generalization. No accuracy drop alone proves a specific cue.
