# Exploratory positive control added after the primary results

The primary run showed exactly 10% texture agreement for the fixed-disk
out-of-distribution diagnostic. That result alone cannot distinguish learned
shape dependence from failure to recognize unfamiliar disk inputs, nor establish
that the synthetic texture bank is learnable. The following additional diagnostic
is therefore declared after observing the primary results and before training
the extra models. It does not replace, select, or alter any primary experiment.

Train the same 101,946-parameter CNN on fixed disks filled with the same ten
textures, with texture identity as the target. Use 1,077 training samples per
epoch, class labels cycling 0..9, random phase per epoch, the same three seeds,
Adam settings, batch size and fixed 50 epochs. Evaluate the same 3,600 fixed-disk
test images used in the primary evaluation. The phases are deterministic but
independent from the training generator. Predict that test accuracy will exceed
80%. Report this control as exploratory, even if it fails. It only establishes
learnability of the texture bank by the architecture under texture supervision;
it does not prove that a primary model used or ignored any specific cue.
