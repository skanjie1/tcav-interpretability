# Beyond Pixels: How TCAV Lets Humans Ask Neural Networks the Right Questions

> *A summary of Been Kim et al.'s paper: "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)" — ICML 2018*

---

You train a deep neural network to classify images. It works, impressively well. But then someone asks: *"Does your model actually care about stripes when it classifies a zebra, or is it using something else entirely?"*

With traditional interpretability tools, answering this question is surprisingly hard. Saliency maps will highlight pixels, but they won't tell you whether those pixels represent "stripes," "grass," or "the photographer's lighting." You're left squinting at heatmaps, trying to mentally bridge the gap between what the machine sees (pixel intensities) and what you care about (concepts).

This is the problem that **Testing with Concept Activation Vectors (TCAV)** sets out to solve. Introduced by Been Kim and colleagues at Google Brain, TCAV offers a fundamentally different approach to interpretability, one that speaks in human concepts rather than pixel coordinates.

---

## The Interpretability Gap: Machines and Humans Speak Different Languages

At its core, the interpretability challenge is a *translation* problem. The paper frames it elegantly using vector spaces: a machine learning model operates in a space $E_m$, spanned by basis vectors corresponding to input features like pixel values and neural activations. Humans, on the other hand, reason in a different space $E_h$, spanned by high-level concepts like "striped," "furry," or "female."

An interpretation, then, is a function $g: E_m \rightarrow E_h$, a mapping from the machine's language to the human's language. When this mapping is linear, the authors call it a **linear interpretability**.

This framing reveals why pixel-level saliency maps are limited. They operate entirely within $E_m$. They tell you *which* pixels matter, but they don't translate those pixels into human-understandable concepts. Furthermore, saliency maps are **local** explanations: they describe what happens for a single input image. If you want to know whether "stripes" matter for the *entire* zebra class, you'd have to manually inspect saliency maps for every zebra image and somehow aggregate your impressions. TCAV's ambition is to do this automatically, globally, and in terms that don't require any ML expertise to understand.

---

## The Key Idea: Concept Activation Vectors (CAVs)

The central insight of the paper is surprisingly elegant. Rather than trying to decipher individual neurons or pixel attributions, TCAV asks: *does the network's internal representation have a meaningful direction that corresponds to a human concept?*

Here's how it works. Suppose you want to know whether a model has learned something about the concept "striped." You gather a small set of example images that represent "striped" (photos of striped objects like fabric, wallpaper, etc.) and a set of random images as counterexamples. You then feed both sets through the network and look at the activations at some internal layer $l$.

At this layer, the "striped" examples will produce one cluster of activation vectors, and the random examples will produce another. A **linear classifier** is trained to separate these two clusters. The **Concept Activation Vector (CAV)** is defined as the vector orthogonal to the decision boundary of this classifier, that is, $v^l_C \in \mathbb{R}^m$, where $m$ is the dimensionality of layer $l$.

This CAV now represents a direction in the network's activation space that corresponds to the human concept "striped." The beauty of this approach is that it's grounded in a well-supported empirical observation: meaningful directions in neural network activation spaces can be efficiently learned via simple linear classifiers. This builds on prior work showing local linearity in neural representations (Alain & Bengio, 2016; Bau et al., 2017; Szegedy et al., 2013).

Critically, **the concepts don't need to exist in the training data.** You can define any concept you want (gender, race, texture, a specific object) simply by providing example images. This gives users enormous flexibility to explore hypotheses about what a model has learned.

---

## From Directions to Decisions: The TCAV Score

Having a CAV tells you that a concept *exists* as a direction in the network's representation. But the real question is: **does this concept actually influence the model's predictions?**

TCAV answers this using **directional derivatives**. Traditional saliency maps compute how sensitive the output is to changes in individual pixels:

$$\frac{\partial h_k(x)}{\partial x_{a,b}}$$

where $h_k$ is the logit for class $k$ and $x_{a,b}$ is a pixel value. TCAV instead computes how sensitive the output is to changes *in the direction of a concept*:

$$S_{C,k,l}(x) = \nabla h_{l,k}(f_l(x)) \cdot v^l_C$$

This **conceptual sensitivity** $S_{C,k,l}(x)$ tells you, for a single input $x$: if you nudged the network's activations in the direction of concept $C$, would the model's prediction for class $k$ increase or decrease?

To get a **global** measure across an entire class, TCAV aggregates over all inputs in that class:

$$TCAV_{Q_{C,k,l}} = \frac{|\{x \in X_k : S_{C,k,l}(x) > 0\}|}{|X_k|}$$

This TCAV score represents the fraction of class- $k$ inputs that were positively influenced by concept $C$. A score near 1.0 means the concept is strongly aligned with that class prediction; a score near 0.5 suggests the concept is irrelevant (no better than random).

The elegance here is that the TCAV score is **a single number** that communicates the global importance of a human concept to a model's predictions for an entire class. No manual inspection of individual examples required.

---

## Guarding Against Noise: Statistical Testing

One pitfall the authors address head-on: since any random set of images will produce *some* CAV, how do you know a particular CAV is meaningful and not just noise?

Their solution is a statistical significance test. Instead of training a single CAV, they perform **500 training runs**, each time using a different random set of counterexamples. A meaningful concept should produce TCAV scores that are *consistent* across all these runs. They apply a two-sided t-test against a null hypothesis of $$TCAV_{Q}$$ = 0.5 (the random baseline), with Bonferroni correction to control the false discovery rate.

This is a crucial safeguard. In their experiments, statistical testing successfully filtered out spurious results. For instance, it eliminated cases where the "dotted" concept returned a falsely high TCAV score for zebras at certain layers. Only after passing this test did "striped" consistently emerge as the dominant concept.

---

## Design Principles: Who Is This For?

TCAV was designed with four explicit goals that distinguish it from most interpretability methods:

1. **Accessibility**: The user doesn't need ML expertise. You define concepts using example images, and you get back a score between 0 and 1.

2. **Customization**: You can test *any* concept, not just those the model was trained on. Want to know if your model is sensitive to gender? Collect some gender-labeled images and test it.

3. **Plug-in readiness**: TCAV works on any already-trained model without retraining or modification. This is critical for practitioners who already have a deployed model.

4. **Global quantification**: Unlike saliency maps that explain one image at a time, TCAV gives you a single number characterizing the importance of a concept to an entire class.

---

## Experiments: What Did TCAV Reveal?

### Validating That CAVs Mean What We Think

Before trusting TCAV's outputs, the authors first demonstrate that CAVs genuinely correspond to their intended concepts through two approaches:

- **Sorting images by CAV similarity**: Using cosine similarity between image activations and a CAV, they sort images from "most similar" to "least similar" to a concept. For example, sorting stripe images by a "CEO" CAV placed pinstripe patterns at the top, intuitively the kind of patterns you'd find on a CEO's suit. Sorting neckties by a "model women" CAV surfaced images of women wearing neckties.

- **Empirical Deep Dream**: They apply activation maximization to CAVs, generating patterns that maximally activate each concept direction. The resulting patterns for "knitted," "corgi," and "Siberian husky" CAVs visually corresponded to their intended concepts, confirming that CAVs capture semantically meaningful directions in the network's representation.

### Insights and Biases in Widely Used Networks

Applying TCAV to GoogleNet and Inception V3 revealed both expected and concerning patterns:

- **Expected findings**: The "red" concept was highly important for fire engine classification; "striped" was dominant for zebras; the "Siberian husky" concept was important for dogsleds. These confirm common-sense intuition.

- **Bias detection**: More troublingly, TCAV found that the "female" concept was highly relevant to the "apron" class, and confirmed quantitatively what prior qualitative work had suggested: that ping-pong ball classification was strongly correlated with a particular racial group. These are biases the networks learned from their training data, despite not being explicitly trained on gender or race categories.

- **Where concepts are learned**: By examining CAV accuracy across layers, they found that simpler concepts like color achieve high classification accuracy in early layers, while more abstract concepts (objects, people) become separable only in higher layers, consistent with the well-known finding that neural networks build increasingly abstract representations through their depth.

An important practical finding: as few as **30 example images** per concept were sufficient to learn meaningful CAVs. For the "dumbbell" class, 30 images of each concept allowed TCAV to identify "arms" as the most important concept, quantitatively confirming a qualitative finding from DeepDream visualizations showing that the network associated dumbbells with arms holding them.

### The Controlled Experiment: TCAV vs. Saliency Maps

This is perhaps the most compelling section of the paper. The authors designed a clever controlled experiment to directly compare TCAV against saliency maps.

They created a dataset of three classes (zebra, cab, cucumber) where each image had a text caption overlaid. A noise parameter $p$ controlled whether the caption matched the class label. They then trained separate networks on datasets with different noise levels ($p$ = 0, 0.3, 1.0).

The key insight: by testing each network on *captionless* images, they could determine ground truth about whether each network relied more on the image content or the caption text.

**TCAV's performance**: The TCAV score closely tracked the ground truth. For the cab class, TCAV correctly showed that the image concept dominated regardless of noise level. For cucumbers, TCAV correctly captured the shifting balance between image and caption reliance as noise changed.

**Saliency maps' failure**: A 50-person human subject experiment on Amazon Mechanical Turk revealed that saliency maps correctly communicated which concept was more important only **52% of the time**, barely above the 50% random baseline. Worse, participants were equally confident in their correct and incorrect answers, meaning saliency maps didn't just fail to communicate; they actively misled users. When one saliency method got the right answer, the other consistently got it wrong.

This is a damning finding for the use of saliency maps as standalone interpretability tools. The information is technically *in* the saliency map, but humans can't reliably extract it.

---

## Medical Application: Diabetic Retinopathy

TCAV's real-world utility was demonstrated through a medical application: interpreting a model that predicts diabetic retinopathy (DR) severity from retinal fundus images. DR is graded on a 5-point scale, with different diagnostic concepts (microaneurysms, laser scars, hemorrhages) being relevant at different severity levels.

For severe DR (level 4), TCAV correctly identified diagnostic concepts like pan-retinal laser scars (PRP) and pre-retinal hemorrhages as highly important, with low scores for irrelevant concepts.

For mild DR (level 1), TCAV revealed something interesting: the concept of hemorrhages/microaneurysms (HMA), which is typically diagnostic of *higher* severity levels, had a surprisingly high TCAV score. This aligned with a known model error: the system frequently over-predicted level 1 cases as level 2. A consulting medical expert confirmed that this insight was actionable, saying she would want to tell the model to de-emphasize HMA for level 1 predictions.

This demonstrates TCAV's value beyond simple validation. It can help domain experts **diagnose model errors** by revealing *why* a model makes certain mistakes in terms they understand.

---

## Bonus: TCAV and Adversarial Examples

The appendix includes a fascinating experiment on adversarial examples. While adversarial perturbations successfully fooled the network into classifying noise images and modified non-zebra images as zebras with high confidence, the TCAV score distributions for these adversarial "zebras" were markedly different from real zebras.

Real zebras showed high TCAV scores for the "striped" concept; adversarial zebras did not. This suggests that TCAV could potentially serve as a detection signal for adversarial attacks, by maintaining a dictionary of expected TCAV score distributions for each class and flagging inputs whose concept profiles deviate significantly.

---

## Relative CAVs: Fine-Grained Comparisons

The paper also introduces **Relative CAVs** for distinguishing between semantically related concepts. Instead of training a classifier against random images, you train it to separate two related concepts (e.g., "dotted" vs. "striped"). The resulting vector captures the dimension along which these concepts differ, enabling finer-grained analysis.

This is particularly useful when related concepts have overlapping representations in the network. For example, to assess whether "striped" is more important than "dotted" for zebra classification, a relative CAV can quantify this comparison directly rather than testing each concept independently against random baselines.

---

## Limitations and Considerations

While TCAV represents a significant advance, the paper implicitly surfaces some limitations worth noting:

- **Linearity assumption**: CAVs assume that concepts correspond to linear directions in activation space. If a concept is encoded non-linearly, a CAV won't capture it.

- **Concept definition quality**: The method is only as good as the example images used to define concepts. Poorly chosen or biased example sets will produce misleading CAVs.

- **Image classification focus**: While the authors note potential for other domains (audio, video, sequences), all experiments are in image classification. Generalizability remains to be demonstrated.

- **Binary sensitivity**: The TCAV$_Q$ score only considers the *sign* of the directional derivative, not its magnitude. Two concepts with very different magnitudes of influence could receive similar TCAV scores.

---

## Why This Matters

TCAV represents a philosophical shift in how we approach interpretability. Instead of asking *"which pixels matter?"*, a question that speaks the machine's language, TCAV asks *"which concepts matter?"*, a question that speaks ours.

In a world where ML models are making consequential decisions in healthcare, criminal justice, and finance, the ability to audit these models in human terms is not just technically interesting; it's necessary. TCAV gives doctors a way to check whether a diagnostic model is using the right clinical features. It gives fairness researchers a way to quantify whether a classifier is relying on race or gender. It gives any user a way to test their hypotheses about what a model has learned.

The paper's controlled experiment is perhaps its most enduring contribution: the empirical demonstration that saliency maps, despite being the most widely used interpretability tool, **fail to communicate model behavior to humans** more often than not. This finding alone should give every practitioner pause, and motivate the adoption of concept-level interpretability methods like TCAV.

---

*Paper: Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). Proceedings of the 35th International Conference on Machine Learning (ICML 2018).* [arXiv:1711.11279](https://arxiv.org/abs/1711.11279)
