Abstract—Image captioning is the task of generating text captions from images. It is a powerful bridging link between computer vision and natural language processing. Here, we propose a deep learning method using Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory (LSTM) networks for sequence generation. We train and test the model with the Flickr8k dataset. The model introduced here obtains noteworthy performance in BLEU scores, justifying its effectiveness in generating coherent and valid image captions.

Keywords—Image Captioning, CNN, LSTM, Deep Learning, Flickr8k, BLEU Score, Neural Networks, Feature Extraction, Sequence Modelling.

I. INTRODUCTION

Image captioning is a cutting-edge artificial intelligence problem that connects the fields of computer vision and natural language processing. Being able to produce descriptive text from an image has a broad set of applications, ranging from but not limited to helping visually impaired users access visual content, improving search engine indexing, and allowing for more intelligent personal assistant systems. Essentially, image captioning encompasses both the perception of visual features and the description of such features in natural language — something that simulates the way people describe scenes.

Previous methods in image captioning previously utilized handcrafted features along with strict grammatical templates. These models were not very flexible and did not generate creative or context-specific descriptions. Therefore, their application was greatly limited in highly specialized domains. But with the emergence of deep learning, especially convolutional neural networks (CNNs) and recurrent neural networks (RNNs), the area has seen a revolutionary shift. These models have made end-to-end training pipelines possible where visual features are extracted from image data directly and linguistic descriptions are generated without pre-defined templates.

CNNs, which are highly renowned for their quality when used in object detection and image classification tasks, are good visual encoders. They perform well in learning high-level abstract features from images and pattern detection with the capacity to learn spatial hierarchies. LSTM networks, a variant of RNNs, have also performed well in sequence modelling tasks, especially natural language generation. Merging these two models allows a system to learn visual semantics to linguistic sequence mapping.

In this paper, we experiment with such a combination system wherein we use CNNs for the extraction of features from images and LSTMs for the generation of captions. We take advantage of the annotated and varied image-caption pairs contained in the Flickr8k dataset as our training set. We outline the structure, preprocessing pipe, training plan, and measurement methods utilized by our system. We also showcase the constraints and potential upgrades for this baseline setup. This paper aims to be an all-embracing depiction of CNN-LSTM-based image captioning systems and show the potential of the same in actual applications.

 upon which more complex architectures can be developed.

III. METHODOLOGY

This section explains the technicalities of our image captioning system. We introduce the dataset used, data preprocessing workflow, model design, training process, and inference process. We aim to establish an efficient and interpretable baseline model with the CNN-LSTM architecture to be used as a starting point for further investigations.

A. Dataset

We utilize the Flickr8k dataset, a popular benchmark in image captioning research. The dataset consists of 8,092 images, each of which has five human-generated captions, for a total of more than 40,000 captions. The dataset is smaller than other datasets such as MS COCO

but large enough to offer diversity for training and testing. The dataset is divided into:

Training set: 6,000 images

Validation set: 1,000 images

Test set: 1,000 images

Each caption usually summarizes the scene in a brief sentence (8–12 words) and identifies important objects and activities.

B. Preprocessing

Data preprocessing plays an important role in enhancing the performance of the model and guaranteeing training stability. We carry out the following operations on both text and image data:

· Image Preprocessing

Resize all the images to 299×299 pixels to match the input size needed by the InceptionV3 model.

Normalize pixels to the range [-1, 1].

Leverage a pre-trained InceptionV3 CNN, trained over ImageNet, to obtain the 2,048-dimensional feature vectors from the final pooling layer.

· Text Preprocessing:

Put all captions in lowercase and remove punctuation.

Insert special tokens <start> and <end> into every caption.

Tokenize the captions using Keras Tokenizer class.

Restrict the vocabulary to the 5,000 most common words.

Pad sequences to the longest caption length with zero-padding.

· Sequence Preparation:

As input, we use input-output pairs for training in the following manner: input is image feature + partial caption, and output is the next word in the sequence.

For example, for the caption "a dog is running", the following training pairs would be:

Input: image + "startseq"

Output: "a"

Input: image + "startseq a"

Output: "dog”.

C. Model Architecture

Our architecture follows the encoder-decoder paradigm:

· Encoder (CNN):

InceptionV3 is used without the end classification layer.

Feature-extracted from images are projected to a 256-unit dense layer with ReLU activation.

· Decoder (LSTM):

A layer of embeddings encodes word indexes into dense 256-dimensional vectors.

The LSTM contains 256 hidden units and takes both the word embedded and image context vector.

0.5 dropout is used to prevent overfitting.

A Dense layer with softmax activation is used at the output to predict the next word.

The model has two inputs:

· Image feature vector (from CNN)

Caption order (word tokens)

These are concatenated and input into the LSTM for word-by-word prediction.

D. Training Details

The model is constructed using the categorical cross-entropy loss and is trained using the Adam optimizer and initial learning rate of 0.001. The following regularization and optimization methods are used:

Early stopping on validation loss with patience of 3 epochs.

Model checkpointing for saving the best model.

Learning rate reduction at plateau.

Batch size of 64.

Training is achieved for 12 epochs with a GPU acceleration using Jupyter Notebook.

We employ teacher forcing at training time, i.e., the ground truth word is input to the decoder at every time step, as opposed to the model's prediction.

E. Inference Strategy

In inference (caption generation), we obtain image features with InceptionV3, and start the sequence with the <start> token. The next word is predicted by the model based on the image and words up to now. We use:

Greedy decoding: Select the highest probability word at each time step.

Beam search (optional): Maintains a few candidate sequences in equilibrium between exploration and exploitation.

IV. EXPERIMENTS AND RESULTS

To assess the performance of our CNN-LSTM-based image captioning model, we conducted a series of experiments on the Flickr8k dataset. We provide both quantitative measures (BLEU scores) and qualitative analysis (example captions and error patterns). These results shed light on the model's strengths and weaknesses, as well as the impact of various design choices.

A. Evaluation Metrics

We measure the quality of generated captions using the BLEU (Bilingual Evaluation Understudy) score, a widely used metric for machine-generated text. BLEU is a metric for the degree to which the n-grams of the candidate (generated) captions overlap with the n-grams of one or more reference (human) captions. BLEU scores vary from 0 (no overlap) to 1 (perfect overlap).

We report:

BLEU-1: Unigram (word-level) accuracy

BLEU-2: Bigram precision

BLEU-3: Trigram score

BLEU-4: Most strict 4-gram precision

Whereas BLEU-1 evaluates literal word accuracy, BLEU-4 challenges fluency, as well as context recall.

B. Quantitative Findings

Following is the demonstration of the model's performance on the test set:

Metric Score

BLEU-1 0.67

BLEU-2 0.48

BLEU-3 0.35

BLEU-4 0.23

These scores show that the model works well in picking the right words (BLEU-1) and is contextually fluent at trigrams level. BLEU-4 is lower compared, and that's to be expected for models that have been trained on low-capacity data like Flickr8k.

C. Qualitative Results

We have also verified the captions generated for some of the test images. They are given below:

Image 1: A child and a dog playing in a meadow.

Caption made: "a boy playing with a dog in a field"

Reference Captions:

A dog and a child are running on the lawn.

A boy plays outdoors with a dog.

Image 2: A man on a wave.

Caption: "a man surfing on a wave on a surfboard"

Reference Captions:

A surfer catches a wave at sea.

"Man on a large ocean wave."

In the majority of cases, the generated captions are almost as good as human captions in terms of object and activity identification. While grammar may not always be perfect, semantics are generally accurate and consistent.

D. Error Analysis

In spite of its overall performance, the model has certain typical failure cases:

Hallucination: The model produces objects or activities that do not exist in the image.

Example: "a man on a bike" to identify a photo of an upright person.

Under-Specification: Captions are too vague or generic.

Example: "a woman in a kitchen" for a photo of a woman cooking.

Repetition or redundancy: Frequent repetition of words, especially in inference without beam search.

Example: "a dog running in the park."

These errors are a sign of overfitting, dataset being too small, or lack of contextual feedback. The use of attention or semantics can potentially minimize these problems.

E. Effect of Training Parameters

We experimented with various combinations:

Embedding dimension: Above 256, it did not result in performance improvement.

LSTM units: 256 provided the best compromise between accuracy and stable training.

Vocabulary size: 5,000-word limit was evaded to avoid overfitting and facilitate effective learning.

Decoding approach: Beam search (beam width 3) gave marginally better captions but at increased computational expense.

V. CONCLUSION

This In this paper, we introduced a deep learning model for automatic image captioning using the integration of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. We experimented and trained our model using the Flickr8k dataset, a typical corpus that provides a small but varied collection of annotated images. Using a CNN (InceptionV3) for visual feature representation and an LSTM for generating the features in the form of natural language, we created an end-to-end system that can produce useful captions.

Our results indicate that our CNN-LSTM model can indeed generate grammatically correct and semantically valid captions for images. The model scores highly on the BLEU evaluation metric, particularly on the unigram and bigram levels (BLEU-1 and BLEU-2), which establishes its ability to identify the most critical image features and generate intelligible descriptions. Qualitative testing also confirms the efficacy of our model, exhibiting its capacity to caption generic scenarios involving people, animals, and generic objects.

But our work also exposed some shortcomings. The model, at times, generates generic, too generic, or too simple captions on complex or ambiguous scenes. It also suffers from hallucinations, redundancy, and under-specification, issues well-documented in sequence generation tasks. These shortcomings can be explained by the comparatively small size

of the Flickr8k dataset, the absence of visual attention mechanisms, and the greedy decoding.

To transcend these constraints and improve future systems, the following research areas can be explored:

Attention Mechanisms: Adding spatial or temporal attention can allow the decoder to selectively focus on the most relevant areas of the image when producing each word. Not only does this improve accuracy, but it also boosts interpretability.

Transformer Architectures: Recent developments in multimodal transformers, including Vision Transformers (ViT), BERT-based encoders, and vision-language models (e.g., CLIP, BLIP), provide novel paradigms for visual-linguistic representation learning.

More data: Having more and diverse data like Flickr30k, MS COCO, or Conceptual Captions can lead to better generalization and more linguistic output.

Reinforcement Learning: Optimizing model behaviour directly for evaluation metrics (e.g., METEOR, CIDEr) using reinforcement learning methods such as Self-Critical Sequence Training (SCST) can also optimize behaviour.

Multimodal Pretraining: Large-scale pretraining of encoders and decoders on image-text pairs and subsequent followed fine-tuning for captioning can greatly improve performance and diminish the requirement for task-specific data.

Real-Time and Application-Specific Models: Utilizing light captioning models for mobile or application-specific applications (e.g., labelling medical images or robotics) is another interesting direction.

Briefly put, although the CNN-LSTM model is a sound and affordable baseline for image captioning, the community is quickly developing new methods and technology. The solution implemented in this work offers a working implementation and a solid foundation for future vision-language model innovation.
