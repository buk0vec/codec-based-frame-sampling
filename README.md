# Codec-based Frame Sampling

Have you ever wondered if the file format of your video can automatically determine the best frames to pass to a multimodal large language model? Have you ever wondered if this approach can give the same accuracy as uniform sampling with a lower token cost?

Well great, because **that what I've been wondering about too.** This project aims to leverage H.264 encoding to select I-frames from the video stream and use them as thumbnails for video understanding tasks. By intelligently constructing our encoding parameters, we should be able to see comperable performance to uniform sampling with many less image input tokens.