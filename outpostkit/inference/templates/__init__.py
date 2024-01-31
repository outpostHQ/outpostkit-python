# from typing import Dict

# from outpostkit._inference.templates.automatic_speech_recognition import (
#     AutomaticSpeechRecognitionInferenceHandler,
# )
# from outpostkit._inference.templates.common import AbstractInferenceHandler
# from outpostkit._inference.templates.conversational import (
#     ConversationalInferenceHandler,
# )
# from outpostkit._inference.templates.depth_estimation import (
#     DepthEstimationInferenceHandler,
# )
# from outpostkit._inference.templates.document_question_answering import (
#     DocumentQuestionAnsweringInferenceHandler,
# )
# from outpostkit._inference.templates.feature_extraction import (
#     FeatureExtractionInferenceHandler,
# )
# from outpostkit._inference.templates.fill_mask import FillMaskInferenceHandler
# from outpostkit._inference.templates.image_classification import (
#     ImageClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.image_segmentation import (
#     ImageSegmentationInferenceHandler,
# )
# from outpostkit._inference.templates.image_to_image import (
#     ImageToImageInferenceHandler,
# )
# from outpostkit._inference.templates.image_to_text import ImageToTextInferenceHandler
# from outpostkit._inference.templates.mask_generation import (
#     MaskGenerationInferenceHandler,
# )
# from outpostkit._inference.templates.object_detection import (
#     ObjectDetectionInferenceHandler,
# )
# from outpostkit._inference.templates.question_answering import (
#     QuestionAnsweringInferenceHandler,
# )
# from outpostkit._inference.templates.summarization import (
#     SummarizationInferenceHandler,
# )
# from outpostkit._inference.templates.table_question_answering import (
#     TableQuestionAnsweringInferenceHandler,
# )
# from outpostkit._inference.templates.text_classification import (
#     TextClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.text_generation import (
#     TextGenerationInferenceHandler,
# )
# from outpostkit._inference.templates.text_to_audio import TextToAudioInferenceHandler
# from outpostkit._inference.templates.token_classification import (
#     TokenClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.translation import TranslationInferenceHandler
# from outpostkit._inference.templates.video_classification import (
#     VideoClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.visual_question_answering import (
#     VisualQuestionAnsweringInferenceHandler,
# )
# from outpostkit._inference.templates.zero_shot_audio_classification import (
#     ZeroShotAudioClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.zero_shot_classification import (
#     ZeroShotClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.zero_shot_image_classification import (
#     ZeroShotImageClassificationInferenceHandler,
# )
# from outpostkit._inference.templates.zero_shot_object_detection import (
#     ZeroShotObjectDetectionInferenceHandler,
# )

# TaskTypeToHandlers: Dict[str, AbstractInferenceHandler] = {
#     "audio-classification": [pipeline, AudioClassificationInferenceHandler],
#     "automatic-speech-recognition": AutomaticSpeechRecognitionInferenceHandler,
#     "conversational": ConversationalInferenceHandler,
#     "depth-estimation": DepthEstimationInferenceHandler,
#     "document-question-answering": DocumentQuestionAnsweringInferenceHandler,
#     "feature-extraction": FeatureExtractionInferenceHandler,
#     "fill-mask": FillMaskInferenceHandler,
#     "image-classification": ImageClassificationInferenceHandler,
#     "image-segmentation": ImageSegmentationInferenceHandler,
#     "image-to-image": ImageToImageInferenceHandler,
#     "image-to-text": ImageToTextInferenceHandler,
#     "mask-generation": MaskGenerationInferenceHandler,
#     "object-detection": ObjectDetectionInferenceHandler,
#     "question-answering": QuestionAnsweringInferenceHandler,
#     "summarization": SummarizationInferenceHandler,
#     "table-question-answering": TableQuestionAnsweringInferenceHandler,
#     "text2text-generation": TextGenerationInferenceHandler,
#     "text-classification": TextClassificationInferenceHandler,
#     "sentiment-analysis": TextClassificationInferenceHandler,
#     "text-generation": TextGenerationInferenceHandler,
#     "text-to-audio": TextToAudioInferenceHandler,
#     "text-to-speech": TextToAudioInferenceHandler,
#     "token-classification": TokenClassificationInferenceHandler,
#     "ner": TokenClassificationInferenceHandler,
#     "translation": TranslationInferenceHandler,
#     "translation_xx_to_yy": TranslationInferenceHandler,
#     "video-classification": VideoClassificationInferenceHandler,
#     "visual-question-answering": VisualQuestionAnsweringInferenceHandler,
#     "zero-shot-classification": ZeroShotClassificationInferenceHandler,
#     "zero-shot-image-classification": ZeroShotImageClassificationInferenceHandler,
#     "zero-shot-audio-classification": ZeroShotAudioClassificationInferenceHandler,
#     "zero-shot-object-detection": ZeroShotObjectDetectionInferenceHandler,
# }
