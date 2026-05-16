import contextlib
import logging
import os
import sys
import tempfile
import time
from concurrent import futures

import grpc

try:
    import torch
    import torch.nn as nn
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import load_peft_weights, set_peft_model_state_dict
except ImportError as exc:
    torch = None
    nn = None
    AutoProcessor = None
    BitsAndBytesConfig = None
    Qwen2_5_VLForConditionalGeneration = None
    process_vision_info = None
    load_peft_weights = None
    set_peft_model_state_dict = None
    REAL_IMPORT_ERROR = exc
else:
    REAL_IMPORT_ERROR = None

sys.path.insert(0, os.path.dirname(__file__))
import worker_pb2
import worker_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("worker")

USE_MOCK = os.environ.get("USE_MOCK", "1") == "1"
DEFAULT_MODEL_PATH = r"C:\Users\rog72\.cache\modelscope\hub\models\Qwen\Qwen2.5-VL-3B-Instruct"
DEFAULT_BEST_MODEL_PATH = r"C:\Users\rog72\kaggle\Review_model\checkpoints\best_model"
ID2LABEL = {0: "\u5b89\u5168", 1: "\u66b4\u529b", 2: "\u8272\u60c5", 3: "\u5e7f\u544a\u6b3a\u8bc8", 4: "\u8c29\u9a82\u5f15\u6218"}

def _normalize_path(path):
    if not path:
        return path
    path = os.path.expandvars(os.path.expanduser(path))
    if os.name != "nt" and len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path

def _autocast_context():
    if torch is not None and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()

if nn is not None:
    class Qwen2VLClassifier(nn.Module):
        def __init__(self, backbone, num_classes=5, hidden_size=2048, dropout=0.1):
            super().__init__()
            self.backbone = backbone
            self.hidden_size = hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **kwargs):
            with _autocast_context():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden = outputs.hidden_states[-1]
            batch_size = hidden.size(0)
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            pooled = hidden[torch.arange(batch_size, device=hidden.device), seq_lengths]
            return self.classifier(pooled.float())
else:
    class Qwen2VLClassifier:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"real model dependencies are unavailable: {REAL_IMPORT_ERROR}")

def _backbone_device(model):
    device = getattr(model.backbone, "device", None)
    if device is not None:
        return device
    return next(model.backbone.parameters()).device

def load_moderation_model():
    if REAL_IMPORT_ERROR is not None:
        raise RuntimeError(f"failed to import real model dependencies: {REAL_IMPORT_ERROR}")

    model_path = _normalize_path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    best_model_path = _normalize_path(os.environ.get("BEST_MODEL_PATH", DEFAULT_BEST_MODEL_PATH))
    classifier_path = os.path.join(best_model_path, "classifier_head.pt")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"MODEL_PATH not found: {model_path}")
    if not os.path.isdir(best_model_path):
        raise FileNotFoundError(f"BEST_MODEL_PATH not found: {best_model_path}")
    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(f"classifier head not found: {classifier_path}")

    logger.info("loading moderation model")
    logger.info("MODEL_PATH=%s", model_path)
    logger.info("BEST_MODEL_PATH=%s", best_model_path)

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=4 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=os.environ.get("DEVICE_MAP", "auto"),
    )

    hidden_size = int(os.environ.get("CLASSIFIER_HIDDEN_SIZE", "2048"))
    model = Qwen2VLClassifier(backbone=base_model, num_classes=len(ID2LABEL), hidden_size=hidden_size)

    set_peft_model_state_dict(model.backbone, load_peft_weights(best_model_path))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=_backbone_device(model)))
    model.eval()

    logger.info("moderation model is ready")
    return model, processor

def _write_temp_image(image_data):
    if not image_data:
        return None
    handle = tempfile.NamedTemporaryFile(prefix="moderation_", suffix=".jpg", delete=False)
    try:
        handle.write(image_data)
        return handle.name
    finally:
        handle.close()

def _to_device(inputs, model):
    device = _backbone_device(model)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

def predict_moderation(model, processor, image_path=None, image_data=None, text_content=None):
    owned_image_path = None
    if image_data:
        owned_image_path = _write_temp_image(image_data)
        image_path = owned_image_path

    try:
        content = []
        prompt = ""

        if image_path and os.path.exists(image_path):
            content.append({"type": "image", "image": f"file://{image_path}"})
            prompt = "\u8bf7\u5206\u6790\u8fd9\u5f20\u56fe\u7247\u7684\u5185\u5bb9\u7c7b\u578b\u3002"

        if text_content:
            prompt += f"\u8bf7\u5224\u65ad\u4ee5\u4e0b\u6587\u672c\u662f\u5426\u8fdd\u89c4\uff1a\n{text_content[:200]}"

        content.append({"type": "text", "text": prompt if prompt else "\u65e0\u5185\u5bb9"})
        messages = [{"role": "user", "content": content}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        if image_path and os.path.exists(image_path):
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt", padding=True)
        else:
            inputs = processor(text=[text_input], return_tensors="pt", padding=True)

        inputs = _to_device(inputs, model)

        with torch.no_grad():
            with _autocast_context():
                logits = model(**inputs)

        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = logits.argmax(dim=-1).item()
        return {
            "is_safe": pred_id == 0,
            "label": ID2LABEL[pred_id],
            "confidence": round(probs[pred_id].item(), 4),
            "all_scores": {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))},
        }
    except Exception as exc:
        logger.exception("moderation inference failed")
        return {"error": str(exc)}
    finally:
        if owned_image_path:
            try:
                os.remove(owned_image_path)
            except OSError:
                pass

def _result_to_proto(result):
    if result.get("error"):
        return worker_pb2.InferenceResult(error=result["error"])
    return worker_pb2.InferenceResult(
        is_safe=result["is_safe"],
        label=result["label"],
        confidence=float(result["confidence"]),
        all_scores={k: float(v) for k, v in result.get("all_scores", {}).items()},
    )

class Worker(worker_pb2_grpc.InferenceWorkerServicer):
    def __init__(self):
        if USE_MOCK:
            logger.info("mock mode enabled")
            self.model = None
            self.processor = None
        else:
            self.model, self.processor = load_moderation_model()

    def RunBatch(self, request, context):
        start = time.time()
        batch_size = len(request.items)
        logger.info("received batch | size=%d", batch_size)

        if USE_MOCK:
            results = self._mock_batch(request.items)
        else:
            results = self._real_batch(request.items)

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info("finished batch | size=%d | %dms", batch_size, elapsed_ms)
        return worker_pb2.BatchResponse(results=results, inference_ms=elapsed_ms)

    def _mock_batch(self, items):
        import random
        time.sleep(random.uniform(0.1, 0.3))

        safe = "\u5b89\u5168"
        violence = "\u66b4\u529b"
        porn = "\u8272\u60c5"
        ad_fraud = "\u5e7f\u544a\u6b3a\u8bc8"
        insult = "\u8c29\u9a82\u5f15\u6218"

        results = []
        for item in items:
            text = item.text_content or ""
            if any(w in text for w in ["\u50bb", "\u8822", "\u8111\u5b50", "\u6709\u75c5"]):
                results.append(worker_pb2.InferenceResult(
                    is_safe=False,
                    label=insult,
                    confidence=0.92,
                    all_scores={safe: 0.03, insult: 0.92, violence: 0.02, porn: 0.01, ad_fraud: 0.02},
                ))
            elif any(w in text for w in ["\u52a0\u5fae\u4fe1", "\u4f18\u60e0\u5238", "\u94fe\u63a5", "\u53d1\u8d22"]):
                results.append(worker_pb2.InferenceResult(
                    is_safe=False,
                    label=ad_fraud,
                    confidence=0.88,
                    all_scores={safe: 0.05, ad_fraud: 0.88, insult: 0.04, violence: 0.02, porn: 0.01},
                ))
            else:
                results.append(worker_pb2.InferenceResult(
                    is_safe=True,
                    label=safe,
                    confidence=0.95,
                    all_scores={safe: 0.95, insult: 0.01, violence: 0.01, porn: 0.01, ad_fraud: 0.02},
                ))
        return results

    def _real_batch(self, items):
        results = []
        for item in items:
            result = predict_moderation(
                self.model,
                self.processor,
                image_data=item.image_data if item.image_data else None,
                text_content=item.text_content or None,
            )
            results.append(_result_to_proto(result))
        return results

def serve():
    port = os.environ.get("WORKER_PORT", "50052")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ],
    )
    worker_pb2_grpc.add_InferenceWorkerServicer_to_server(Worker(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("python worker started | port=%s | mock=%s", port, USE_MOCK)
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
