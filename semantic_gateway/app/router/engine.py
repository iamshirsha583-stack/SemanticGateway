import time
import re
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.router.models import RouteDecision, RouteDefinition


class SemanticEngine:
    """
    Hybrid Intent & Complexity Router:
    Layer 1: Fast Heuristic & Complexity Pre-Filter (<1ms execution before embedding)
    Layer 2: Adaptive Semantic Cosine Matching using SentenceTransformer & NumPy dot product.
    """

    def __init__(self, settings: Settings, model: Optional[SentenceTransformer] = None):
        self.settings = settings
        model_name = settings.EMBEDDING_MODEL_NAME
        # Support short name 'all-MiniLM-L6-v2' or full 'sentence-transformers/all-MiniLM-L6-v2'
        if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model = model or SentenceTransformer(model_name)
        self.routes: List[RouteDefinition] = []
        self.anchor_embeddings: Optional[np.ndarray] = None  # Shape (N, D) normalized
        self.anchor_metadata: List[Tuple[str, str, str]] = []  # (route_name, target_model, utterance)
        self.fast_indices: List[int] = []
        self.deep_indices: List[int] = []
        self._is_indexed = False

    def load_routes(self, routes_path: Optional[str] = None) -> None:
        path_str = routes_path or self.settings.ROUTES_FILE
        path = Path(path_str)

        if not path.exists():
            parent_path = Path(__file__).resolve().parent.parent.parent / path_str
            if parent_path.exists():
                path = parent_path
            else:
                raise FileNotFoundError(f"Routes file not found at: {path.absolute()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_routes = data.get("routes", [])
        self.routes = []

        all_utterances: List[str] = []
        self.anchor_metadata = []

        for r in raw_routes:
            r_name = r.get("name", "deep_lane")
            target_model = r.get("target_model") or (
                self.settings.FAST_MODEL_NAME if r_name == "fast_lane" else self.settings.FRONTIER_MODEL_NAME
            )
            utterances = r.get("utterances", [])

            self.routes.append(
                RouteDefinition(
                    name=r_name,
                    description=r.get("description", ""),
                    target_tier="fast" if r_name == "fast_lane" else "frontier",
                    utterances=utterances
                )
            )

            for utt in utterances:
                all_utterances.append(utt)
                self.anchor_metadata.append((r_name, target_model, utt))

        if not all_utterances:
            raise ValueError("No anchor utterances defined in routes configuration.")

        # Pre-calculate route index groups
        self.fast_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "fast_lane"]
        self.deep_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "deep_lane"]

        # Compute normalized sentence embeddings for all anchor utterances
        raw_vecs = self.model.encode(
            all_utterances,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.anchor_embeddings = raw_vecs.astype(np.float32)
        self._is_indexed = True

    def extract_prompt(self, messages: list) -> str:
        """Extract the last user message text from chat messages list."""
        if not messages:
            return ""
        for msg in reversed(messages):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            if role == "user":
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
                    return " ".join(texts)
        last_msg = messages[-1]
        content = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else "")
        return str(content)

    def _normalize_query(self, prompt: str) -> str:
        """Normalize query casing and strip excess punctuation before computing embeddings."""
        cleaned = prompt.strip().lower()
        cleaned = re.sub(r'[^\w\s\+\-\*\/\%\.]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or prompt.strip().lower()

    def _heuristic_prefilter(self, prompt: str) -> Optional[Tuple[str, str, float]]:
        """
        Layer 1: Fast Heuristic & Complexity Pre-Filter (Executes in <1ms before embedding).
        Returns (target_route, matched_reason, similarity_score) if matched, otherwise None.
        """
        stripped = prompt.strip()
        if not stripped:
            return ("fast_lane", "heuristic:empty_input", 1.0)

        words = stripped.split()
        word_count = len(words)

        # 1. High Complexity Guard:
        # Token length > 120 words
        if word_count > 120:
            return ("deep_lane", "heuristic:token_length_exceeded", 1.0)

        # Explicit multi-step reasoning signals and code generation keywords
        complexity_pattern = r'\b(implement|architect|byzantine|concurrency|smart\s+contract|dockerfile|debug\s+this\s+stacktrace)\b'
        if re.search(complexity_pattern, stripped, re.IGNORECASE):
            return ("deep_lane", "heuristic:high_complexity_keyword", 1.0)

        # 2. Fast Trivial Filter:
        # If input text (stripped) is <= 25 characters and matches conversational intents
        if len(stripped) <= 25:
            conversational_pattern = r'\b(hi|hii|hello|hey|bye|byebye|thanks|thank\s+you|ok|okay)\b'
            if re.search(conversational_pattern, stripped, re.IGNORECASE):
                return ("fast_lane", "heuristic:conversational_intent", 1.0)

        # 3. Pure Math / Arithmetic Filter:
        # If the stripped input contains only numbers, math operators, whitespace, or percentages
        if re.match(r'^[\d\s\+\-\*\/\(\)\.\=\%]+$', stripped) and re.search(r'[\d\+\-\*\/\=\%]', stripped):
            return ("fast_lane", "heuristic:pure_arithmetic", 1.0)

        # Starts with simple math terms like "calculate", "solve", "what is [number]"
        simple_math_pattern = r'^(calculate|solve|what\s+is\s+\d+)\b'
        if re.match(simple_math_pattern, stripped, re.IGNORECASE):
            return ("fast_lane", "heuristic:simple_math_intent", 1.0)

        return None

    def classify(self, prompt: str) -> RouteDecision:
        """
        Hybrid Intent & Complexity Routing:
        Layer 1: Heuristic & Complexity Pre-Filter (<1ms).
        Layer 2: Adaptive Semantic Cosine Matching using dense embeddings.
        """
        start_time = time.perf_counter()

        # ==========================================
        # LAYER 1: Fast Heuristic & Complexity Pre-Filter (<1ms)
        # ==========================================
        prefilter_result = self._heuristic_prefilter(prompt)
        if prefilter_result is not None:
            target_route, matched_reason, score = prefilter_result
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if target_route == "fast_lane":
                selected_model = self.settings.FAST_MODEL_NAME
                base_url = self.settings.FAST_MODEL_BASE_URL
                api_key = self.settings.FAST_MODEL_API_KEY
            else:
                selected_model = getattr(self.settings, "FRONTIER_MODEL_NAME", None) or getattr(self.settings, "DEEP_MODEL_NAME", "llama-3.3-70b-versatile")
                base_url = getattr(self.settings, "FRONTIER_MODEL_BASE_URL", getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1"))
                api_key = getattr(self.settings, "FRONTIER_MODEL_API_KEY", getattr(self.settings, "DEEP_MODEL_API_KEY", ""))

            return RouteDecision(
                target_route=target_route,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                similarity_score=round(score, 4),
                classification_time_ms=round(elapsed_ms, 3),
                decision_source="HEURISTIC_PREFILTER",
                matched_utterance=matched_reason
            )

        # ==========================================
        # LAYER 2: Adaptive Semantic Cosine Matching
        # ==========================================
        if not self._is_indexed or self.anchor_embeddings is None:
            raise RuntimeError("SemanticEngine is not initialized with routes.")

        # Normalize query casing and strip excess punctuation
        normalized_prompt = self._normalize_query(prompt)

        # Compute normalized embedding for the query prompt
        query_vec = self.model.encode(
            [normalized_prompt],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]

        # Vectorized Cosine Similarity via dot product
        similarities = np.dot(self.anchor_embeddings, query_vec)

        # Compute separate max similarity scores for fast_lane and deep_lane
        fast_sims = similarities[self.fast_indices] if self.fast_indices else np.array([])
        deep_sims = similarities[self.deep_indices] if self.deep_indices else np.array([])

        sim_fast = float(np.max(fast_sims)) if len(fast_sims) > 0 else -1.0
        fast_best_local_idx = int(np.argmax(fast_sims)) if len(fast_sims) > 0 else -1

        sim_deep = float(np.max(deep_sims)) if len(deep_sims) > 0 else -1.0
        deep_best_local_idx = int(np.argmax(deep_sims)) if len(deep_sims) > 0 else -1

        threshold = self.settings.SIMILARITY_THRESHOLD
        word_count = len(prompt.strip().split())

        # Adaptive routing decision logic
        if sim_deep > sim_fast and (sim_deep >= threshold or sim_deep > (sim_fast + 0.05)):
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx] if deep_best_local_idx != -1 else 0
            best_score = sim_deep
        elif word_count < 8 and sim_fast >= 0.35:
            # Short query adaptive sensitivity (< 8 words and sim_fast >= 0.35)
            target_route = "fast_lane"
            best_idx = self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0
            best_score = sim_fast
        elif sim_fast >= threshold and sim_fast >= sim_deep:
            target_route = "fast_lane"
            best_idx = self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0
            best_score = sim_fast
        else:
            # Preserve deep_lane as fallback for high-entropy and unmatched inputs
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx] if deep_best_local_idx != -1 else (self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0)
            best_score = sim_deep if sim_deep > -1.0 else 0.0

        matched_route, route_model, matched_utt = self.anchor_metadata[best_idx]

        if target_route == "fast_lane":
            selected_model = self.settings.FAST_MODEL_NAME or route_model
            base_url = self.settings.FAST_MODEL_BASE_URL
            api_key = self.settings.FAST_MODEL_API_KEY
        else:
            selected_model = getattr(self.settings, "FRONTIER_MODEL_NAME", None) or getattr(self.settings, "DEEP_MODEL_NAME", None) or route_model
            base_url = getattr(self.settings, "FRONTIER_MODEL_BASE_URL", getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1"))
            api_key = getattr(self.settings, "FRONTIER_MODEL_API_KEY", getattr(self.settings, "DEEP_MODEL_API_KEY", ""))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RouteDecision(
            target_route=target_route,
            selected_model=selected_model,
            base_url=base_url,
            api_key=api_key,
            similarity_score=round(best_score, 4),
            classification_time_ms=round(elapsed_ms, 3),
            decision_source="SEMANTIC_EMBEDDING",
            matched_utterance=matched_utt
        )

    # Alias method for route calls
    route = classify


# Alias class for backward compatibility
SemanticRouterEngine = SemanticEngine
