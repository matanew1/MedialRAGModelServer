import json
import os
import re
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from fuzzywuzzy import fuzz
import joblib
import logging
from threading import Lock
import warnings

# Suppress FutureWarning for encoder_attention_mask
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

try:
    from .model import call_llm
except ImportError:
    from model import call_llm

from dotenv import load_dotenv, find_dotenv

# Configure logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "diseases.json"))
INDEX_PATH = os.getenv("INDEX_PATH", str(PROJECT_ROOT / "index" / "faiss.index"))
TFIDF_INDEX_PATH = os.getenv("TFIDF_INDEX_PATH", str(PROJECT_ROOT / "index" / "tfidf.joblib"))
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "intfloat/multilingual-e5-large")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "10"))
LAZY_INIT = os.getenv("RAG_LAZY_INIT", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_state_lock = Lock()
_initialized = False
_embedder = None
_reranker = None
_medical_data = None
_index = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_embedding_dim = None
_index_built = False

# Expanded Hebrew medical synonyms
HEBREW_MEDICAL_SYNONYMS = {
    "חום": ["חום גבוה", "קדחת", "פייבר"],
    "כאב": ["כאבים", "אי נוחות", "צער", "סבל"],
    "בחילה": ["בחילות", "הקאה", "גועל נפש"],
    "שלשול": ["שלשולים", "דיזנטריה", "יציאות רכות"],
    "עצירות": ["קושי ביציאות", "קונסטיפציה"],
    "כאב ראש": ["מיגרנה", "צפלגיה", "הדאכה"],
    "כאב בטן": ["כאבי בטן", "קוליק", "גסטרלגיה"],
    "עייפות": ["תשישות", "חולשה", "לאות"],
    "סחרחורת": ["סחרחורות", "ורטיגו", "דיזינס"],
    "קוצר נשימה": ["קשיי נשימה", "דיספניאה"],
    "שיעול": ["שיעול יבש", "שיעול עם ליחה", "טוסיס"],
    "נזלת": ["גודש באף", "רינוריאה"],
    "כאב גרון": ["דלקת גרון", "פארינגיטיס"],
    "גרד": ["גירוד", "פרוריטוס"],
    "דמעת": ["דמעות", "לקרימציה"],
    "אלרגיה": ["תגובה אלרגית", "היפרסנסיטיביות"],
    "אסתמה": ["קצרת", "ברונכיאלית"],
    "לחץ דם": ["לחץ דם גבוה", "היפרטנזיה", "לחץ דם נמוך", "היפוטנזיה"],
    "סוכר": ["סוכרת", "דיאבטס"],
    "דלקת": ["אינפלמציה", "זיהום"]
}

# Category keywords for filtering relevance
CATEGORY_KEYWORDS = {
    "מחלות עיניים": ["עיניים", "עין", "דמעת", "אדמומיות", "גירוד", "הפרשה", "ראייה", "רגישות לאור"],
    "מחלות זיהומיות": ["זיהום", "חום", "שיעול", "נזלת", "דלקת"],
    "מחלות נוירולוגיות": ["כאב ראש", "סחרחורת", "בלבול", "התקפים"],
    "מחלות עור": ["פריחה", "גירוד", "יובש", "אדמומיות בעור"]
}

def normalize_hebrew_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r'[^\u0590-\u05FF\u0020-\u007F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_medical_terms(text):
    normalized_text = normalize_hebrew_text(text)
    expanded_terms = set()
    for term, synonyms in HEBREW_MEDICAL_SYNONYMS.items():
        if term in normalized_text or any(syn in normalized_text for syn in synonyms):
            expanded_terms.update(synonyms)
    return text + " " + " ".join(expanded_terms) if expanded_terms else text

def create_enhanced_text(item):
    parts = []
    if 'disease' in item:
        parts.append(item['disease'] * 2)  # Double weight for disease name
    if 'symptoms_detailed' in item:
        symptoms = item['symptoms_detailed']
        if 'primary' in symptoms:
            parts.extend([s * 2 for s in symptoms['primary']])  # Double weight for primary symptoms
        if 'secondary' in symptoms:
            parts.extend(symptoms['secondary'])
        if 'severity_scale' in symptoms:
            parts.append(f"חומרה {symptoms['severity_scale']}")
    if 'symptoms' in item:
        parts.append(item['symptoms'])
    if 'category' in item:
        parts.append(item['category'] * 2)  # Double weight for category
    if 'subcategory' in item:
        parts.append(item['subcategory'])
    if 'complications' in item:
        parts.extend(item['complications'])
    if 'differential_diagnosis' in item:
        parts.extend(item['differential_diagnosis'])
    if 'common' in item:
        common_map = {1: "מאוד נדיר", 2: "נדיר", 3: "בינוני", 4: "שכיח", 5: "מאוד שכיח"}
        parts.append(f"שכיחות {common_map.get(item['common'], 'לא ידוע')}")
    if 'red_flags' in item:
        parts.extend([f"דגל אדום {rf}" for rf in item['red_flags']])
    if 'treatment' in item:
        treatment = item['treatment']
        if 'primary' in treatment:
            parts.extend(treatment['primary'])
        if 'medications' in treatment:
            parts.extend(treatment['medications'])
        if 'contraindications' in treatment:
            parts.extend([f"התווית נגד {c}" for c in treatment['contraindications']])
    if 'prognosis' in item:
        parts.append(f"פרוגנוזה {item['prognosis']}")
    if 'prevention' in item:
        parts.extend([f"מניעה {p}" for p in item['prevention']])
    if 'lab_tests' in item:
        parts.extend([f"בדיקה {lt}" for lt in item['lab_tests']])
    if 'risk_factors' in item:
        parts.extend([f"גורם סיכון {rf}" for rf in item['risk_factors']])
    if 'duration' in item:
        parts.append(f"משך {item['duration']}")
    if 'contagious' in item:
        parts.append(f"מדבק {'כן' if item['contagious'] else 'לא'}")
    if 'age_groups' in item:
        parts.extend([f"קבוצת גיל {ag}" for ag in item['age_groups']])
    enhanced_text = " ".join(parts)
    return expand_medical_terms(enhanced_text)

def _initialize_if_needed():
    global _initialized, _embedder, _reranker, _medical_data, _index, _tfidf_vectorizer, _tfidf_matrix, _embedding_dim, _index_built
    if _initialized and _embedder is not None:
        return
    with _state_lock:
        if _initialized and _embedder is not None:
            return
        
        logger.info("Initializing RAG system...")
        try:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Medical data file not found at {DATA_PATH}")
            with open(DATA_PATH, encoding="utf-8") as f:
                _medical_data = json.load(f)["data"]
            logger.info(f"Loaded {len(_medical_data)} medical records")
            
            _embedder = SentenceTransformer(EMBEDDER_MODEL)
            _reranker = CrossEncoder(RERANK_MODEL)
            
            enhanced_texts = [create_enhanced_text(item) for item in _medical_data]
            
            # Expected dimension for intfloat/multilingual-e5-large is 1024
            expected_dim = 1024
            if os.path.exists(INDEX_PATH):
                try:
                    _index = faiss.read_index(INDEX_PATH)
                    _embedding_dim = _index.d
                    if _embedding_dim != expected_dim:
                        logger.warning(f"FAISS index dimension {_embedding_dim} does not match expected {expected_dim}. Rebuilding index...")
                        _index = None
                    else:
                        logger.info(f"Loaded FAISS index with dimension {_embedding_dim}")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}. Rebuilding index...")
                    _index = None
            
            if _index is None:
                logger.info("Building new FAISS index...")
                embeddings = _embedder.encode(enhanced_texts, convert_to_numpy=True, show_progress_bar=True)
                _embedding_dim = embeddings.shape[1]
                if _embedding_dim != expected_dim:
                    raise ValueError(f"Embedding dimension {_embedding_dim} does not match expected {expected_dim}")
                Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
                
                nlist = max(50, len(_medical_data) // 10)
                quantizer = faiss.IndexFlatIP(_embedding_dim)
                _index = faiss.IndexIVFFlat(quantizer, _embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                _index.train(embeddings)
                _index.add(embeddings)
                faiss.write_index(_index, INDEX_PATH)
                _index_built = True
                logger.info(f"Built FAISS IVFFlat index with {len(embeddings)} vectors")
                del embeddings
            
            if os.path.exists(TFIDF_INDEX_PATH):
                try:
                    tfidf_data = joblib.load(TFIDF_INDEX_PATH)
                    _tfidf_vectorizer = tfidf_data['vectorizer']
                    _tfidf_matrix = tfidf_data['matrix']
                    logger.info("Loaded TF-IDF index")
                except Exception as e:
                    logger.warning(f"Failed to load TF-IDF index: {e}. Rebuilding index...")
                    _tfidf_vectorizer = None
            
            if _tfidf_vectorizer is None:
                logger.info("Building TF-IDF index...")
                _tfidf_vectorizer = TfidfVectorizer(
                    analyzer='word',
                    token_pattern=r'[\u0590-\u05FF]+',
                    max_features=10000,
                    ngram_range=(1, 3)
                )
                _tfidf_matrix = _tfidf_vectorizer.fit_transform(enhanced_texts)
                joblib.dump({'vectorizer': _tfidf_vectorizer, 'matrix': _tfidf_matrix}, TFIDF_INDEX_PATH)
                logger.info("Built TF-IDF index")
            
            _initialized = True
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

def _ensure_initialized():
    if not _initialized:
        _initialize_if_needed()

def _build_prompt(query: str, context_items):
    context_str = "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in context_items)
    return f"""שאלה: {query}

מידע רפואי רלוונטי:
{context_str}

תשובה: אני ממש מבין כמה לא נעים זה יכול להיות לחוות {query}. בוא נבין יחד מה הכי סביר שקורה, בצורה פשוטה וברורה, ואיך אפשר לעזור לך:

**מה נראה הכי סביר**:
- **{context_items[0]['disease']}**: התסמינים שלך ({query}) ממש מתאימים ל{context_items[0]['disease']}, כי הם כוללים {context_items[0]['symptoms']}. זו מחלה {context_items[0].get('common', 3) >= 4 and 'די שכיחה' or 'שלא רואים כל יום'}. {context_items[0].get('prognosis', 'בדרך כלל זה משתפר עם טיפול מתאים.')}

**איך אני יודע שזה מתאים**:
התסמינים שתיארת תואמים את התסמינים העיקריים של {context_items[0]['disease']}, והיא {context_items[0].get('category') == 'מחלות עיניים' and 'קשורה לעיניים' or 'קשורה למערכת מסוימת בגוף'}. {f'היא גם נפוצה בקרב {", ".join(context_items[0].get("age_groups", []))}' if context_items[0].get("age_groups") else 'היא יכולה להופיע בכל גיל.'}

**אפשרויות אחרות שכדאי לשקול**:
{''.join(f'- {item["disease"]}: {item["symptoms"]} ({item.get("prognosis", "פרוגנוזה לא ידועה")})\n' for item in context_items[1:]) if len(context_items) > 1 else '- כרגע אין אפשרויות אחרות שמתאימות ממש.'}

**מה אפשר לעשות עכשיו**:
- **דבר עם רופא**: כדאי לפנות לרופא {context_items[0].get('category') == 'מחלות עיניים' and 'עיניים' or 'כללי'} כדי שיבדוק אותך מקרוב{f', במיוחד אם אתה מרגיש {", ".join(context_items[0].get("red_flags", []))}' if context_items[0].get("red_flags") else ''}. {f'בדיקות כמו {", ".join(context_items[0].get("lab_tests", []))} יכולות לעזור לאשר את האבחנה.' if context_items[0].get("lab_tests") else 'בדיקה פשוטה יכולה לעזור לאשר את המצב.'}
- **טיפים להקלה**: {f'{", ".join(context_items[0]["treatment"]["primary"])}.' if context_items[0].get("treatment", {}).get("primary") else 'נסה לנוח ולהקפיד על שתייה מרובה בינתיים.'} {f'אם הרופא ממליץ, אפשר לשקול {", ".join(context_items[0]["treatment"].get("medications", []))}.' if context_items[0].get("treatment", {}).get("medications") else ''} {f'משך המחלה בדרך כלל הוא {context_items[0].get("duration", "לא ידוע")}.' if context_items[0].get("duration") else ''}
- **איך למנוע החמרה**: {f'{", ".join(context_items[0]["prevention"])}.' if context_items[0].get("prevention") else 'שטוף ידיים לעיתים קרובות והימנע ממגע עם אנשים חולים, במיוחד אם זה מדבק.'} {f'שים לב שזה {context_items[0].get("contagious", False) and "כן מדבק" or "לא מדבק"}.' if "contagious" in context_items[0] else ''}

**מתי לרוץ לרופא**:
אם אתה מרגיש {f'{", ".join(context_items[0].get("red_flags", ["משהו ממש לא נורמלי"]))}' if context_items[0].get("red_flags") else 'כאב חזק, קושי לנשום, או שהמצב מחמיר'}, תפנה מיד לרופא או למיון.

אני כאן כדי לעזור, אז אם יש לך עוד שאלות או משהו לא ברור, תגיד לי! 😊

**שימו לב**: המידע הזה נועד לעזור לך להבין מה קורה, אבל הוא לא תחליף לבדיקה אצל רופא אמיתי."""

def hybrid_search(query: str, top_k: int = VECTOR_SEARCH_TOP_K):
    _ensure_initialized()
    
    try:
        enhanced_query = expand_medical_terms(normalize_hebrew_text(query))
        query_embedding = _embedder.encode([enhanced_query], convert_to_numpy=True)
        if query_embedding.shape[1] != _embedding_dim:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {_embedding_dim}")
        _index.nprobe = min(10, _index.ntotal // 10)
        embedding_scores, embedding_indices = _index.search(query_embedding, k=top_k * 2)
        # Normalize embedding scores to [0, 1]
        embedding_scores = embedding_scores[0]
        if embedding_scores.size > 0:
            embedding_scores = (embedding_scores - np.min(embedding_scores)) / (np.max(embedding_scores) - np.min(embedding_scores) + 1e-8)
        else:
            embedding_scores = np.zeros_like(embedding_scores)
        
        query_tfidf = _tfidf_vectorizer.transform([enhanced_query])
        tfidf_scores = cosine_similarity(query_tfidf, _tfidf_matrix).flatten()
        tfidf_indices = np.argsort(tfidf_scores)[::-1][:top_k * 2]
        
        combined_scores = {}
        for idx, score in zip(embedding_indices[0], embedding_scores):
            if idx >= 0:
                combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.6
        for idx in tfidf_indices:
            combined_scores[idx] = combined_scores.get(idx, 0) + tfidf_scores[idx] * 0.4
        
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        unique_indices = []
        seen_diseases = set()
        
        for idx, score in sorted_items:
            disease = normalize_hebrew_text(_medical_data[idx].get('disease', 'Unknown'))
            if not any(fuzz.ratio(disease, seen) > 80 for seen in seen_diseases):
                unique_indices.append(idx)
                seen_diseases.add(disease)
            if len(unique_indices) >= top_k:
                break
        
        candidates = [_medical_data[idx] for idx in unique_indices]
        candidate_scores = [combined_scores[idx] for idx in unique_indices]
        candidate_texts = [create_enhanced_text(item) for item in candidates]
        
        query_candidate_pairs = [[enhanced_query, text] for text in candidate_texts]
        rerank_scores = _reranker.predict(query_candidate_pairs)
        # Normalize and adjust rerank scores
        if rerank_scores.size > 0:
            rerank_scores = np.clip(rerank_scores, 0, None)
            rerank_scores = (rerank_scores - np.min(rerank_scores)) / (np.max(rerank_scores) - np.min(rerank_scores) + 1e-8)
            # Boost scores based on symptom match, prevalence, and category relevance
            query_terms = set(normalize_hebrew_text(query).split())
            for i, candidate in enumerate(candidates):
                primary_symptoms = set(normalize_hebrew_text(" ".join(candidate.get('symptoms_detailed', {}).get('primary', []))).split())
                secondary_symptoms = set(normalize_hebrew_text(" ".join(candidate.get('symptoms_detailed', {}).get('secondary', []))).split())
                primary_match_ratio = len(query_terms.intersection(primary_symptoms)) / max(len(query_terms), 1) * 1.5
                secondary_match_ratio = len(query_terms.intersection(secondary_symptoms)) / max(len(query_terms), 1) * 0.5
                prevalence_score = candidate.get('common', 3) / 5.0 * 1.5  # Increased prevalence weight
                # Category relevance boost
                category = candidate.get('category', '')
                category_score = 0
                for cat, keywords in CATEGORY_KEYWORDS.items():
                    if cat == category and any(kw in query_terms for kw in keywords):
                        category_score = 0.2
                        break
                rerank_scores[i] = rerank_scores[i] * 0.5 + primary_match_ratio * 0.2 + secondary_match_ratio * 0.1 + prevalence_score * 0.15 + category_score
        else:
            rerank_scores = np.zeros_like(rerank_scores)
        
        sorted_indices = np.argsort(rerank_scores)[::-1]
        
        final_items = [candidates[i] for i in sorted_indices]
        final_scores = [rerank_scores[i] for i in sorted_indices]
        
        return final_items, final_scores
    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}", exc_info=True)
        raise

def get_answer(query: str):
    _ensure_initialized()
    
    try:
        context_items, scores = hybrid_search(query)
        
        if not context_items:
            return "לא מצאתי מידע שמתאים בדיוק לתסמינים שתיארת. אולי תוכל להוסיף פרטים כמו כמה זמן זה נמשך, או אם יש תסמינים נוספים? אני כאן לעזור!", 0, 0.0
        
        prompt = _build_prompt(query, context_items)
        answer = call_llm(prompt)
        
        retrieved_count = len(context_items)
        avg_confidence = float(np.mean(scores)) if scores else 0.0
        
        return answer, retrieved_count, avg_confidence
    except Exception as e:
        logger.error(f"Error in get_answer: {e}", exc_info=True)
        raise

def get_answer_with_debug(query: str):
    _ensure_initialized()
    
    try:
        context_items, scores = hybrid_search(query)
        
        prompt = _build_prompt(query, context_items)
        answer = call_llm(prompt)
        
        retrieved_count = len(context_items)
        avg_confidence = float(np.mean(scores)) if scores else 0.0
        
        debug_info = {
            "embedder_model": EMBEDDER_MODEL,
            "rerank_model": RERANK_MODEL,
            "embedding_dim": _embedding_dim,
            "vector_search_top_k": VECTOR_SEARCH_TOP_K,
            "index_path": INDEX_PATH,
            "tfidf_index_path": TFIDF_INDEX_PATH,
            "index_built": _index_built,
            "retrieved_items": [
                {
                    "disease": item.get("disease"),
                    "symptoms": item.get("symptoms"),
                    "score": float(score)
                }
                for item, score in zip(context_items, scores)
            ],
            "prompt_preview": prompt[:700],
            "lazy_init": LAZY_INIT,
            "initialized": _initialized,
        }
        return answer, debug_info, retrieved_count, avg_confidence
    except Exception as e:
        logger.error(f"Error in get_answer_with_debug: {e}", exc_info=True)
        raise

if not LAZY_INIT:
    _initialize_if_needed()