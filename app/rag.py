"""RAG service with lazy initialization to reduce startup memory footprint.

On low-memory platforms (e.g., Render free tier 512Mi), loading the
SentenceTransformer and computing embeddings at import may OOM. This refactor
defers all heavy work until the first query.
"""

import json
import os
import re
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

# Handle both direct execution and module import contexts
try:
    # When imported as a module (e.g., from tests)
    from .model import call_llm
except ImportError:
    # When run directly or imported from another module in same directory
    from model import call_llm

from dotenv import load_dotenv, find_dotenv
from threading import Lock

# Load environment variables from .env file (resolve regardless of CWD)
env_path = find_dotenv(usecwd=True) or str((Path(__file__).resolve().parent.parent / ".env"))
load_dotenv(dotenv_path=env_path)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "diseases.json"))
INDEX_PATH = os.getenv("INDEX_PATH", str(PROJECT_ROOT / "index" / "faiss.index"))
TFIDF_INDEX_PATH = os.getenv("TFIDF_INDEX_PATH", str(PROJECT_ROOT / "index" / "tfidf.pkl"))
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))
LAZY_INIT = os.getenv("RAG_LAZY_INIT", "true").lower() == "true"

_state_lock = Lock()
_initialized = False
_embedder = None
_medical_data = None
_index = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_embedding_dim = None
_index_built = False

# Hebrew medical terms and synonyms - מינימלי בלבד
HEBREW_MEDICAL_SYNONYMS = {
    "חום": ["חום גבוה", "קדחת"],
    "כאב": ["כאבים"],
    "בחילה": ["בחילות", "הקאה"],
    "שלשול": ["שלשולים"],
    "עצירות": ["קושי ביציאות"],
    "כאב ראש": ["מיגרנה"],
    "כאב בטן": ["כאבי בטן"],
    "עייפות": ["תשישות", "חולשה"],
    "סחרחורת": ["סחרחורות"],
    "קוצר נשימה": ["קשיי נשימה"],
    "שיעול": ["שיעול יבש", "שיעול עם ליחה"],
    "נזלת": ["גודש באף"],
    "כאב גרון": ["דלקת גרון"],
    "גרד": ["גירוד"],
    "דמעת": ["דמעות"]
}

def safe_float_convert(value):
    """Safely convert value to float, handling numpy types"""
    try:
        if hasattr(value, '__float__'):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

def normalize_hebrew_text(text):
    """נרמול טקסט עברי עם טיפול בניקוד וסימני פיסוק"""
    if not text:
        return ""
    
    print(f"   🔧 Normalizing text: '{text}'")
    
    # הסרת ניקוד
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    print(f"   🔧 After removing diacritics: '{text}'")
    
    # המרה לאותיות קטנות
    text = text.lower()
    print(f"   🔧 After converting to lowercase: '{text}'")
    
    # הסרת סימני פיסוק מיותרים
    text = re.sub(r'[^\u0590-\u05FF\u0020-\u007F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    result = text.strip()
    print(f"   🔧 Final result: '{result}'")
    
    return result

def expand_medical_terms(text):
    """הרחבת מונחים רפואיים עם מילים נרדפות"""
    print(f"   🔍 Expanding medical terms: '{text}'")
    
    normalized_text = normalize_hebrew_text(text)
    expanded_terms = []
    
    print(f"   🔍 Searching for synonyms...")
    for term, synonyms in HEBREW_MEDICAL_SYNONYMS.items():
        if term in normalized_text:
            print(f"   🔍 Found term: '{term}' -> synonyms: {synonyms}")
            expanded_terms.extend(synonyms)
    
    if expanded_terms:
        result = text + " " + " ".join(expanded_terms)
        print(f"   🔍 Expanded text: '{result}'")
        return result
    
    print(f"   🔍 No synonyms found, returning original text")
    return text

def create_enhanced_text(item):
    """יצירת טקסט מועשר עם כל המידע הרלוונטי"""
    disease = item.get('disease', '')
    print(f"   📝 Creating enhanced text for: {disease}")
    
    # התחלה עם שם המחלה
    base_text = f"{disease}"
    print(f"   📝 Base text: '{base_text}'")
    
    # הוספת תסמינים ראשיים - חשוב מאוד!
    if 'symptoms_detailed' in item:
        symptoms_detailed = item['symptoms_detailed']
        if 'primary' in symptoms_detailed:
            primary_symptoms = " ".join(symptoms_detailed['primary'])
            base_text += " " + primary_symptoms
            print(f"   📝 Adding primary symptoms: '{primary_symptoms}'")
        if 'secondary' in symptoms_detailed:
            secondary_symptoms = " ".join(symptoms_detailed['secondary'])
            base_text += " " + secondary_symptoms
            print(f"   📝 Adding secondary symptoms: '{secondary_symptoms}'")
    
    # הוספת תסמינים מפורטים
    if 'symptoms' in item:
        symptoms = item['symptoms']
        base_text += f" {symptoms}"
        print(f"   📝 Adding detailed symptoms: '{symptoms}'")
    
    # הוספת קטגוריה ותת-קטגוריה
    if 'category' in item:
        category = item['category']
        base_text += f" {category}"
        print(f"   📝 Adding category: '{category}'")
    if 'subcategory' in item:
        subcategory = item['subcategory']
        base_text += f" {subcategory}"
        print(f"   📝 Adding subcategory: '{subcategory}'")
    
    # הוספת סיבוכים
    if 'complications' in item:
        complications = " ".join(item['complications'])
        base_text += " " + complications
        print(f"   📝 Adding complications: '{complications}'")
    
    # הוספת אבחנה מבדלת
    if 'differential_diagnosis' in item:
        diff_diagnosis = " ".join(item['differential_diagnosis'])
        base_text += " " + diff_diagnosis
        print(f"   📝 Adding differential diagnosis: '{diff_diagnosis}'")
    
    # הוספת מידע על שכיחות
    if 'common' in item:
        common = item['common']
        base_text += f" שכיחות {common}"
        print(f"   📝 Adding commonality: {common}")
    
    # הוספת מידע על חומרה
    if 'symptoms_detailed' in item and 'severity_scale' in item['symptoms_detailed']:
        severity = item['symptoms_detailed']['severity_scale']
        base_text += f" חומרה {severity}"
        print(f"   📝 Adding severity: '{severity}'")
    
    print(f"   📝 Text before expansion: '{base_text[:100]}...'")
    
    # הרחבת מונחים רפואיים
    enhanced_text = expand_medical_terms(base_text)
    
    print(f"   📝 Final text: '{enhanced_text[:100]}...'")
    
    return enhanced_text

def _initialize_if_needed():
    global _initialized, _embedder, _medical_data, _index, _tfidf_vectorizer, _tfidf_matrix, _embedding_dim, _index_built
    if _initialized and _embedder is not None:
        return
    with _state_lock:
        if _initialized and _embedder is not None:
            return
        
        print("Initializing RAG system...")
        
        # Load data
        with open(DATA_PATH, encoding="utf-8") as f:
            _medical_data = json.load(f)["data"]
        
        print(f"Loaded {len(_medical_data)} medical records")
        
        # Load model (better multilingual model for Hebrew)
        _embedder = SentenceTransformer(EMBEDDER_MODEL)
        
        # Create enhanced texts for better search
        print(f"\n📝 === CREATING ENHANCED TEXTS ===")
        enhanced_texts = []
        for i, item in enumerate(_medical_data):
            if i < 3:  # רק 3 הראשונים לדוגמה
                enhanced_text = create_enhanced_text(item)
                enhanced_texts.append(enhanced_text)
            else:
                enhanced_text = create_enhanced_text(item)
                enhanced_texts.append(enhanced_text)
        
        print(f"📝 Created {len(enhanced_texts)} enhanced texts")
        
        # Build / load FAISS index
        if os.path.exists(INDEX_PATH):
            _index = faiss.read_index(INDEX_PATH)
            _embedding_dim = _index.d
            print(f"Loaded existing FAISS index with dimension {_embedding_dim}")
        else:
            print("Building new FAISS index...")
            embeddings = _embedder.encode(enhanced_texts, convert_to_numpy=True, show_progress_bar=True)
            _embedding_dim = int(embeddings.shape[1])
            Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
            
            # Use IndexFlatIP for better similarity search
            _index = faiss.IndexFlatIP(_embedding_dim)
            _index.add(embeddings)
            faiss.write_index(_index, INDEX_PATH)
            _index_built = True
            print(f"Built FAISS index with {len(embeddings)} vectors")
            del embeddings
        
        # Build TF-IDF index for keyword-based search
        if os.path.exists(TFIDF_INDEX_PATH):
            import pickle
            with open(TFIDF_INDEX_PATH, 'rb') as f:
                tfidf_data = pickle.load(f)
                _tfidf_vectorizer = tfidf_data['vectorizer']
                _tfidf_matrix = tfidf_data['matrix']
            print("Loaded existing TF-IDF index")
        else:
            print("Building TF-IDF index...")
            _tfidf_vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'[\u0590-\u05FF]+',
                max_features=10000,
                ngram_range=(1, 2)
            )
            _tfidf_matrix = _tfidf_vectorizer.fit_transform(enhanced_texts)
            
            # Save TF-IDF index
            import pickle
            tfidf_data = {
                'vectorizer': _tfidf_vectorizer,
                'matrix': _tfidf_matrix
            }
            with open(TFIDF_INDEX_PATH, 'wb') as f:
                pickle.dump(tfidf_data, f)
            print("Built TF-IDF index")
        
        _initialized = True
        print("RAG system initialized successfully")

def _ensure_initialized():
    if not _initialized:
        _initialize_if_needed()

def _build_prompt(query: str, context_items):
    context_str = "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in context_items)
    return f"""אתה עוזר רפואי חכם ומומחה. על סמך הנתונים הבאים, ענה על השאלה בצורה מאוזנת ומעשית:

שאלה: {query}

מידע רפואי רלוונטי:
{context_str}

הוראות חשובות:
1. ענה בעברית ברורה ומקצועית
2. התחל עם האפשרויות הנפוצות והפשוטות יותר
3. ציין את שכיחות המחלה (נפוץ/שכיח/נדיר)
4. אל תתמקד רק במחלות חמורות - כלול גם מצבים קלים ונפוצים
5. תן הקשר - האם זה מצב דחוף או שניתן לחכות
6. ציין מתי יש צורך לפנות לרופא
7. תן המלצות מעשיות לטיפול ראשוני
8. הרגע את המשתמש - אל תפחיד אותו ללא סיבה

תשובה:"""

def hybrid_search(query: str, top_k: int = 5):
    """חיפוש היברידי המשלב embedding ו-TF-IDF"""
    _ensure_initialized()
    
    print(f"\n🔍 === DETAILED SEARCH PROCESS ===")
    print(f"📝 Original query: '{query}'")
    
    # נרמול השאלה
    normalized_query = normalize_hebrew_text(query)
    print(f"📝 Normalized query: '{normalized_query}'")
    
    enhanced_query = expand_medical_terms(query)
    print(f"📝 Enhanced query: '{enhanced_query}'")
    
    # חיפוש מדויק לפי מילות מפתח
    print(f"\n🔍 === EXACT KEYWORD SEARCH ===")
    exact_matches = []
    for i, item in enumerate(_medical_data):
        # בדיקת התאמה מדויקת לתסמינים ראשיים
        if 'symptoms_detailed' in item and 'primary' in item['symptoms_detailed']:
            primary_symptoms = item['symptoms_detailed']['primary']
            query_words = normalized_query.split()
            matches = sum(1 for word in query_words if any(word in symptom for symptom in primary_symptoms))
            if matches >= 2:  # לפחות 2 מילים תואמות
                exact_matches.append((i, matches, item))
                print(f"   EXACT MATCH: {item.get('disease', 'Unknown')} - {matches} matches")
    
    print(f"\n🔍 === EMBEDDING SEARCH ===")
    # חיפוש embedding
    query_embedding = _embedder.encode([enhanced_query], convert_to_numpy=True)
    embedding_scores, embedding_indices = _index.search(query_embedding, k=top_k*2)
    
    print(f"📊 Embedding results (Top {top_k*2}):")
    for i, (score, idx) in enumerate(zip(embedding_scores[0], embedding_indices[0])):
        disease = _medical_data[idx].get('disease', 'Unknown')
        print(f"   {i+1}. {disease} - Score: {score:.3f} (Index: {idx})")
    
    print(f"\n🔍 === TF-IDF SEARCH ===")
    # חיפוש TF-IDF
    query_tfidf = _tfidf_vectorizer.transform([enhanced_query])
    tfidf_scores = cosine_similarity(query_tfidf, _tfidf_matrix).flatten()
    tfidf_indices = np.argsort(tfidf_scores)[::-1][:top_k*2]
    
    print(f"📊 TF-IDF results (Top {top_k*2}):")
    for i, idx in enumerate(tfidf_indices):
        score = tfidf_scores[idx]
        disease = _medical_data[idx].get('disease', 'Unknown')
        print(f"   {i+1}. {disease} - Score: {score:.3f} (Index: {idx})")
    
    print(f"\n🔍 === COMBINING RESULTS ===")
    # שילוב התוצאות
    combined_scores = {}
    
    # הוספת תוצאות חיפוש מדויק (משקל גבוה)
    if exact_matches:
        print(f"📊 Adding exact matches (weight: 1.0):")
        for idx, matches, item in exact_matches:
            combined_scores[idx] = matches * 1.0  # משקל גבוה לחיפוש מדויק
            disease = item.get('disease', 'Unknown')
            print(f"   {disease}: {matches} exact matches * 1.0 = {matches * 1.0:.3f}")
    
    # הוספת תוצאות embedding
    print(f"📊 Adding Embedding results (weight: 0.6):")
    for i, (score, idx) in enumerate(zip(embedding_scores[0], embedding_indices[0])):
        if idx not in combined_scores:
            combined_scores[idx] = 0
        combined_scores[idx] += score * 0.6
        disease = _medical_data[idx].get('disease', 'Unknown')
        print(f"   {disease}: {score:.3f} * 0.6 = {score * 0.6:.3f}")
    
    # הוספת תוצאות TF-IDF
    print(f"📊 Adding TF-IDF results (weight: 0.4):")
    for idx in tfidf_indices:
        if idx not in combined_scores:
            combined_scores[idx] = 0
        tfidf_score = tfidf_scores[idx]
        combined_scores[idx] += tfidf_score * 0.4
        disease = _medical_data[idx].get('disease', 'Unknown')
        print(f"   {disease}: {tfidf_score:.3f} * 0.4 = {tfidf_score * 0.4:.3f}")
    
    # מיון לפי ציון משולב
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # סינון כפילויות - רק מחלה אחת מכל סוג
    unique_indices = []
    seen_diseases = set()
    
    for idx, score in sorted_items:
        disease = _medical_data[idx].get('disease', 'Unknown')
        if disease not in seen_diseases:
            unique_indices.append(idx)
            seen_diseases.add(disease)
            if len(unique_indices) >= top_k:
                break
    
    print(f"\n🏆 === FINAL RESULTS (Top {len(unique_indices)}) ===")
    for i, idx in enumerate(unique_indices):
        disease = _medical_data[idx].get('disease', 'Unknown')
        score = combined_scores[idx]
        print(f"   {i+1}. {disease} - Combined score: {score:.3f} (Index: {idx})")
    
    return [_medical_data[idx] for idx in unique_indices], [combined_scores[idx] for idx in unique_indices]

def get_answer(query: str):
    _ensure_initialized()
    
    # חיפוש היברידי
    context_items, scores = hybrid_search(query, VECTOR_SEARCH_TOP_K)
    
    # סינון חכם לפי התאמה לתסמינים
    filtered_items = []
    filtered_scores = []
    query_words = normalize_hebrew_text(query).split()
    
    print(f"\n🔍 === SMART FILTERING ===")
    print(f"Query words: {query_words}")
    
    for item, score in zip(context_items, scores):
        # בדיקת התאמה לתסמינים ראשיים
        relevance_score = 0
        if 'symptoms_detailed' in item and 'primary' in item['symptoms_detailed']:
            primary_symptoms = item['symptoms_detailed']['primary']
            print(f"   Checking: {item.get('disease', 'Unknown')}")
            print(f"   Primary symptoms: {primary_symptoms}")
            
            # חישוב התאמה מדויקת
            matches = 0
            for word in query_words:
                for symptom in primary_symptoms:
                    if word in symptom or symptom in word:
                        matches += 1
                        print(f"     MATCH: '{word}' in '{symptom}'")
            
            relevance_score = matches / len(query_words) if query_words else 0
            print(f"   Relevance score: {relevance_score:.3f}")
        
        # בדיקה נוספת - האם המחלה קשורה לעיניים ולא למחלות מערכתיות
        is_eye_related = False
        disease_name = item.get('disease', '').lower()
        category = item.get('category', '').lower()
        
        # מילות מפתח שמעידות על מחלת עיניים
        eye_keywords = ['עיניים', 'עין', 'לחמית', 'קרנית', 'דלקת עיניים', 'דלקת לחמית']
        if any(keyword in disease_name for keyword in eye_keywords) or any(keyword in category for keyword in eye_keywords):
            is_eye_related = True
        
        # מילות מפתח שמעידות על מחלה מערכתית (לסינון)
        systemic_keywords = ['תסמונת', 'מחלה אוטואימונית', 'מחלה מערכתית', 'פוליציתמיה', 'שורגן']
        is_systemic = any(keyword in disease_name for keyword in systemic_keywords)
        
        # רק מחלות עם התאמה טובה ורלוונטיות
        if (relevance_score >= 0.6 or score >= 4.0) and (is_eye_related or not is_systemic):
            filtered_items.append(item)
            filtered_scores.append(score)
            print(f"   ✅ ACCEPTED: {item.get('disease', 'Unknown')} - relevance: {relevance_score:.3f}, score: {score:.3f}, eye_related: {is_eye_related}")
        else:
            reason = "score too low" if relevance_score < 0.6 and score < 4.0 else "systemic disease" if is_systemic else "not eye related"
            print(f"   ❌ REJECTED: {item.get('disease', 'Unknown')} - relevance: {relevance_score:.3f}, score: {score:.3f}, eye_related: {is_eye_related}, reason: {reason}")
    
    if not filtered_items:
        return "לא מצאתי מידע רלוונטי לשאלתך. נסה לפרט יותר את הסימפטומים או השתמש במילים אחרות."
    
    # בניית prompt עם מידע מפורט
    prompt = _build_prompt(query, filtered_items)
    
    try:
        answer = call_llm(prompt)
        
        # הוספת מידע על התוצאות
        debug_info = "\n\nמידע על התוצאות שנמצאו:"
        for i, (item, score) in enumerate(zip(filtered_items, filtered_scores)):
            score_float = safe_float_convert(score)
            debug_info += f"\n{i+1}. {item.get('disease', 'N/A')} - ציון התאמה: {score_float:.3f}"
        
        return answer + debug_info
        
    except Exception as e:
        return f"שגיאה בקבלת תשובה: {str(e)}"

def get_answer_with_debug(query: str):
    """Returns (answer, debug_info) including embedding and search details."""
    _ensure_initialized()
    
    # חיפוש היברידי
    context_items, scores = hybrid_search(query, VECTOR_SEARCH_TOP_K)
    
    prompt = _build_prompt(query, context_items)
    answer = call_llm(prompt)
    
    debug_info = {
        "embedder_model": EMBEDDER_MODEL,
        "embedding_dim": _embedding_dim,
        "vector_search_top_k": VECTOR_SEARCH_TOP_K,
        "index_path": INDEX_PATH,
        "tfidf_index_path": TFIDF_INDEX_PATH,
        "index_built": _index_built,
        "retrieved_items": [
            {
                "disease": item.get("disease"), 
                "symptoms": item.get("symptoms"),
                "score": safe_float_convert(score)
            }
            for item, score in zip(context_items, scores)
        ],
        "prompt_preview": prompt[:700],
        "lazy_init": LAZY_INIT,
        "initialized": _initialized,
    }
    return answer, debug_info

# Optionally eager initialize if lazy disabled
if not LAZY_INIT:
    _initialize_if_needed()
