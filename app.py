import os
import re
import io
import json
from flask import Flask, render_template, request, jsonify, send_file, Response
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# -------------------------------------------------------------
# NLTK Setup
# -------------------------------------------------------------
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload for multi-files

# Common technical skill dictionary for granular skill extraction
TECH_SKILLS_DB = {
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
    'html', 'html5', 'css', 'css3', 'react', 'react.js', 'vue', 'vue.js', 'angular', 'next.js', 'node.js', 'express',
    'flask', 'django', 'fastapi', 'spring', 'spring boot', 'laravel', 'asp.net',
    'sql', 'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'cassandra', 'dynamodb', 'oracle',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git', 'github', 'gitlab', 'jenkins', 'terraform', 'ansible',
    'linux', 'bash', 'rest api', 'restful', 'graphql', 'grpc', 'microservices', 'agile', 'scrum',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'scikit-learn', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv', 'data analysis', 'data science', 'power bi', 'tableau',
    'spark', 'hadoop', 'kafka', 'airflow', 'etl', 'computer vision', 'llm', 'generative ai'
}

def get_stop_words():
    try:
        return set(stopwords.words('english'))
    except Exception:
        return {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'it', 'as', 'will', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'we', 'they', 'you', 'he', 'she'}

# -------------------------------------------------------------
# Module 6.1: Resume Upload & Extraction Module
# -------------------------------------------------------------
def extract_text_from_pdf(pdf_file_stream):
    """Extracts raw text from PDF stream using PyPDF2."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

# -------------------------------------------------------------
# Module 6.3: Text Preprocessing Module
# -------------------------------------------------------------
def preprocess_text(text):
    """
    Cleans extracted text by:
    1. Lowercasing
    2. Removing URLs, emails & unwanted special characters
    3. Tokenizing and removing NLTK stopwords
    """
    if not text:
        return "", []
    
    raw_lower = text.lower()
    # Remove URLs and emails
    cleaned = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', ' ', raw_lower)
    # Remove punctuation except technical modifiers (+ and #)
    cleaned = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', cleaned)
    
    stop_words = get_stop_words()
    tokens = [w.strip('.') for w in cleaned.split() if w.strip('.')]
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1 and not w.isnumeric()]
    
    processed_string = ' '.join(filtered_tokens)
    return processed_string, filtered_tokens

# -------------------------------------------------------------
# Module 6.4 & 6.5: Feature Extraction & Similarity Module
# -------------------------------------------------------------
def compute_tfidf_similarity(resume_processed, jd_processed):
    """
    Converts text data into numerical TF-IDF feature vectors and
    computes Cosine Similarity between Resume and Job Description.
    """
    if not resume_processed or not jd_processed:
        return 0.0, [], []
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
    
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = round(float(sim) * 100, 2)
    
    # Extract top TF-IDF feature weights for JD
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray()[0]
    top_indices = jd_vector.argsort()[::-1][:15]
    
    top_features = [{"feature": feature_names[i], "weight": round(float(jd_vector[i]), 3)} 
                    for i in top_indices if jd_vector[i] > 0]
    
    return score, top_features, vectorizer

# -------------------------------------------------------------
# Module 6.6: Skill Analysis & Ranking Module
# -------------------------------------------------------------
def extract_skills_and_keywords(text):
    """Extracts detected skills (from skill dictionary) and top high-frequency keywords."""
    text_lower = text.lower()
    matched_skills = set()
    
    for skill in TECH_SKILLS_DB:
        # Regex word boundary check for each technical skill
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            matched_skills.add(skill)
            
    return matched_skills

def analyze_single_candidate(candidate_name, raw_resume_text, raw_jd_text):
    """Executes the full pipeline for a single candidate."""
    resume_processed, resume_tokens = preprocess_text(raw_resume_text)
    jd_processed, jd_tokens = preprocess_text(raw_jd_text)
    
    match_score, top_features, _ = compute_tfidf_similarity(resume_processed, jd_processed)
    
    resume_skills = extract_skills_and_keywords(raw_resume_text)
    jd_skills = extract_skills_and_keywords(raw_jd_text)
    
    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills.difference(resume_skills)))
    
    # Generate tailored ATS recommendations
    recommendations = []
    if match_score >= 70:
        recommendations.append("🌟 High Compatibility: Resume demonstrates strong alignment with job requirements.")
    elif match_score >= 45:
        recommendations.append("👍 Moderate Match: Candidate meets core criteria; adding specific missing skills will boost ATS rank.")
    else:
        recommendations.append("⚠️ Low Match: Key technical proficiencies are absent or under-represented.")
        
    if missing_skills:
        recommendations.append(f"Recommended keywords to include: {', '.join(missing_skills[:8])}.")
        
    word_count = len(raw_resume_text.split())
    if word_count < 150:
        recommendations.append("Resume content is short (<150 words). Provide more detailed project/work descriptions.")
    elif word_count > 1200:
        recommendations.append("Resume length is long (>1200 words). Aim for a focused 1-2 page format.")
        
    return {
        "candidate_name": candidate_name,
        "match_score": match_score,
        "resume_word_count": word_count,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "top_features": top_features,
        "recommendations": recommendations,
        "preprocessing_summary": {
            "raw_word_count": len(raw_resume_text.split()),
            "processed_tokens_count": len(resume_tokens),
            "sample_tokens": resume_tokens[:12]
        }
    }

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Module 6.1 - 6.6 Single Candidate Analysis Endpoint."""
    try:
        jd_text = request.form.get('job_description', '').strip()
        if not jd_text:
            return jsonify({'success': False, 'error': 'Please provide a target job description.'}), 400
        
        resume_text = ""
        candidate_name = "Candidate Resume"
        
        if 'resume_pdf' in request.files and request.files['resume_pdf'].filename != '':
            pdf_file = request.files['resume_pdf']
            candidate_name = pdf_file.filename
            resume_text = extract_text_from_pdf(io.BytesIO(pdf_file.read()))
        elif 'resume_text' in request.form and request.form.get('resume_text', '').strip() != '':
            resume_text = request.form.get('resume_text', '').strip()
            candidate_name = "Direct Text Input"
            
        if not resume_text.strip():
            return jsonify({'success': False, 'error': 'Please upload a PDF file or paste resume text.'}), 400
            
        result = analyze_single_candidate(candidate_name, resume_text, jd_text)
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Module 6.6 Multi-Resume Batch Ranking Endpoint."""
    try:
        jd_text = request.form.get('job_description', '').strip()
        if not jd_text:
            return jsonify({'success': False, 'error': 'Please provide a target job description.'}), 400
            
        uploaded_files = request.files.getlist('resumes_pdf')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'success': False, 'error': 'Please select multiple PDF resume files for ranking.'}), 400
            
        candidates_results = []
        for file in uploaded_files:
            if file and file.filename.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(io.BytesIO(file.read()))
                if raw_text.strip():
                    analysis = analyze_single_candidate(file.filename, raw_text, jd_text)
                    candidates_results.append(analysis)
                    
        if not candidates_results:
            return jsonify({'success': False, 'error': 'Could not extract valid text from the provided PDF files.'}), 400
            
        # Rank candidates in descending order of match_score using Pandas
        df = pd.DataFrame(candidates_results)
        df_sorted = df.sort_values(by='match_score', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        
        ranked_list = df_sorted.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'total_evaluated': len(ranked_list),
            'ranked_candidates': ranked_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export_ranking_csv', methods=['POST'])
def export_ranking_csv():
    """Exports batch candidate rankings to CSV using Pandas."""
    try:
        data = request.get_json()
        if not data or 'candidates' not in data:
            return jsonify({'error': 'No candidate data provided'}), 400
            
        records = []
        for idx, c in enumerate(data['candidates'], 1):
            records.append({
                'Rank': idx,
                'Candidate / Filename': c.get('candidate_name', 'N/A'),
                'Match Score (%)': c.get('match_score', 0),
                'Matched Skills Count': len(c.get('matched_skills', [])),
                'Matched Skills': ', '.join(c.get('matched_skills', [])),
                'Missing Skills Count': len(c.get('missing_skills', [])),
                'Missing Skills': ', '.join(c.get('missing_skills', []))
            })
            
        df = pd.DataFrame(records)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return Response(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=resume_screening_ranking_report.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
