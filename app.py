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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload for batch uploads

# Standard Technical Skills Dictionary for granular skill gap analysis
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

def extract_text_from_file(file_obj, filename):
    """
    Extracts plain text from either a PDF or TXT file stream safely.
    Handles corrupted files and encodings gracefully.
    """
    filename_lower = filename.lower()
    text = ""
    try:
        if filename_lower.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file_obj)
            if len(pdf_reader.pages) == 0:
                return False, "PDF contains 0 pages."
            for page_num, page in enumerate(pdf_reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename_lower.endswith('.txt'):
            raw_bytes = file_obj.read()
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            return False, "Unsupported file format. Please upload PDF or TXT."
    except Exception as e:
        return False, f"Unable to read file: {str(e)}"

    text = text.strip()
    if not text:
        return False, "File is empty or contains no readable text."
    return True, text

def preprocess_text(text):
    """
    NLP Preprocessing:
    1. Lowercase normalization
    2. Strip URLs, emails, special punctuation
    3. Tokenization and stopword removal
    """
    if not text:
        return "", []
    
    raw_lower = text.lower()
    cleaned = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', ' ', raw_lower)
    cleaned = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', cleaned)
    
    stop_words = get_stop_words()
    tokens = [w.strip('.') for w in cleaned.split() if w.strip('.')]
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1 and not w.isnumeric()]
    
    return ' '.join(filtered_tokens), filtered_tokens

def extract_skills_from_text(text):
    """Identifies technical skills present in text using regex boundary matching."""
    text_lower = text.lower()
    found_skills = set()
    for skill in TECH_SKILLS_DB:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def compute_similarity_and_features(resume_clean, jd_clean):
    """
    Computes Cosine Similarity using TF-IDF (1-gram and 2-gram).
    Extracts top matching features and term weights.
    """
    if not resume_clean or not jd_clean:
        return 0.0, [], []
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = round(float(sim) * 100, 2)
    
    feature_names = vectorizer.get_feature_names_out()
    resume_vec = tfidf_matrix[0].toarray()[0]
    jd_vec = tfidf_matrix[1].toarray()[0]
    
    # Identify overlapping features with non-zero weights in both
    matching_terms = []
    for idx, name in enumerate(feature_names):
        if resume_vec[idx] > 0 and jd_vec[idx] > 0:
            matching_terms.append({
                "term": name,
                "jd_weight": round(float(jd_vec[idx]), 3),
                "resume_weight": round(float(resume_vec[idx]), 3)
            })
    
    matching_terms = sorted(matching_terms, key=lambda x: x["jd_weight"], reverse=True)[:10]
    return score, matching_terms, vectorizer

def generate_match_explanation(score, matched_skills, missing_skills, matching_terms):
    """
    Generates transparent, explainable reasoning for why the candidate received the score.
    Based strictly on TF-IDF cosine similarity, term overlaps, and domain skill coverage.
    """
    reasons = []
    
    if score >= 75:
        reasons.append(f"Strong overall alignment ({score}% match). The resume shares high TF-IDF semantic overlap with the job description.")
    elif score >= 50:
        reasons.append(f"Moderate alignment ({score}% match). Core terminology is present, with room for additional technical coverage.")
    else:
        reasons.append(f"Low compatibility ({score}% match). The resume lacks several core qualifications and keywords specified in the job posting.")
        
    if matched_skills:
        reasons.append(f"Matched {len(matched_skills)} key technical requirement(s): {', '.join(matched_skills[:6])}.")
    else:
        reasons.append("No direct domain skills from the job description were detected in the resume.")
        
    if missing_skills:
        reasons.append(f"Identified {len(missing_skills)} missing requirement(s): {', '.join(missing_skills[:6])}.")
        
    if matching_terms:
        top_terms = [t["term"] for t in matching_terms[:5]]
        reasons.append(f"High-frequency matching terms: {', '.join(top_terms)}.")
        
    return reasons

def process_single_candidate(candidate_name, raw_text, jd_text):
    """Runs complete pipeline for a candidate and returns structured evaluation dict."""
    resume_clean, resume_tokens = preprocess_text(raw_text)
    jd_clean, jd_tokens = preprocess_text(jd_text)
    
    score, matching_terms, _ = compute_similarity_and_features(resume_clean, jd_clean)
    
    resume_skills = extract_skills_from_text(raw_text)
    jd_skills = extract_skills_from_text(jd_text)
    
    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills.difference(resume_skills)))
    
    reasons = generate_match_explanation(score, matched_skills, missing_skills, matching_terms)
    
    # Recommendations
    recommendations = []
    if missing_skills:
        recommendations.append(f"Add verifiable experience in: {', '.join(missing_skills[:6])}.")
    if len(raw_text.split()) < 150:
        recommendations.append("Resume content is short (<150 words). Expand on technical projects and quantifiable achievements.")
    elif len(raw_text.split()) > 1200:
        recommendations.append("Resume is lengthy (>1200 words). Condense into a concise 1-2 page format.")
    if score < 50:
        recommendations.append("Tailor bullet points to mirror target responsibilities and industry tools from the job description.")
        
    preview_words = raw_text.split()[:200]
    preview_text = " ".join(preview_words) + ("..." if len(raw_text.split()) > 200 else "")
    
    return {
        "candidate_name": candidate_name,
        "match_score": score,
        "word_count": len(raw_text.split()),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matching_terms": matching_terms,
        "explanation": reasons,
        "recommendations": recommendations,
        "preview_text": preview_text
    }

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Single candidate evaluation endpoint with robust validation."""
    try:
        jd_text = request.form.get('job_description', '').strip()
        if not jd_text:
            return jsonify({'success': False, 'error': 'Please enter a job description.'}), 400
        if len(jd_text) < 20 or len(jd_text.split()) < 4:
            return jsonify({'success': False, 'error': 'Job description is too short. Please provide detailed requirements.'}), 400

        raw_resume = ""
        candidate_name = "Candidate Resume"

        if 'resume_file' in request.files and request.files['resume_file'].filename != '':
            file_obj = request.files['resume_file']
            candidate_name = file_obj.filename
            success, result = extract_text_from_file(io.BytesIO(file_obj.read()), candidate_name)
            if not success:
                return jsonify({'success': False, 'error': f"Unable to read this resume: {result}"}), 400
            raw_resume = result
        elif 'resume_text' in request.form and request.form.get('resume_text', '').strip() != '':
            raw_resume = request.form.get('resume_text', '').strip()
            candidate_name = "Pasted Resume"

        if not raw_resume.strip():
            return jsonify({'success': False, 'error': 'Please upload at least one resume.'}), 400
        if len(raw_resume.split()) < 5:
            return jsonify({'success': False, 'error': 'The provided resume is too short or empty.'}), 400

        result = process_single_candidate(candidate_name, raw_resume, jd_text)
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': f"An error occurred during processing: {str(e)}"}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Multi-resume screening and ranking endpoint with summary analytics."""
    try:
        jd_text = request.form.get('job_description', '').strip()
        if not jd_text:
            return jsonify({'success': False, 'error': 'Please enter a job description.'}), 400
        if len(jd_text) < 20 or len(jd_text.split()) < 4:
            return jsonify({'success': False, 'error': 'Job description is too short. Please provide detailed requirements.'}), 400

        uploaded_files = request.files.getlist('resumes_files')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'success': False, 'error': 'Please upload at least one resume.'}), 400

        candidates_data = []
        errors = []
        seen_names = set()

        for file_obj in uploaded_files:
            fname = file_obj.filename
            if not fname:
                continue
            if fname in seen_names:
                continue
            seen_names.add(fname)

            success, content = extract_text_from_file(io.BytesIO(file_obj.read()), fname)
            if not success:
                errors.append(f"{fname}: {content}")
                continue

            analysis = process_single_candidate(fname, content, jd_text)
            candidates_data.append(analysis)

        if not candidates_data:
            err_msg = "Unable to process any uploaded resumes. " + ("; ".join(errors) if errors else "")
            return jsonify({'success': False, 'error': err_msg}), 400

        # Rank candidates using Pandas
        df = pd.DataFrame(candidates_data)
        df_sorted = df.sort_values(by='match_score', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        ranked_list = df_sorted.to_dict(orient='records')

        # Summary analytics
        scores = [c['match_score'] for c in ranked_list]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        top_cand = ranked_list[0]['candidate_name'] if ranked_list else "N/A"
        highest_score = ranked_list[0]['match_score'] if ranked_list else 0

        # Aggregate unique matched skills across candidates
        all_matched_skills = set()
        for c in ranked_list:
            all_matched_skills.update(c['matched_skills'])

        summary_metrics = {
            'candidates_analyzed': len(ranked_list),
            'top_candidate': top_cand,
            'highest_match': highest_score,
            'average_match': avg_score,
            'total_unique_skills_found': len(all_matched_skills),
            'warnings': errors
        }

        return jsonify({
            'success': True,
            'summary': summary_metrics,
            'ranked_candidates': ranked_list
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"An error occurred during screening: {str(e)}"}), 500

@app.route('/export_ranking_csv', methods=['POST'])
def export_ranking_csv():
    """Exports batch candidate rankings to CSV via Pandas."""
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
                'Word Count': c.get('word_count', 0),
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

@app.route('/export_ranking_json', methods=['POST'])
def export_ranking_json():
    """Exports structured candidate evaluations to downloadable JSON."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        json_str = json.dumps(data, indent=2)
        return Response(
            json_str,
            mimetype="application/json",
            headers={"Content-disposition": "attachment; filename=resume_screening_report.json"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
