import os
import re
import io
from flask import Flask, render_template, request, jsonify
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data safely
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

def get_stop_words():
    try:
        return set(stopwords.words('english'))
    except Exception:
        return {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'it', 'as'}

def extract_text_from_pdf(pdf_file_stream):
    """Extracts all text from an uploaded PDF stream using PyPDF2."""
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

def clean_text(text):
    """Clean text by lowercasing, removing urls/special chars, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, top_n=25):
    """Extract top significant keywords from text (excluding stopwords)."""
    stop_words = get_stop_words()
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.-]{1,25}\b', text.lower())
    filtered = [w for w in words if w not in stop_words and len(w) > 2 and not w.isnumeric()]
    
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
        
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:top_n]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        resume_text = ""
        job_description = request.form.get('job_description', '').strip()

        # Check if PDF uploaded
        if 'resume_pdf' in request.files and request.files['resume_pdf'].filename != '':
            pdf_file = request.files['resume_pdf']
            resume_text = extract_text_from_pdf(io.BytesIO(pdf_file.read()))
        elif 'resume_text' in request.form and request.form.get('resume_text', '').strip() != '':
            resume_text = request.form.get('resume_text', '').strip()

        if not resume_text:
            return jsonify({'success': False, 'error': 'Please upload a PDF resume or paste resume text.'}), 400

        if not job_description:
            return jsonify({'success': False, 'error': 'Please provide the target job description.'}), 400

        # Text cleaning
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(job_description)

        if not clean_resume or not clean_jd:
            return jsonify({'success': False, 'error': 'Input text is too short or contains only invalid characters.'}), 400

        # Vectorization and similarity computation
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_percentage = round(float(similarity_score) * 100, 1)

        # Keyword & Skill Breakdown
        jd_keywords = extract_keywords(job_description, top_n=20)
        resume_words = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.-]{1,25}\b', clean_resume))

        matched_keywords = [kw for kw in jd_keywords if kw in resume_words]
        missing_keywords = [kw for kw in jd_keywords if kw not in resume_words]

        # Recommendation generation
        recommendations = []
        if match_percentage >= 75:
            recommendations.append("🌟 Outstanding Match! Your resume closely aligns with the job requirements.")
        elif match_percentage >= 50:
            recommendations.append("👍 Good Match! Incorporating the missing keywords below can boost your ATS pass rate.")
        else:
            recommendations.append("⚠️ Low Match. Consider tailoring your experience and skills to reflect the job requirements.")

        if missing_keywords:
            recommendations.append(f"Consider adding key terms such as: {', '.join(missing_keywords[:6])}.")

        word_count_resume = len(resume_text.split())
        if word_count_resume < 200:
            recommendations.append("Your resume appears quite brief (<200 words). Ensure you detail your responsibilities and achievements.")
        elif word_count_resume > 1200:
            recommendations.append("Your resume is quite extensive (>1200 words). Aim for a concise 1-2 page layout.")

        return jsonify({
            'success': True,
            'match_score': match_percentage,
            'resume_word_count': word_count_resume,
            'jd_word_count': len(job_description.split()),
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
