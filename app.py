from flask import Flask, render_template_string, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model dan urutan kolom hasil training
model = joblib.load('rf_heart_model.pkl')
model_columns = joblib.load('model_columns.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioTree — Deteksi Penyakit Jantung</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background: #eef3f9; color: #1e293b; }
        .hero { background: linear-gradient(120deg, #0b3963, #1d70b8); color: #fff; padding: 2.2rem 0 4rem; }
        .main-wrap { margin-top: -2.8rem; padding-bottom: 3rem; }
        .card-box { background: #fff; border-radius: 18px; border: 1px solid #e2e8f0; box-shadow: 0 10px 28px rgba(15,76,129,.08); }
        .form-label { font-weight: 600; font-size: .85rem; color: #334155; }
        .form-control, .form-select { border-radius: 10px; font-size: .9rem; }
        .btn-run { background: linear-gradient(135deg, #0f4c81, #1d70b8); border: none; border-radius: 12px; font-weight: 700; padding: .8rem; }
    </style>
</head>
<body>
<div class="hero">
    <div class="container d-flex flex-wrap justify-content-between align-items-center gap-2">
        <div>
            <h1 class="h4 fw-bold mb-1"><i class="bi bi-heart-pulse-fill text-danger me-2"></i>CardioTree</h1>
            <p class="small text-white-50 mb-0">Prediksi Risiko Penyakit Jantung — Random Forest Classifier (TA-06)</p>
        </div>
        <span class="badge bg-white bg-opacity-10 border border-white border-opacity-25 px-3 py-2 rounded-pill">
            Ensemble Learning • Flask Single-File
        </span>
    </div>
</div>

<div class="container main-wrap">
    <div class="row g-4">
        <div class="col-lg-7">
            <div class="card-box p-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h6 fw-bold text-primary mb-0"><i class="bi bi-clipboard2-pulse me-1"></i> Data Klinis Pasien</h2>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-sm btn-outline-success fw-semibold" onclick="preset('sehat')">Contoh Sehat</button>
                        <button type="button" class="btn btn-sm btn-outline-danger fw-semibold" onclick="preset('risiko')">Contoh Risiko</button>
                    </div>
                </div>
                <form method="POST">
                    <div class="row g-3">
                        <div class="col-md-4"><label class="form-label">Usia (Tahun)</label><input type="number" id="age" name="age" class="form-control" value="{{ f.get('age', 54) }}" required></div>
                        <div class="col-md-4"><label class="form-label">Jenis Kelamin</label>
                            <select id="sex" name="sex" class="form-select">
                                <option value="1" {% if f.get('sex','1')=='1' %}selected{% endif %}>Laki-laki</option>
                                <option value="0" {% if f.get('sex')=='0' %}selected{% endif %}>Perempuan</option>
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Nyeri Dada (cp)</label>
                            <select id="cp" name="cp" class="form-select">
                                {% for v in ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'] %}
                                <option value="{{v}}" {% if f.get('cp','Asymptomatic')==v %}selected{% endif %}>{{v}}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Tekanan Darah</label><input type="number" id="trestbps" name="trestbps" class="form-control" value="{{ f.get('trestbps', 130) }}" required></div>
                        <div class="col-md-4"><label class="form-label">Kolesterol (mg/dl)</label><input type="number" id="chol" name="chol" class="form-control" value="{{ f.get('chol', 240) }}" required></div>
                        <div class="col-md-4"><label class="form-label">Gula Puasa &gt; 120</label>
                            <select id="fbs" name="fbs" class="form-select">
                                <option value="0" {% if f.get('fbs','0')=='0' %}selected{% endif %}>Tidak</option>
                                <option value="1" {% if f.get('fbs')=='1' %}selected{% endif %}>Ya</option>
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Detak Jantung Maks</label><input type="number" id="thalach" name="thalach" class="form-control" value="{{ f.get('thalach', 150) }}" required></div>
                        <div class="col-md-4"><label class="form-label">Angina Olahraga</label>
                            <select id="exang" name="exang" class="form-select">
                                <option value="0" {% if f.get('exang','0')=='0' %}selected{% endif %}>Tidak</option>
                                <option value="1" {% if f.get('exang')=='1' %}selected{% endif %}>Ya</option>
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">ST Depression</label><input type="number" step="0.1" id="oldpeak" name="oldpeak" class="form-control" value="{{ f.get('oldpeak', 1.2) }}" required></div>
                        <div class="col-md-4"><label class="form-label">EKG Istirahat</label>
                            <select id="restecg" name="restecg" class="form-select">
                                {% for v in ['Normal','ST-T Abnormality','LV Hypertrophy'] %}
                                <option value="{{v}}" {% if f.get('restecg','Normal')==v %}selected{% endif %}>{{v}}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Kemiringan ST</label>
                            <select id="slope" name="slope" class="form-select">
                                {% for v in ['Upsloping','Flat','Downsloping'] %}
                                <option value="{{v}}" {% if f.get('slope','Flat')==v %}selected{% endif %}>{{v}}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Pembuluh Utama (0-3)</label><input type="number" min="0" max="3" id="ca" name="ca" class="form-control" value="{{ f.get('ca', 0) }}" required></div>
                        <div class="col-12"><label class="form-label">Thalassemia (thal)</label>
                            <select id="thal" name="thal" class="form-select">
                                {% for v in ['Normal','Fixed Defect','Reversible Defect'] %}
                                <option value="{{v}}" {% if f.get('thal','Reversible Defect')==v %}selected{% endif %}>{{v}}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary btn-run w-100 mt-4 text-white">
                        <i class="bi bi-search-heart me-1"></i> Prediksi Risiko Jantung
                    </button>
                </form>
            </div>
        </div>

        <div class="col-lg-5">
            <div class="card-box p-4 h-100 d-flex flex-column justify-content-between">
                {% if res %}
                <div>
                    <div class="alert {{ 'alert-danger' if res.risk else 'alert-success' }} rounded-4 p-3 mb-3">
                        <h3 class="h6 fw-bold mb-1"><i class="bi {{ 'bi-exclamation-triangle-fill' if res.risk else 'bi-check-circle-fill' }} me-1"></i> {{ res.title }}</h3>
                        <p class="small mb-0">{{ res.desc }}</p>
                    </div>
                    <div class="d-flex justify-content-between small fw-bold mb-1">
                        <span class="text-danger">Risiko: {{ res.p_risk }}%</span>
                        <span class="text-success">Normal: {{ res.p_safe }}%</span>
                    </div>
                    <div class="progress mb-4" style="height:12px;border-radius:99px">
                        <div class="progress-bar bg-danger" style="width:{{ res.p_risk }}%"></div>
                        <div class="progress-bar bg-success" style="width:{{ res.p_safe }}%"></div>
                    </div>
                    <h6 class="small fw-bold text-uppercase text-muted mb-2">Top 4 Fitur Paling Berpengaruh</h6>
                    <div class="bg-light p-3 rounded-3 border">
                        {% for name, val in res.top %}
                        <div class="mb-2">
                            <div class="d-flex justify-content-between small"><span>{{ name }}</span><strong class="text-primary">{{ val }}%</strong></div>
                            <div class="progress" style="height:6px"><div class="progress-bar bg-primary" style="width:{{ val*3 }}%"></div></div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% else %}
                <div class="text-center my-auto py-5">
                    <i class="bi bi-heart-pulse text-primary opacity-75" style="font-size:3.5rem"></i>
                    <h3 class="h6 fw-bold mt-3">Siap Melakukan Prediksi</h3>
                    <p class="small text-muted">Isi parameter klinis di samping atau klik tombol preset untuk simulasi cepat.</p>
                </div>
                {% endif %}
                <div class="border-top pt-2 mt-3 text-center text-muted" style="font-size:.75rem"> Pembelajaran Mesin • Random Forest</div>
            </div>
        </div>
    </div>
</div>
<script>
function preset(t){
    const d = t==='sehat'
        ? {age:41,sex:'0',cp:'Atypical Angina',trestbps:118,chol:195,fbs:'0',thalach:175,exang:'0',oldpeak:0,restecg:'Normal',slope:'Upsloping',ca:0,thal:'Normal'}
        : {age:63,sex:'1',cp:'Asymptomatic',trestbps:155,chol:285,fbs:'1',thalach:118,exang:'1',oldpeak:2.8,restecg:'ST-T Abnormality',slope:'Flat',ca:2,thal:'Reversible Defect'};
    Object.keys(d).forEach(k => document.getElementById(k).value = d[k]);
}
</script>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
  res = None
  if request.method == 'POST':
    num_cols = [
        'age',
        'sex',
        'trestbps',
        'chol',
        'fbs',
        'thalach',
        'exang',
        'oldpeak',
        'ca',
    ]
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    row = {c: float(request.form[c]) for c in num_cols}
    row.update({c: request.form[c] for c in cat_cols})

    df_in = pd.get_dummies(pd.DataFrame([row]), columns=cat_cols).reindex(
        columns=model_columns, fill_value=0
    )
    pred = int(model.predict(df_in)[0])
    proba = model.predict_proba(df_in)[0]
    top = sorted(
        zip(model_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:4]

    res = {
        'risk': pred == 1,
        'title': (
            'TERINDIKASI RISIKO PENYAKIT JANTUNG'
            if pred == 1
            else 'KONDISI JANTUNG NORMAL'
        ),
        'desc': (
            'Disarankan konsultasi kardiologi lebih lanjut.'
            if pred == 1
            else 'Parameter klinis berada dalam batas aman.'
        ),
        'p_risk': round(proba[1] * 100, 1),
        'p_safe': round(proba[0] * 100, 1),
        'top': [(k, round(v * 100, 1)) for k, v in top],
    }
  return render_template_string(HTML_TEMPLATE, res=res, f=request.form)


if __name__ == '__main__':
  app.run(debug=True)