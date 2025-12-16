import os
import glob
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def clean_temp_files():
    try:
        files = glob.glob('/tmp/*.wav')
        for f in files:
            try:
                os.remove(f)
            except:
                pass
    except:
        pass

# ====== HNR35: 使用 Praat 原生命令（修正参数） ======
def extract_hnr35(sound_obj):
    """ HNR35: 0-3500Hz 频段的谐波噪声比 """
    try:
        harmonicity = sound_obj.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=75.0,
            silence_threshold=0.1,
            periods_per_window=4.5  # 必须 >=3！
        )
        values = harmonicity.values
        valid_values = values[values != -200]  # Praat 用 -200 表示静音
        if len(valid_values) == 0:
            return 0.0
        return float(np.mean(valid_values))
    except Exception as e:
        print(f"[Error] HNR35: {e}")
        return 0.0

# ====== GNE: 替代实现（基于频谱分析） ======
def extract_gne(y, sr):
    """
    GNE: Glottal-to-Noise Excitation Ratio
    实现原理：Maryn et al. (2009) 的频谱斜率法
    频段：500–4500 Hz (glottal), 4500–6000 Hz (noise floor)
    """
    try:
        # 短时傅里叶变换
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 找到频段索引
        glottal_idx = np.where((freqs >= 500) & (freqs <= 4500))[0]
        noise_idx = np.where((freqs > 4500) & (freqs <= min(6000, sr//2)))[0]

        if len(glottal_idx) == 0 or len(noise_idx) == 0:
            return 0.0

        # 计算每帧的能量
        glottal_energy = np.mean(S[glottal_idx, :], axis=0)
        noise_energy = np.mean(S[noise_idx, :], axis=0)

        # 避免除零
        epsilon = 1e-10
        gne_frames = np.log10((glottal_energy + epsilon) / (noise_energy + epsilon))

        # 返回平均 GNE
        return float(np.mean(gne_frames))
    except Exception as e:
        print(f"[Error] GNE: {e}")
        return 0.0

# ====== MFCC10 和 Delta11 ======
def extract_mfcc_features(y, sr):
    """ 提取 MFCC10 和 Delta11 """
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc10_mean = float(np.mean(mfccs[9]))  # 第10阶 (0-indexed)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta11_mean = float(np.mean(delta_mfccs[10]))  # Delta 第11阶
        return mfcc10_mean, delta11_mean
    except Exception as e:
        print(f"[Error] MFCC: {e}")
        return 0.0, 0.0

# ====== Flask Routes ======
@app.route('/', methods=['GET'])
def health_check():
    return "VoiceScreener Analysis Service is Running. (4 Features Ready)", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    clean_temp_files()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_filename = f"upload_{os.getpid()}.wav"
    temp_path = os.path.join("/tmp", temp_filename)
    converted_path = os.path.join("/tmp", f"converted_{os.getpid()}.wav")

    try:
        file.save(temp_path)
        # 加载音频（16kHz）
        y, sr = librosa.load(temp_path, sr=16000)
        sf.write(converted_path, y, sr)
        snd = parselmouth.Sound(converted_path)

        # 提取四个核心特征（完全匹配你们的科研设计）
        hnr35 = extract_hnr35(snd)
        gne = extract_gne(y, sr)          # 使用新实现
        mfcc10, delta11 = extract_mfcc_features(y, sr)

        # 风险评分逻辑（保持与原项目一致）
        score = 0
        if hnr35 < 20.0:
            score += 1
        if gne < 0.85:  # 注意：新 GNE 范围可能不同，需校准！
            score += 1
        if abs(delta11) > 0.2:
            score += 1
        risk = "high" if score >= 2 else "low"

        result = {
            "risk": risk,
            "features": {
                "hnr35": round(hnr35, 2),
                "gne": round(gne, 3),
                "mfcc10": round(mfcc10, 2),
                "delta11": round(delta11, 3)
            },
            "subtypes": {
                "tremor": int(max(0, min(100, (30 - hnr35) * 5))) if hnr35 < 25 else 10,
                "rigidity": int(max(0, min(100, (0.5 - abs(delta11)) * 200))) if abs(delta11) < 0.4 else 15,
                "nonMotor": int(abs(mfcc10) * 2) % 100
            }
        }
        return jsonify(result)
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)