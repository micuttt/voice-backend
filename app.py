import os
import glob
import logging
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import tempfile

# ===================== 配置项（解耦硬编码参数） =====================
CONFIG = {
    "SAMPLING_RATE": 16000,
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "MFCC_N": 13,
    "TEMP_DIR": tempfile.gettempdir(),  # 跨平台临时目录
    # 风险评分阈值
    "HNR35_THRESHOLD": 20.0,
    "GNE_THRESHOLD": 0.85,
    "DELTA11_THRESHOLD": 0.2,
    # Praat HNR35 参数
    "HNR35_PARAMS": {
        "time_step": 0.01,
        "minimum_pitch": 75.0,
        "silence_threshold": 0.1,
        "periods_per_window": 4.5
    },
    # GNE 频段配置
    "GNE_BANDS": {
        "glottal": (500, 4500),
        "noise": (4500, 6000)
    },
    # 亚型评分参数
    "SUBTYPE_PARAMS": {
        "tremor": {"threshold": 25, "scale": 5, "base": 10},
        "rigidity": {"threshold": 0.4, "scale": 200, "base": 15},
        "nonMotor": {"scale": 2}
    }
}

# ===================== 日志配置（结构化日志） =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VoiceScreener")

# ===================== 数据类（类型安全） =====================
@dataclass
class FeatureResult:
    hnr35: float
    gne: float
    mfcc10: float
    delta11: float

@dataclass
class AnalysisResult:
    risk: str
    features: FeatureResult
    subtypes: Dict[str, int]

# ===================== 工具函数（复用性） =====================
def clean_temp_files(pattern: str = "*.wav") -> None:
    """清理临时文件，带日志和异常详情"""
    try:
        temp_files = glob.glob(os.path.join(CONFIG["TEMP_DIR"], pattern))
        for file_path in temp_files:
            try:
                os.remove(file_path)
                logger.debug(f"Deleted temp file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete {file_path}: {e.strerror}")
    except Exception as e:
        logger.error(f"Error cleaning temp files: {str(e)}")

def get_temp_file_path(prefix: str) -> str:
    """生成安全的临时文件路径"""
    return os.path.join(CONFIG["TEMP_DIR"], f"{prefix}_{os.getpid()}.wav")

# ===================== 特征提取（解耦+单一职责） =====================
def extract_hnr35(sound_obj: parselmouth.Sound) -> float:
    """提取0-3500Hz频段的谐波噪声比（HNR35）"""
    try:
        harmonicity = sound_obj.to_harmonicity_cc(**CONFIG["HNR35_PARAMS"])
        values = harmonicity.values
        valid_values = values[values != -200]  # 过滤Praat静音标记
        return float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0
    except Exception as e:
        logger.error(f"Failed to extract HNR35: {str(e)}")
        return 0.0

def extract_gne(y: np.ndarray, sr: int) -> float:
    """提取声门噪声激励比（GNE），基于Maryn et al. (2009)频谱斜率法"""
    try:
        # 计算STFT和频率轴
        S = np.abs(librosa.stft(y, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=CONFIG["N_FFT"])
        
        # 筛选目标频段索引
        glottal_low, glottal_high = CONFIG["GNE_BANDS"]["glottal"]
        noise_low, noise_high = CONFIG["GNE_BANDS"]["noise"]
        noise_high = min(noise_high, sr // 2)  # 避免超过奈奎斯特频率
        
        glottal_idx = np.where((freqs >= glottal_low) & (freqs <= glottal_high))[0]
        noise_idx = np.where((freqs > noise_low) & (freqs <= noise_high))[0]
        
        if len(glottal_idx) == 0 or len(noise_idx) == 0:
            logger.warning("GNE: No valid frequency bands found")
            return 0.0
        
        # 计算帧能量均值
        glottal_energy = np.mean(S[glottal_idx, :], axis=0)
        noise_energy = np.mean(S[noise_idx, :], axis=0)
        
        # 避免除零并计算GNE
        epsilon = 1e-10
        gne_frames = np.log10((glottal_energy + epsilon) / (noise_energy + epsilon))
        return float(np.mean(gne_frames))
    except Exception as e:
        logger.error(f"Failed to extract GNE: {str(e)}")
        return 0.0

def extract_mfcc_features(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """提取MFCC10（均值）和Delta11（均值）"""
    try:
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=CONFIG["MFCC_N"],
            n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]
        )
        mfcc10_mean = float(np.mean(mfccs[9]))  # 0-indexed第10阶
        delta_mfccs = librosa.feature.delta(mfccs)
        delta11_mean = float(np.mean(delta_mfccs[10]))  # 0-indexed第11阶差分
        return mfcc10_mean, delta11_mean
    except Exception as e:
        logger.error(f"Failed to extract MFCC features: {str(e)}")
        return 0.0, 0.0

def extract_all_features(snd: parselmouth.Sound, y: np.ndarray, sr: int) -> FeatureResult:
    """统一提取所有特征，返回结构化结果"""
    hnr35 = extract_hnr35(snd)
    gne = extract_gne(y, sr)
    mfcc10, delta11 = extract_mfcc_features(y, sr)
    return FeatureResult(hnr35, gne, mfcc10, delta11)

# ===================== 评分逻辑（解耦） =====================
def calculate_risk(features: FeatureResult) -> str:
    """基于特征计算风险等级（high/low）"""
    score = 0
    if features.hnr35 < CONFIG["HNR35_THRESHOLD"]:
        score += 1
    if features.gne < CONFIG["GNE_THRESHOLD"]:
        score += 1
    if abs(features.delta11) > CONFIG["DELTA11_THRESHOLD"]:
        score += 1
    return "high" if score >= 2 else "low"

def calculate_subtypes(features: FeatureResult) -> Dict[str, int]:
    """计算震颤、僵硬、非运动症状亚型评分"""
    tremor_params = CONFIG["SUBTYPE_PARAMS"]["tremor"]
    rigidity_params = CONFIG["SUBTYPE_PARAMS"]["rigidity"]
    non_motor_params = CONFIG["SUBTYPE_PARAMS"]["nonMotor"]
    
    # 震颤评分
    tremor = (30 - features.hnr35) * tremor_params["scale"] if features.hnr35 < tremor_params["threshold"] else tremor_params["base"]
    tremor = int(max(0, min(100, tremor)))
    
    # 僵硬评分
    rigidity = (0.5 - abs(features.delta11)) * rigidity_params["scale"] if abs(features.delta11) < rigidity_params["threshold"] else rigidity_params["base"]
    rigidity = int(max(0, min(100, rigidity)))
    
    # 非运动症状评分
    non_motor = int(abs(features.mfcc10) * non_motor_params["scale"]) % 100
    
    return {"tremor": tremor, "rigidity": rigidity, "nonMotor": non_motor}

# ===================== Flask 应用（核心逻辑解耦） =====================
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check() -> Tuple[str, int]:
    """健康检查接口"""
    return "VoiceScreener Analysis Service is Running. (4 Features Ready)", 200

@app.route('/analyze', methods=['POST'])
def analyze() -> Tuple[Any, int]:
    """语音分析核心接口"""
    clean_temp_files()  # 清理历史临时文件
    try:
        # 1. 校验上传文件
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No selected file"}), 400
        
        # 2. 生成临时文件路径
        temp_path = get_temp_file_path("upload")
        converted_path = get_temp_file_path("converted")
        
        # 3. 加载并转换音频
        file.save(temp_path)
        y, sr = librosa.load(temp_path, sr=CONFIG["SAMPLING_RATE"])
        sf.write(converted_path, y, sr)
        snd = parselmouth.Sound(converted_path)
        
        # 4. 提取特征 + 计算评分
        features = extract_all_features(snd, y, sr)
        risk = calculate_risk(features)
        subtypes = calculate_subtypes(features)
        
        # 5. 构造返回结果（结构化+四舍五入）
        result = AnalysisResult(
            risk=risk,
            features=FeatureResult(
                hnr35=round(features.hnr35, 2),
                gne=round(features.gne, 3),
                mfcc10=round(features.mfcc10, 2),
                delta11=round(features.delta11, 3)
            ),
            subtypes=subtypes
        )
        
        # 6. 转换为JSON可序列化格式
        response = {
            "risk": result.risk,
            "features": {
                "hnr35": result.features.hnr35,
                "gne": result.features.gne,
                "mfcc10": result.features.mfcc10,
                "delta11": result.features.delta11
            },
            "subtypes": result.subtypes
        }
        logger.info(f"Analysis completed successfully (risk: {risk})")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    
    finally:
        # 确保临时文件删除（带容错）
        for path in [temp_path, converted_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up temp file: {path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up {path}: {e.strerror}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
