import os
import glob
import logging
import numpy as np
import librosa
import parselmouth
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import tempfile
import pickle
from pydub import AudioSegment
import io
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
# 关键：导入praat call方法（和训练代码一致）
from parselmouth.praat import call

# ===================== 核心配置（自适应任意时长音频） =====================
CONFIG = {
    # 音频基础配置
    "SAMPLING_RATE": 16000,
    "N_FFT": 1024,  
    "HOP_LENGTH": 256,  
    "TEMP_DIR": tempfile.gettempdir(),
    # 模型/特征文件路径
    "MODEL_PATH": "./final_diagnosis_model.pkl",
    "SCALER_PATH": "./scaler.pkl",
    "FEATURE_ORDER_PATH": "./feature_order.pkl",
    # 亚型分类器路径
    "SUBTYPE_CLASSIFIER_PATH": "./subtype_classifier.pkl",
    # 诊断阈值 & 性别编码
    "PD_THRESHOLD": 0.5,
    "GENDER_MAP": {"0": 0, "1": 1, "unknown": 1},
    # Praat参数（和训练代码完全一致，动态适配时长）
    "PRAAT_PARAMS": {
        "f0min": 75.0,          # 训练时用的基频最小值
        "f0max": 500.0,         # 训练时用的基频最大值
        # 动态时间步长：短音频用小步长，长音频用标准步长
        "pitch_time_step_short": 0.005,  # 短音频（<5s）
        "pitch_time_step_normal": 0.01,  # 常规音频（5-60s）
        "pitch_time_step_long": 0.02,    # 长音频（>60s）
        # Jitter/Shimmer提取参数（和训练代码完全一致）
        "jitter_min_period": 0.0001,
        "jitter_max_period": 0.02,
        "jitter_factor": 1.3,
        "shimmer_amplitude_factor": 1.6
    },
    # 亚型特征权重（8维特征适配）
    "SUBTYPE_WEIGHTS": {
        "tremor": {
            "Shim_dB": -0.3, "Jitter_PPQ": 0.25, "MFCC4": 0.15,
            "Shim_loc": -0.2, "Jitter_rel": 0.2
        },
        "rigidity": {
            "MFCC4": 0.3, "Shi_APQ11": -0.25, "Shim_APQ5": -0.2,
            "Jitter_abs": 0.15
        },
        "motor": {
            "Shim_dB": -0.2, "Jitter_PPQ": 0.2, "MFCC4": 0.2,
            "Shim_loc": -0.15
        },
        "non_motor": {
            "Shi_APQ11": -0.3, "Shim_APQ5": -0.25, "Jitter_abs": 0.15,
            "Jitter_rel": 0.1
        }
    },
    # 亚型概率归一化参数
    "SUBTYPE_SMOOTH": 0.1,
    "SUBTYPE_SCALE": 1.2,
    # 音频时长配置（自适应）
    "MIN_AUDIO_DURATION": 0.5,  # 最小音频时长（秒）
    "SHORT_AUDIO_THRESHOLD": 5.0,  # 短音频阈值（<5s）
    "LONG_AUDIO_THRESHOLD": 60.0   # 长音频阈值（>60s）
}

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PDVoiceDiagnoser")

# ========== FFmpeg路径配置 ==========
AudioSegment.converter = "ffmpeg"
AudioSegment.ffprobe = "ffprobe"

# ===================== 数据类 =====================
@dataclass
class FeatureResult:
    acoustic_8d: np.ndarray
    gender: int
    key_features: Dict[str, float]
    feature_warnings: List[str]  # 新增：特征提取警告

@dataclass
class DiagnosisResult:
    pd_prob: float
    diagnosis: str
    risk: str
    features: Dict[str, float]
    subtype_probs: Dict[str, float]
    feature_warnings: List[str]

# ===================== 模型加载（单例模式） =====================
class ModelManager:
    _model = None
    _scaler = None
    _feature_order = None
    _subtype_classifier = None

    @classmethod
    def load_all(cls):
        """加载模型、标准化器、8维特征顺序"""
        if cls._feature_order is None:
            try:
                with open(CONFIG["FEATURE_ORDER_PATH"], 'rb') as f:
                    cls._feature_order = pickle.load(f)
                logger.info(f"加载8维特征顺序成功，共{len(cls._feature_order)}个特征：{cls._feature_order}")
            except Exception as e:
                logger.error(f"加载特征顺序失败：{e}")
                raise

        if cls._scaler is None:
            try:
                with open(CONFIG["SCALER_PATH"], 'rb') as f:
                    cls._scaler = pickle.load(f)
                logger.info("加载8维特征标准化器成功")
            except Exception as e:
                logger.error(f"加载标准化器失败：{e}")
                raise

        if cls._model is None:
            try:
                with open(CONFIG["MODEL_PATH"], 'rb') as f:
                    cls._model = pickle.load(f)
                logger.info("加载PD诊断模型成功")
            except Exception as e:
                logger.error(f"加载模型失败：{e}")
                raise

        if cls._subtype_classifier is None:
            try:
                with open(CONFIG["SUBTYPE_CLASSIFIER_PATH"], 'rb') as f:
                    cls._subtype_classifier = pickle.load(f)
                logger.info("加载亚型预训练分类器成功")
            except FileNotFoundError:
                logger.warning("未找到预训练亚型分类器，将使用基于文献的规则分类法")
                cls._subtype_classifier = "rule_based"
            except Exception as e:
                logger.error(f"加载亚型分类器失败：{e}，使用规则分类法")
                cls._subtype_classifier = "rule_based"

        return cls._model, cls._scaler, cls._feature_order, cls._subtype_classifier

# ===================== 工具函数 =====================
def clean_temp_files(pattern: str = "*.wav") -> None:
    """清理临时音频文件"""
    try:
        for file in glob.glob(os.path.join(CONFIG["TEMP_DIR"], pattern)):
            os.remove(file)
            logger.debug(f"删除临时文件：{file}")
    except Exception as e:
        logger.warning(f"清理临时文件失败：{e}")

def get_temp_path(prefix: str) -> str:
    """生成唯一临时文件路径"""
    return os.path.join(CONFIG["TEMP_DIR"], f"{prefix}_{os.getpid()}_{np.random.randint(10000)}.wav")

def convert_numpy_type(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_type(i) for i in obj]
    else:
        return obj

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def get_adaptive_pitch_step(duration: float) -> float:
    """根据音频时长动态选择Pitch时间步长"""
    if duration < CONFIG["SHORT_AUDIO_THRESHOLD"]:
        return CONFIG["PRAAT_PARAMS"]["pitch_time_step_short"]
    elif duration > CONFIG["LONG_AUDIO_THRESHOLD"]:
        return CONFIG["PRAAT_PARAMS"]["pitch_time_step_long"]
    else:
        return CONFIG["PRAAT_PARAMS"]["pitch_time_step_normal"]

# ===================== 8维声学特征提取（复用训练代码的Praat逻辑） =====================
def extract_jitter_shimmer_praat(sound: parselmouth.Sound, duration: float) -> Tuple[Dict[str, float], List[str]]:
    """
    完全复用训练数据时的提取方法：Praat PointProcess方式提取Jitter/Shimmer
    自适应音频时长，动态调整Pitch参数
    """
    # 初始化特征（和训练/诊断的特征名对齐）
    jitter_shimmer_feats = {
        # Jitter特征
        "Jitter_rel": 0.0,
        "Jitter_abs": 0.0,
        "Jitter_PPQ": 0.0,
        # Shimmer特征
        "Shim_loc": 0.0,
        "Shim_dB": 0.0,
        "Shim_APQ5": 0.0,
        "Shi_APQ11": 0.0
    }
    warnings = []
    
    try:
        # 1. 动态选择Pitch时间步长（核心自适应逻辑）
        pitch_time_step = get_adaptive_pitch_step(duration)
        
        # 2. 复用训练代码：创建Pitch和PointProcess
        pitch = call(sound, "To Pitch", pitch_time_step, 
                     CONFIG["PRAAT_PARAMS"]["f0min"], CONFIG["PRAAT_PARAMS"]["f0max"])
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 
                            CONFIG["PRAAT_PARAMS"]["f0min"], CONFIG["PRAAT_PARAMS"]["f0max"])
        
        # 3. 提取Jitter特征（和训练代码完全一致）
        jitter_shimmer_feats["Jitter_rel"] = float(call(
            pointProcess, "Get jitter (local)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"]
        ))
        jitter_shimmer_feats["Jitter_abs"] = float(call(
            pointProcess, "Get jitter (local, absolute)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"]
        ))
        jitter_shimmer_feats["Jitter_PPQ"] = float(call(
            pointProcess, "Get jitter (ppq5)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"]
        ))
        
        # 4. 提取Shimmer特征（和训练代码完全一致）
        jitter_shimmer_feats["Shim_loc"] = float(call(
            [sound, pointProcess], "Get shimmer (local)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"],
            CONFIG["PRAAT_PARAMS"]["shimmer_amplitude_factor"]
        ))
        jitter_shimmer_feats["Shim_dB"] = float(call(
            [sound, pointProcess], "Get shimmer (local_dB)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"],
            CONFIG["PRAAT_PARAMS"]["shimmer_amplitude_factor"]
        ))
        jitter_shimmer_feats["Shim_APQ5"] = float(call(
            [sound, pointProcess], "Get shimmer (apq5)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"],
            CONFIG["PRAAT_PARAMS"]["shimmer_amplitude_factor"]
        ))
        jitter_shimmer_feats["Shi_APQ11"] = float(call(
            [sound, pointProcess], "Get shimmer (apq11)", 0, 0,
            CONFIG["PRAAT_PARAMS"]["jitter_min_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_max_period"],
            CONFIG["PRAAT_PARAMS"]["jitter_factor"],
            CONFIG["PRAAT_PARAMS"]["shimmer_amplitude_factor"]
        ))
        
        # 5. 自适应异常值处理：仅对短音频过滤异常值
        if duration < CONFIG["SHORT_AUDIO_THRESHOLD"]:
            for feat_name, val in jitter_shimmer_feats.items():
                if np.isnan(val) or val < 0 or val > 1:
                    jitter_shimmer_feats[feat_name] = 0.0
                    warnings.append(f"短音频特征异常：{feat_name}={val}，已重置为0")
        
        logger.info(
            f"Jitter/Shimmer提取成功（时长{duration:.2f}s，Pitch步长{pitch_time_step}）："
            f"Jitter_PPQ={jitter_shimmer_feats['Jitter_PPQ']:.6f}, "
            f"Shi_APQ11={jitter_shimmer_feats['Shi_APQ11']:.6f}"
        )
        
    except Exception as e:
        warning_msg = f"Jitter/Shimmer提取失败（Praat标准逻辑）：{str(e)}"
        logger.warning(warning_msg)
        warnings.append(warning_msg)
    
    return jitter_shimmer_feats, warnings

def extract_mfcc4(y: np.ndarray, sr: int, duration: float) -> Tuple[float, List[str]]:
    """提取MFCC4特征（自适应音频时长）"""
    warnings = []
    try:
        # 动态调整MFCC提取参数：短音频用更小的窗口，长音频用标准窗口
        if duration < CONFIG["SHORT_AUDIO_THRESHOLD"]:
            n_fft = 512
            hop_length = 128
        elif duration > CONFIG["LONG_AUDIO_THRESHOLD"]:
            n_fft = 2048
            hop_length = 512
        else:
            n_fft = CONFIG["N_FFT"]
            hop_length = CONFIG["HOP_LENGTH"]
        
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13,
            n_fft=n_fft, hop_length=hop_length,
            center=True, pad_mode='reflect'
        )
        mfcc4 = float(np.mean(mfccs[4]))
        if np.isnan(mfcc4):
            mfcc4 = 0.0
            warnings.append("MFCC4特征提取失败：计算结果为NaN")
        logger.info(f"MFCC4提取成功（时长{duration:.2f}s，n_fft={n_fft}）：{mfcc4:.6f}")
        return mfcc4, warnings
    except Exception as e:
        warning_msg = f"MFCC4特征提取失败：{str(e)}"
        logger.warning(warning_msg)
        warnings.append(warning_msg)
        return 0.0, warnings

def extract_all_features(sound: parselmouth.Sound, y: np.ndarray, sr: int, gender: str, duration: float) -> FeatureResult:
    """提取8维声学特征 + 性别编码（自适应音频时长）"""
    # 1. 提取Jitter/Shimmer（自适应时长）
    jitter_shimmer_feats, js_warnings = extract_jitter_shimmer_praat(sound, duration)
    
    # 2. 提取MFCC4（自适应时长）
    mfcc4, mfcc4_warnings = extract_mfcc4(y, sr, duration)
    
    # 3. 合并特征和警告
    all_feats = {**jitter_shimmer_feats, "MFCC4": mfcc4}
    all_warnings = js_warnings + mfcc4_warnings
    
    # 4. 按feature_order顺序拼接8维数组（原有逻辑不变）
    model, scaler, feature_order, subtype_clf = ModelManager.load_all()
    acoustic_8d = []
    key_features = {}
    for feat_name in feature_order:
        val = all_feats.get(feat_name, 0.0)
        acoustic_8d.append(float(val))
        # 记录关键亚型特征
        if feat_name in CONFIG["SUBTYPE_WEIGHTS"]["tremor"] or feat_name in CONFIG["SUBTYPE_WEIGHTS"]["rigidity"]:
            key_features[feat_name] = val

    acoustic_8d = np.array(acoustic_8d, dtype=np.float64)

    # 5. 性别编码（原有逻辑不变）
    gender_code = CONFIG["GENDER_MAP"].get(gender, CONFIG["GENDER_MAP"]["unknown"])
    
    return FeatureResult(
        acoustic_8d=acoustic_8d,
        gender=gender_code,
        key_features=key_features,
        feature_warnings=all_warnings
    )

# ===================== PD亚型概率计算 =====================
def calculate_subtype_probs(feature_result: FeatureResult, pd_prob: float) -> Dict[str, float]:
    """计算PD各亚型概率（适配8维特征）"""
    key_features = feature_result.key_features
    weights = CONFIG["SUBTYPE_WEIGHTS"]
    subtype_scores = {}

    # 计算各亚型原始得分
    for subtype, feat_weights in weights.items():
        score = 0.0
        for feat_name, weight in feat_weights.items():
            score += key_features.get(feat_name, 0.0) * weight
        subtype_scores[subtype] = score

    # 结合PD概率调整得分
    adjusted_scores = {}
    for subtype, score in subtype_scores.items():
        adjusted_scores[subtype] = score * pd_prob * CONFIG["SUBTYPE_SCALE"] + CONFIG["SUBTYPE_SMOOTH"]

    # Sigmoid转换
    sigmoid_scores = {subtype: sigmoid(score) for subtype, score in adjusted_scores.items()}

    # 归一化
    total = sum(sigmoid_scores.values())
    normalized_probs = {
        subtype: round(prob / total, 4) 
        for subtype, prob in sigmoid_scores.items()
    }

    # 低PD概率时降低亚型概率
    if pd_prob < CONFIG["PD_THRESHOLD"]:
        for subtype in normalized_probs:
            normalized_probs[subtype] = round(normalized_probs[subtype] * pd_prob, 4)

    logger.info(f"亚型概率计算完成：{normalized_probs}")
    return normalized_probs

# ===================== 诊断核心逻辑 =====================
def diagnose(feature_result: FeatureResult) -> DiagnosisResult:
    """使用预训练模型诊断PD（自适应音频时长）"""
    model, scaler, feature_order, subtype_clf = ModelManager.load_all()
    
    # 1. 标准化8维声学特征
    acoustic_scaled = scaler.transform(feature_result.acoustic_8d.reshape(1, -1))
    # 2. 拼接性别特征（8+1=9维）
    input_9d = np.hstack([acoustic_scaled, np.array([[feature_result.gender]], dtype=np.float64)])
    
    # 适配sklearn版本
    try:
        pd_prob = float(model.predict_proba(input_9d)[0, 1])
    except AttributeError:
        logger.warning("sklearn版本兼容问题，改用decision_function计算PD概率")
        decision_score = model.decision_function(input_9d)[0]
        pd_prob = float(1 / (1 + np.exp(-decision_score)))
    
    # 诊断结论和风险等级
    diagnosis = "患有PD" if pd_prob >= CONFIG["PD_THRESHOLD"] else "健康"
    if pd_prob >= 0.8:
        risk = "高风险"
    elif pd_prob >= 0.5:
        risk = "中风险"
    else:
        risk = "低风险"
    
    # 构造特征字典
    feat_dict = {}
    for name, val in zip(feature_order, feature_result.acoustic_8d):
        feat_dict[name] = round(float(val), 4)
    feat_dict["gender"] = int(feature_result.gender)

    # 计算亚型概率
    subtype_probs = calculate_subtype_probs(feature_result, pd_prob)

    return DiagnosisResult(
        pd_prob=round(pd_prob, 4),
        diagnosis=diagnosis,
        risk=risk,
        features=feat_dict,
        subtype_probs=subtype_probs,
        feature_warnings=feature_result.feature_warnings
    )

# ===================== Flask API服务 =====================
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check() -> Tuple[str, int]:
    """健康检查接口"""
    return "PD Voice Diagnosis Service is Running (Adaptive Audio Duration Support)", 200

@app.route('/analyze', methods=['POST'])
@app.route('/diagnose', methods=['POST'])
def pd_diagnose() -> Tuple[Any, int]:
    """核心诊断接口：自适应处理任意时长音频（3s~180s+）"""
    clean_temp_files()
    temp_paths = []
    temp_path = ""
    converted_path = ""
    try:
        # 1. 校验请求参数
        if 'file' not in request.files:
            logger.warning("请求无音频文件")
            return jsonify({"code":400, "msg":"未上传音频文件", "data":None}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("未选择音频文件")
            return jsonify({"code":400, "msg":"未选择音频文件", "data":None}), 400
        
        # 获取性别参数
        gender = request.form.get("gender", "unknown")
        if gender not in ["0", "1", "unknown"]:
            logger.warning(f"非法性别参数：{gender}，使用默认值unknown")
            gender = "unknown"
        logger.info(f"接收诊断请求：文件={file.filename}，性别={gender}，后端编码={CONFIG['GENDER_MAP'][gender]}")

        # 2. 保存并转换音频
        temp_path = get_temp_path("upload_audio")
        converted_path = get_temp_path("converted_audio")
        temp_paths = [temp_path, converted_path]
        
        # 保存原始文件
        file.save(temp_path)
        logger.info(f"原始文件已保存到：{temp_path}，大小：{os.path.getsize(temp_path)}字节")

        # 转换为标准WAV（16kHz、单声道、16位）
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", temp_path,
            "-ar", str(CONFIG["SAMPLING_RATE"]),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            "-y",
            converted_path
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg转换失败：{result.stderr}")
            raise Exception(f"音频格式转换失败：{result.stderr}")
        
        # 读取转换后的音频
        import wave
        with wave.open(converted_path, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(-1)
        
        # 转换为numpy数组并归一化
        y = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        y = y / np.iinfo(np.int16).max
        
        # 3. 自适应音频处理：移除固定3秒补零，保留最小时长校验
        current_duration = len(y) / sr
        logger.info(f"原始音频时长：{current_duration:.2f}秒")
        
        # 校验最小时长
        if current_duration < CONFIG["MIN_AUDIO_DURATION"]:
            raise Exception(f"音频过短（{current_duration:.2f}秒），请上传至少{CONFIG['MIN_AUDIO_DURATION']}秒的语音音频")
        
        # 不再强制补零/截断！保留原始时长
        final_y = y
        final_duration = current_duration
        
        # 生成parselmouth Sound对象
        sound = parselmouth.Sound(final_y, sampling_frequency=CONFIG["SAMPLING_RATE"])
        logger.info(f"处理后音频时长：{final_duration:.2f}秒（原始时长，未做截断/补零）")

        # 检测音频有效性（是否全为静音）
        if np.max(np.abs(final_y)) < 1e-6:
            raise Exception("上传的音频文件无有效语音数据（全为静音/噪音），请重新上传包含清晰语音的音频")

        # 4. 提取8维特征 + 诊断（传入实际时长用于自适应）
        features = extract_all_features(sound, final_y, CONFIG["SAMPLING_RATE"], gender, final_duration)
        result = diagnose(features)
        logger.info(f"诊断完成：PD概率={result.pd_prob}，结论={result.diagnosis}，亚型概率={result.subtype_probs}")

        # 5. 构造返回结果
        msg = "诊断成功" if len(result.feature_warnings) == 0 else "诊断成功（部分特征提取失败）"
        response = convert_numpy_type({
            "code": 200,
            "msg": msg,
            "data": {
                "audio_duration": round(final_duration, 2),  # 返回实际音频时长
                "pd_probability": result.pd_prob,
                "diagnosis": result.diagnosis,
                "risk_level": result.risk,
                "acoustic_features": result.features,
                "feature_warnings": result.feature_warnings,
                "subtype_probabilities": {
                    "tremor_type": result.subtype_probs["tremor"],
                    "rigidity_type": result.subtype_probs["rigidity"],
                    "motor_type": result.subtype_probs["motor"],
                    "non_motor_type": result.subtype_probs["non_motor"]
                },
                "subtype_explanation": {
                    "tremor_type": "震颤主导型：以静止性震颤为核心，Shim_dB偏低、Jitter_PPQ偏高",
                    "rigidity_type": "僵直主导型：以肌肉僵直为核心，MFCC4偏高、Shi_APQ11偏低",
                    "motor_type": "运动型：包含震颤和僵直特征，运动症状显著",
                    "non_motor_type": "非运动型：以非运动症状为核心，Shi_APQ11偏低"
                }
            }
        })
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"诊断异常：{str(e)}", exc_info=True)
        error_response = convert_numpy_type({
            "code": 500,
            "msg": f"诊断失败：{str(e)}",
            "data": None
        })
        return jsonify(error_response), 500

    finally:
        # 清理临时文件
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"已清理临时文件：{path}")
                except Exception as e:
                    logger.warning(f"清理临时文件{path}失败：{e}")

# ===================== 启动服务 =====================
if __name__ == '__main__':
    # 预加载模型
    try:
        ModelManager.load_all()
    except Exception as e:
        logger.error(f"服务启动失败：{e}")
        exit(1)
    # 适配端口
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)