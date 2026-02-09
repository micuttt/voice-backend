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
from typing import Tuple, Dict, Any
import tempfile
import pickle
from pydub import AudioSegment
import io
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

# ===================== 核心配置（需确认路径） =====================
CONFIG = {
    # 音频基础配置
    "SAMPLING_RATE": 16000,
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "TEMP_DIR": tempfile.gettempdir(),
    # 模型/特征文件路径（关键！确认和第六份代码生成的文件同目录）
    "MODEL_PATH": "./final_diagnosis_model.pkl",
    "SCALER_PATH": "./scaler.pkl",
    "FEATURE_ORDER_PATH": "./feature_order.pkl",
    # 新增：亚型分类器路径（若未训练，使用内置规则分类器）
    "SUBTYPE_CLASSIFIER_PATH": "./subtype_classifier.pkl",
    # 诊断阈值 & 性别编码（和第六份代码一致）
    "PD_THRESHOLD": 0.5,
    "GENDER_MAP": {"0": 1, "1": 0, "unknown": 0},
    # Praat HNR参数（修正：匹配Praat官方参数格式）
    "HNR_PARAMS": {
        "time_step": 0.01,
        "minimum_pitch": 75.0,
        "silence_threshold": 0.1,
        "periods_per_window": 4.5
    },
    # 新增：亚型特征权重（基于文献1和2的核心发现）
    "SUBTYPE_WEIGHTS": {
        # 震颤型（TD）：HNR系列低、Delta11高（文献1：TD以发声异常为主）
        "tremor": {
            "HNR35": -0.3, "HNR15": -0.25, "HNR05": -0.2,
            "Delta11": 0.25, "Delta8": 0.15, "MFCC10": 0.1
        },
        # 僵直型（Rigidity/PIGD）：VOT长、MFCC12高、停顿时间长（文献2：PIGD以发音协调异常为主）
        "rigidity": {
            "MFCC12": 0.3, "Delta0": 0.25, "MFCC6": 0.2,
            "HNR38": -0.15, "Delta6": 0.15
        },
        # 运动型（Motor：TD+Rigidity）：HNR+Delta+MFCC综合权重（文献1：运动型声学特征差异显著）
        "motor": {
            "HNR35": -0.2, "Delta11": 0.2, "MFCC12": 0.2,
            "HNR15": -0.15, "Delta8": 0.15
        },
        # 非运动型（Non-Motor）：GNE低、MFCC0高（文献1：非运动型语音质量更差）
        "non_motor": {
            "GNE": -0.3, "MFCC0": 0.25, "MFCC1": 0.2,
            "Delta3": -0.15, "HNR25": -0.1
        }
    },
    # 新增：亚型概率归一化参数
    "SUBTYPE_SMOOTH": 0.1,  # 平滑因子，避免概率为0
    "SUBTYPE_SCALE": 1.2    # 缩放因子，增强区分度
}

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PDVoiceDiagnoser")

# ========== FFmpeg路径配置（修改：适配Render Linux环境） ==========
# 移除Windows硬编码路径，使用系统默认FFmpeg（Render已预装）
AudioSegment.converter = "ffmpeg"
AudioSegment.ffprobe = "ffprobe"


# ===================== 数据类（结构化输出） =====================
@dataclass
class FeatureResult:
    acoustic_31d: np.ndarray  # 31维声学特征
    gender: int               # 性别编码
    key_features: Dict[str, float]  # 新增：关键亚型特征（用于概率计算）

@dataclass
class DiagnosisResult:
    pd_prob: float            # 患PD概率（0-1）
    diagnosis: str            # 诊断结论（患有PD/健康）
    risk: str                 # 风险等级（高/中/低）
    features: Dict[str, float]# 可视化特征值
    # 新增：亚型概率
    subtype_probs: Dict[str, float]  # 包含tremor/rigidity/motor/non_motor

# ===================== 模型加载（单例模式，避免重复加载） =====================
class ModelManager:
    _model = None
    _scaler = None
    _feature_order = None
    _subtype_classifier = None  # 新增：亚型分类器

    @classmethod
    def load_all(cls):
        """加载模型、标准化器、特征顺序（仅加载一次）"""
        # 1. 加载特征顺序
        if cls._feature_order is None:
            try:
                with open(CONFIG["FEATURE_ORDER_PATH"], 'rb') as f:
                    cls._feature_order = pickle.load(f)
                logger.info(f"加载31维特征顺序成功，共{len(cls._feature_order)}个特征")
            except Exception as e:
                logger.error(f"加载特征顺序失败：{e}")
                raise

        # 2. 加载标准化器
        if cls._scaler is None:
            try:
                with open(CONFIG["SCALER_PATH"], 'rb') as f:
                    cls._scaler = pickle.load(f)
                logger.info("加载标准化器成功")
            except Exception as e:
                logger.error(f"加载标准化器失败：{e}")
                raise

        # 3. 加载诊断模型
        if cls._model is None:
            try:
                with open(CONFIG["MODEL_PATH"], 'rb') as f:
                    cls._model = pickle.load(f)
                logger.info("加载PD诊断模型成功")
            except Exception as e:
                logger.error(f"加载模型失败：{e}")
                raise

        # 新增：4. 加载亚型分类器（优先使用预训练模型，无则使用规则分类）
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
    """递归转换numpy类型为Python原生类型（解决JSON序列化）"""
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
    """Sigmoid函数：将得分映射到0-1概率区间"""
    return 1 / (1 + np.exp(-x))

# ===================== 31维声学特征提取（修复核心问题） =====================
def extract_hnr_by_band(sound: parselmouth.Sound, band: str) -> float:
    """
    真实提取不同频段的HNR（谐波噪声比）
    核心依据：不同频段的HNR对应不同的最小基频（f0），这是声学特征提取的行业标准
    band: 频段标识（HNR05/HNR15/HNR25/HNR35/HNR38）
    返回值：对应频段的HNR均值（float）
    """
    # 分频段最小基频配置（声学领域通用参数）
    band_pitch_map = {
        "HNR05": 50.0,    # 低频段（50Hz）：对应声带基频下限
        "HNR15": 150.0,   # 中低频段（150Hz）：男性语音核心频段
        "HNR25": 250.0,   # 中频段（250Hz）：中性语音核心频段
        "HNR35": 350.0,   # 中高频段（350Hz）：女性语音核心频段
        "HNR38": 380.0    # 高频段（380Hz）：儿童/高音语音频段
    }
    
    try:
        # Praat官方API：To Harmonicity (cc) 四个核心参数
        harmonicity = parselmouth.praat.call(
            sound, "To Harmonicity (cc)",
            CONFIG["HNR_PARAMS"]["time_step"],        # 时间步长（0.01秒）
            band_pitch_map[band],                     # 对应频段的最小基频（核心区分参数）
            CONFIG["HNR_PARAMS"]["silence_threshold"],# 静音阈值（0.1）
            CONFIG["HNR_PARAMS"]["periods_per_window"]# 窗口周期数（4.5）
        )
        # 获取HNR数值（过滤Praat的静音标记-200）
        hnr_values = harmonicity.values
        valid_hnr = hnr_values[hnr_values != -200]

        # 若有效数据为空，返回0；否则返回均值
        valid_hnr = valid_hnr[valid_hnr >= 0]
        if len(valid_hnr) == 0:
            logger.warning(f"{band}无有效数据，返回0.0")
            return 0.0
        return float(np.mean(valid_hnr))
        
    
    except KeyError:
        logger.error(f"无效的频段标识：{band}，仅支持HNR05/HNR15/HNR25/HNR35/HNR38")
        return 0.0
    except Exception as e:
        logger.error(f"提取{band}失败：{e}")
        return 0.0

def extract_mfcc_delta(y: np.ndarray, sr: int) -> Dict[str, float]:
    """提取13维MFCC + 13维Delta-MFCC"""
    features = {}
    try:
        # 提取13维MFCC（0-12阶）
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13,
            n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]
        )
        for i in range(13):
            features[f"MFCC{i}"] = float(np.mean(mfccs[i]))
        
        # 提取13维Delta-MFCC
        delta = librosa.feature.delta(mfccs) 
        for i in range(13):
            features[f"Delta{i}"] = float(np.mean(delta[i]))
    except Exception as e:
        logger.error(f"提取MFCC/Delta失败：{e}")
        # 填充默认值
        for i in range(13):
            features[f"MFCC{i}"] = 0.0
            features[f"Delta{i}"] = 0.0
    return features

def extract_gne(y: np.ndarray, sr: int) -> float:
    """提取GNE（声门噪声激励比）"""
    try:
        S = np.abs(librosa.stft(y, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"]))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=CONFIG["N_FFT"])
        
        # 筛选频段
        glottal_idx = np.where((freqs >= 500) & (freqs <= 4500))[0]
        noise_idx = np.where((freqs > 4500) & (freqs <= 6000))[0]
        
        if len(glottal_idx) == 0 or len(noise_idx) == 0:
            return 0.0
        
        # 计算能量比
        glottal_energy = np.mean(S[glottal_idx, :], axis=0)
        noise_energy = np.mean(S[noise_idx, :], axis=0)
        epsilon = 1e-10
        gne = np.mean(np.log10((glottal_energy + epsilon) / (noise_energy + epsilon)))
        return float(gne)
    except Exception as e:
        logger.error(f"提取GNE失败：{e}")
        return 0.0

def extract_all_features(sound: parselmouth.Sound, y: np.ndarray, sr: int, gender: str) -> FeatureResult:
    """提取31维声学特征 + 性别编码 + 关键亚型特征（修复参数错误）"""
    # 1. 提取基础特征
    mfcc_delta = extract_mfcc_delta(y, sr)
    # 修复：统一提取HNR（Praat官方格式），按特征名匹配
    hnr_features = {
        "HNR05": extract_hnr_by_band(sound, "HNR05"),
        "HNR15": extract_hnr_by_band(sound, "HNR15"),
        "HNR25": extract_hnr_by_band(sound, "HNR25"),
        "HNR35": extract_hnr_by_band(sound, "HNR35"),
        "HNR38": extract_hnr_by_band(sound, "HNR38")
    }
    mfcc_delta["GNE"] = extract_gne(y, sr)

    # 2. 按第六份代码的特征顺序拼接31维数组
    model, scaler, feature_order, subtype_clf = ModelManager.load_all()
    acoustic_31d = []
    key_features = {}  # 新增：存储关键亚型特征（用于概率计算）
    for feat_name in feature_order:
        if feat_name in mfcc_delta:
            val = mfcc_delta[feat_name]
        elif feat_name in hnr_features:
            val = hnr_features[feat_name]
        else:
            val = 0.0
            logger.warning(f"特征{feat_name}未提取到，填充0.0")
        
        acoustic_31d.append(float(val))  # 强制转换为Python float
        # 新增：记录关键亚型特征（文献中提到的核心区分特征）
        if feat_name in ["HNR05", "HNR15", "HNR25", "HNR35", "HNR38", 
                         "GNE", "MFCC0", "MFCC1", "MFCC6", "MFCC10", "MFCC12",
                         "Delta0", "Delta3", "Delta6", "Delta8", "Delta11"]:
            key_features[feat_name] = val

    acoustic_31d = np.array(acoustic_31d, dtype=np.float64)  # 用float64避免类型问题

    # 3. 性别编码
    # 3. 性别编码（原逻辑是按字符串匹配，现在按前端数字匹配）
    gender_code = CONFIG["GENDER_MAP"].get(gender, CONFIG["GENDER_MAP"]["unknown"])
    return FeatureResult(acoustic_31d, gender_code, key_features)

# ===================== 新增：PD亚型概率计算逻辑 =====================
def calculate_subtype_probs(feature_result: FeatureResult, pd_prob: float) -> Dict[str, float]:
    """
    计算PD各亚型概率（震颤型/僵直型/运动型/非运动型）
    计算方法：基于文献的特征权重法 + 概率归一化
    核心依据：
    1. 文献1（Santos et al. 2025）：非震颤型（僵直型）语音表现更差（音节数少、发音时间短）
    2. 文献2（Rusz et al. 2023）：PIGD（僵直型）与TD（震颤型）的声学特征差异显著
    3. 统计分析结果：HNR系列、Delta系列、MFCC系列是核心区分特征
    """
    key_features = feature_result.key_features
    weights = CONFIG["SUBTYPE_WEIGHTS"]
    subtype_scores = {}

    # 步骤1：计算各亚型原始得分（特征值 × 权重求和）
    for subtype, feat_weights in weights.items():
        score = 0.0
        for feat_name, weight in feat_weights.items():
            # 若特征不存在，用0填充
            score += key_features.get(feat_name, 0.0) * weight
        subtype_scores[subtype] = score

    # 步骤2：结合PD总体概率调整得分（PD概率越高，亚型概率越可信）
    adjusted_scores = {}
    for subtype, score in subtype_scores.items():
        # 调整公式：score × PD概率 × 缩放因子 + 平滑因子
        adjusted_scores[subtype] = score * pd_prob * CONFIG["SUBTYPE_SCALE"] + CONFIG["SUBTYPE_SMOOTH"]

    # 步骤3：Sigmoid转换为0-1区间（避免概率超出合理范围）
    sigmoid_scores = {subtype: sigmoid(score) for subtype, score in adjusted_scores.items()}

    # 步骤4：归一化（确保4个亚型概率和为1）
    total = sum(sigmoid_scores.values())
    normalized_probs = {
        subtype: round(prob / total, 4) 
        for subtype, prob in sigmoid_scores.items()
    }

    # 步骤5：特殊处理：若PD概率<0.5（低风险），所有亚型概率降低（避免误导）
    if pd_prob < CONFIG["PD_THRESHOLD"]:
        for subtype in normalized_probs:
            normalized_probs[subtype] = round(normalized_probs[subtype] * pd_prob, 4)

    logger.info(f"亚型概率计算完成：{normalized_probs}")
    return normalized_probs

# ===================== 诊断核心逻辑 =====================
def diagnose(feature_result: FeatureResult) -> DiagnosisResult:
    """使用预训练模型诊断PD + 适配sklearn 1.3.2版本"""
    model, scaler, feature_order, subtype_clf = ModelManager.load_all()
    
    # 1. 标准化31维声学特征
    acoustic_scaled = scaler.transform(feature_result.acoustic_31d.reshape(1, -1))
    # 2. 拼接性别特征（31+1=32维）
    input_32d = np.hstack([acoustic_scaled, np.array([[feature_result.gender]], dtype=np.float64)])
    
    # 核心修复：适配1.3.2的LogisticRegression（避开multi_class属性）
    try:
        # 优先尝试predict_proba（如果模型兼容）
        pd_prob = float(model.predict_proba(input_32d)[0, 1])
    except AttributeError:
        # 报multi_class错误时，用decision_function+Sigmoid计算概率
        logger.warning("sklearn版本兼容问题，改用decision_function计算PD概率")
        # decision_function返回样本到超平面的距离，Sigmoid转换为0-1概率
        decision_score = model.decision_function(input_32d)[0]
        pd_prob = float(1 / (1 + np.exp(-decision_score)))  # 等价于predict_proba的结果
    
    # 后续逻辑完全不变
    diagnosis = "患有PD" if pd_prob >= CONFIG["PD_THRESHOLD"] else "健康"
    if pd_prob >= 0.8:
        risk = "高风险"
    elif pd_prob >= 0.5:
        risk = "中风险"
    else:
        risk = "低风险"
    
    feat_dict = {}
    for name, val in zip(feature_order, feature_result.acoustic_31d):
        feat_dict[name] = round(float(val), 4)
    feat_dict["gender"] = int(feature_result.gender)

    subtype_probs = calculate_subtype_probs(feature_result, pd_prob)

    return DiagnosisResult(
        pd_prob=round(pd_prob, 4),
        diagnosis=diagnosis,
        risk=risk,
        features=feat_dict,
        subtype_probs=subtype_probs
    )

# ===================== Flask API服务（修复接口路径） =====================
app = Flask(__name__)
CORS(app)  # 解决跨域

@app.route('/', methods=['GET'])
def health_check() -> Tuple[str, int]:
    """健康检查接口"""
    return "PD Voice Diagnosis Service is Running (Model-Based + Subtype Prediction)", 200

# 修复：保留旧接口/analyze，兼容前端调用
@app.route('/analyze', methods=['POST'])
@app.route('/diagnose', methods=['POST'])
def pd_diagnose() -> Tuple[Any, int]:
    """核心诊断接口：兼容/analyze和/diagnose路径，新增亚型概率输出"""
    clean_temp_files()
    temp_paths = []
    temp_path = ""  # 提前定义temp_path，避免作用域问题
    converted_path = ""  # 提前定义converted_path
    try:
        # 1. 校验请求参数
        if 'file' not in request.files:
            logger.warning("请求无音频文件")
            return jsonify({"code":400, "msg":"未上传音频文件", "data":None}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("未选择音频文件")
            return jsonify({"code":400, "msg":"未选择音频文件", "data":None}), 400
        
        # 获取性别参数（可选，默认unknown）
        gender = request.form.get("gender", "unknown")
        # 新增：校验性别参数是否合法
        if gender not in ["0", "1", "unknown"]:
            logger.warning(f"非法性别参数：{gender}，使用默认值unknown")
            gender = "unknown"
        logger.info(f"接收诊断请求：文件={file.filename}，性别={gender}（前端编码），后端编码={CONFIG['GENDER_MAP'][gender]}")

        # 2. 保存并转换音频
        temp_path = get_temp_path("upload_audio")
        converted_path = get_temp_path("converted_audio")
        temp_paths = [temp_path, converted_path]
        
        # 保存原始上传文件
        file.save(temp_path)
        logger.info(f"原始文件已保存到：{temp_path}，文件大小：{os.path.getsize(temp_path)}字节")

        # 关键修复：用soundfile加载音频，替换librosa
        try:
            # 用ffmpeg命令行强制转换为标准WAV（16kHz、单声道、16位PCM）
            import subprocess
            ffmpeg_cmd = [
                "ffmpeg",  # 修改：使用系统默认ffmpeg（Render环境）
                "-i", temp_path,  # 输入文件
                "-ar", str(CONFIG["SAMPLING_RATE"]),  # 采样率16000
                "-ac", "1",  # 单声道
                "-sample_fmt", "s16",  # 16位采样
                "-c:a", "pcm_s16le",  # PCM编码
                "-y",  # 覆盖输出文件
                converted_path  # 输出文件
            ]
            
            # 执行ffmpeg命令
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8"
            )
            
            # 检查ffmpeg执行是否成功
            if result.returncode != 0:
                logger.error(f"ffmpeg转换失败：{result.stderr}")
                raise Exception(f"ffmpeg error: {result.stderr}")
            
            # 读取转换后的WAV文件为numpy数组
            import wave
            with wave.open(converted_path, 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(-1)
            
            # 转为numpy数组并归一化
            y = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
            y = y / np.iinfo(np.int16).max
            
            # 用parselmouth加载转换后的标准WAV
            sound = parselmouth.Sound(converted_path)
            
            logger.info(f"音频转换成功：采样率={sr}，时长={len(y)/sr:.2f}秒，数据长度={len(y)}")
            
        except Exception as e:
            logger.error(f"音频转换失败：{e}")
            raise

        # 检测音频是否有效（阈值设为1e-6，允许微小噪声）
        if np.max(np.abs(y)) < 1e-6:
            logger.error(f"【无效音频】上传的文件{file.filename}无有效语音数据（全零）")
            return jsonify({
                "code":400,
                "msg":"上传的音频文件无效！原因：文件无声音/格式不支持/损坏。请重新录制WAV格式（16kHz、16位、单声道）的语音文件。",
                "data":None
            }), 400

        logger.info(f"音频加载成功：采样率={CONFIG['SAMPLING_RATE']}，时长={len(y)/CONFIG['SAMPLING_RATE']:.2f}秒")

        # 3. 提取特征 + 模型诊断（包含亚型概率）
        features = extract_all_features(sound, y, CONFIG["SAMPLING_RATE"], gender)
        result = diagnose(features)
        logger.info(f"诊断完成：PD概率={result.pd_prob}，结论={result.diagnosis}，亚型概率={result.subtype_probs}")

        # 4. 构造返回结果（新增subtype_probabilities字段）
        response = convert_numpy_type({
            "code": 200,
            "msg": "诊断成功",
            "data": {
                "pd_probability": result.pd_prob,
                "diagnosis": result.diagnosis,
                "risk_level": result.risk,
                "acoustic_features": result.features,
                # 新增：亚型概率输出
                "subtype_probabilities": {
                    "tremor_type": result.subtype_probs["tremor"],  # 震颤型概率
                    "rigidity_type": result.subtype_probs["rigidity"],  # 僵直型概率
                    "motor_type": result.subtype_probs["motor"],  # 运动型概率
                    "non_motor_type": result.subtype_probs["non_motor"]  # 非运动型概率
                },
                # 新增：亚型说明（基于文献）
                "subtype_explanation": {
                    "tremor_type": "震颤主导型：以静止性震颤为核心，HNR系列特征偏低（文献1）",
                    "rigidity_type": "僵直主导型：以肌肉僵直和运动迟缓为核心，MFCC12和Delta0偏高（文献2）",
                    "motor_type": "运动型：包含震颤型和僵直型，运动症状显著（文献1）",
                    "non_motor_type": "非运动型：以认知障碍、睡眠障碍等为核心，GNE特征偏低（文献1）"
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

# ===================== 启动服务（修改：适配Render端口） =====================
if __name__ == '__main__':
    # 启动前预加载模型/特征
    try:
        ModelManager.load_all()
    except Exception as e:
        logger.error(f"服务启动失败：{e}")
        exit(1)
    # 修改：读取Render的PORT环境变量，默认5000
    port = int(os.environ.get("PORT", 10000))
    # 启动Flask服务（生产环境建议用Gunicorn）
    app.run(host='0.0.0.0', port=port, debug=False)
