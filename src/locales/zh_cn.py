pic_text2zh_cn = {
    "Risk Level": "风险等级",
    "Score": "风险评分",
    "Reason": "原因",
    "Bird's-eye view": "俯视图",
    "AUDIO WARNING: Pedestrian crossing risk detected!": "语音告警：检测到行人过街风险！",
    "Experimental real-time webcam mode": "实时摄像头演示模式（实验性）",
    "ROI-based risk scoring requires camera-specific ROI calibration.": "基于 ROI 的风险评分需要针对摄像头单独标定 ROI。",
    "Crosswalk Risk Warning - Webcam Demo": "斑马线风险预警 - 摄像头演示",
}

risk_level2zh_cn = {
    "DANGER": "极高",
    "HIGH": "高",
    "MEDIUM": "中",
    "LOW": "低"
}

reason2zh_cn = {
    "No pedestrian-related risk detected": "未检测到风险",
    "No pedestrian in crosswalk or waiting zone": "没有人在斑马线内",
    "Pedestrian waiting near unsignalized crosswalk": "有人在无信号灯控制的斑马线旁等候",
    "Pedestrian inside unsignalized crosswalk": "有人经过无信号灯控制的斑马线",
    "Pedestrian in {crosswalk} and vehicle critically close":
        "行人在{crosswalk}，车辆距离极近",
    "Pedestrian in {crosswalk} and short TTC-like risk detected":
        "行人在{crosswalk}，检测到极短的类碰撞时间风险",
    "Pedestrian in {crosswalk}, vehicle approaching, and distance decreasing":
        "行人在{crosswalk}，车辆驶近且距离持续缩短",
    "Pedestrian in {crosswalk} and high surrogate risk score detected":
        "行人在{crosswalk}，替代风险评分偏高",
    "Pedestrian in {crosswalk} and vehicle close":
        "行人在{crosswalk}，车辆距离较近",
    "Pedestrian in {crosswalk} and vehicle in approach zone":
        "行人在{crosswalk}，车辆进入接近区",
    "Pedestrian in {crosswalk} and elevated surrogate risk score":
        "行人在{crosswalk}，替代风险评分升高",
    "Pedestrian in {crosswalk} with moderate surrogate risk score":
        "行人在{crosswalk}，替代风险评分中等",
}

crosswalk_name2zh_cn = {
    "main_crosswalk": "主斑马线",
    "secondary_crosswalk": "副斑马线",
    "crosswalk": "斑马线",
}

roi_label2zh_cn = {
    "main crosswalk": "主斑马线",
    "main approach": "主路车辆接近区",
    "waiting zone": "行人等候区",
    "secondary crosswalk": "副斑马线",
    "secondary approach": "副路车辆接近区",
    "main": "主",
    "secondary": "副",
    "main app": "主接近区",
    "sec app": "副接近区",
}

class_name2zh_cn = {
    "person": "行人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "truck": "卡车",
}

chart_text2zh_cn = {
    "Image-space Surrogate Risk Score Timeline": "图像空间替代风险评分时间线",
    "Time (seconds)": "时间（秒）",
    "Risk score (0–100)": "风险评分（0–100）",
    "MEDIUM threshold": "中风险阈值",
    "HIGH threshold": "高风险阈值",
    "DANGER threshold": "极高风险阈值",
    "HIGH samples": "高风险样本",
    "DANGER samples": "极高风险样本",
}

LOCALE = {
    "suffix": "zh_cn",
    "matplotlib_font": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "pic_text": pic_text2zh_cn,
    "risk_level": risk_level2zh_cn,
    "reason": reason2zh_cn,
    "crosswalk_name": crosswalk_name2zh_cn,
    "roi_label": roi_label2zh_cn,
    "class_name": class_name2zh_cn,
    "chart_text": chart_text2zh_cn,
}
